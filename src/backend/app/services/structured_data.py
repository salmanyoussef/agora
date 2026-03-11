"""
Detect whether a resource is structured enough for computation and normalize it
into a machine-usable object (dataframe / records) for the Technical Agent.

Pipeline: download → detect structure → parse into records → build technical context.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Formats we consider structured enough for computation (tabular or record-based).
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
RECORDS_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".geojson"}
STRUCTURED_EXTENSIONS = TABULAR_EXTENSIONS | RECORDS_EXTENSIONS

# Max rows to load per resource (avoid huge datasets in memory).
DEFAULT_MAX_ROWS = 10_000
# Max columns to keep for tabular data (avoids OOM and timeouts on very wide CSVs).
MAX_TABULAR_COLUMNS = 150
# Max file size (bytes) for JSON/GeoJSON before we refuse to load (avoids OOM on huge geo files).
MAX_JSON_FILE_BYTES = 25 * 1024 * 1024  # 25 MB (full load into memory).
# Max file size (bytes) for Excel (.xlsx/.xls); openpyxl can use ~50× file size in RAM—keep conservative.
MAX_EXCEL_FILE_BYTES = 25 * 1024 * 1024  # 25 MB
# Max file size (bytes) for CSV/TSV; we only load first 150 cols + max_rows so file can be larger on disk.
MAX_CSV_FILE_BYTES = 100 * 1024 * 1024  # 100 MB
# Rows to include in technical context preview for the LLM.
DEFAULT_PREVIEW_ROWS = 50

StructureKind = Literal["tabular", "records", "unsuitable"]


@dataclass
class ParsedData:
    """Normalized structured data from a single resource."""

    records: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    format: str  # e.g. "csv", "jsonl", "geojson"
    schema_summary: str  # human-readable column names + dtypes
    resource_id: str = ""
    metadata: str = ""

    def to_preview_json(self, max_rows: int = DEFAULT_PREVIEW_ROWS) -> str:
        """First N records as JSON for LLM context."""
        preview = self.records[:max_rows]
        return json.dumps(preview, ensure_ascii=False, indent=2)


def _extension_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return ext


def _resource_format(resource: Optional[Dict[str, Any]]) -> Optional[str]:
    if not resource:
        return None
    fmt = (resource.get("format") or "").strip().lower()
    if fmt:
        return fmt
    mime = (resource.get("mime") or "").strip().lower()
    if mime:
        return mime.split(";")[0].strip()
    return None


def detect_structure(
    path: str,
    resource: Optional[Dict[str, Any]] = None,
) -> StructureKind:
    """
    Detect whether the file at path is structured enough for computation.

    Returns:
        "tabular" for CSV, TSV, XLS, XLSX.
        "records" for JSON (array of objects), JSONL, GeoJSON (FeatureCollection).
        "unsuitable" otherwise.
    """
    ext = _extension_from_path(path)
    if ext in TABULAR_EXTENSIONS:
        return "tabular"
    if ext in RECORDS_EXTENSIONS:
        return "records"
    # Optional: infer from resource metadata when file has no / wrong extension
    fmt = _resource_format(resource)
    if fmt:
        if fmt in ("csv", "tsv", "xlsx", "xls", "text/csv", "text/tab-separated-values"):
            return "tabular"
        if fmt in ("json", "jsonl", "ndjson", "geojson", "application/json", "application/geo+json"):
            return "records"
    return "unsuitable"


def _parse_tabular(
    path: str,
    max_rows: int,
    ext: str,
) -> ParsedData:
    try:
        file_size = os.path.getsize(path)
    except OSError:
        file_size = 0

    if ext in (".xlsx", ".xls"):
        if file_size > MAX_EXCEL_FILE_BYTES:
            logger.warning(
                "Excel file too large to parse safely: %s (%d MB); skipping to avoid OOM",
                path,
                file_size // (1024 * 1024),
            )
            return ParsedData(
                records=[],
                columns=[],
                row_count=0,
                format=ext.lstrip("."),
                schema_summary=f"(Excel file too large: {file_size // (1024*1024)} MB; skipped)",
            )
    elif ext in (".csv", ".tsv"):
        if file_size > MAX_CSV_FILE_BYTES:
            logger.warning(
                "CSV/TSV file too large to parse safely: %s (%d MB); skipping to avoid OOM",
                path,
                file_size // (1024 * 1024),
            )
            return ParsedData(
                records=[],
                columns=[],
                row_count=0,
                format=ext.lstrip("."),
                schema_summary=f"(CSV/TSV file too large: {file_size // (1024*1024)} MB; skipped)",
            )

    read_csv_kw: dict = {
        "nrows": max_rows,
        "encoding": "utf-8",
        "on_bad_lines": "skip",
        "low_memory": False,
    }
    if ext == ".csv":
        # Read only first N columns from the start so we never load 700+ columns into memory.
        try:
            header_df = pd.read_csv(path, nrows=0, encoding="utf-8", on_bad_lines="skip")
            ncols = len(header_df.columns)
            usecols = list(range(min(MAX_TABULAR_COLUMNS, ncols)))
            read_csv_kw["usecols"] = usecols
        except Exception as e:
            logger.debug("Could not get CSV header for usecols, reading all columns: %s", e)
        df = pd.read_csv(path, **read_csv_kw)
    elif ext == ".tsv":
        try:
            header_df = pd.read_csv(path, nrows=0, sep="\t", encoding="utf-8", on_bad_lines="skip")
            ncols = len(header_df.columns)
            usecols = list(range(min(MAX_TABULAR_COLUMNS, ncols)))
            read_csv_kw["usecols"] = usecols
        except Exception as e:
            logger.debug("Could not get TSV header for usecols, reading all columns: %s", e)
        df = pd.read_csv(path, sep="\t", **read_csv_kw)
    elif ext == ".xlsx":
        df = pd.read_excel(path, nrows=max_rows, engine="openpyxl")
    elif ext == ".xls":
        try:
            df = pd.read_excel(path, nrows=max_rows, engine="xlrd")
        except ImportError:
            df = pd.read_excel(path, nrows=max_rows)
        except Exception as e:
            logger.warning("read_excel .xls failed for %s: %s", path, e)
            raise ValueError(f"Cannot read .xls file: {path}") from e
    else:
        df = pd.read_csv(path, **read_csv_kw)

    df = df.dropna(axis=1, how="all")
    columns = [str(c) for c in df.columns]
    # Fallback if usecols wasn't used (e.g. other tabular ext): cap columns after read.
    if len(columns) > MAX_TABULAR_COLUMNS:
        logger.info(
            "Tabular resource has %d columns; keeping first %d to avoid OOM/slowness",
            len(columns),
            MAX_TABULAR_COLUMNS,
        )
        df = df.iloc[:, :MAX_TABULAR_COLUMNS]
        columns = [str(c) for c in df.columns]
    records = df.to_dict(orient="records")
    # Coerce non-serializable types for JSON preview
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
            elif hasattr(v, "item"):
                try:
                    r[k] = v.item()
                except (ValueError, AttributeError):
                    r[k] = str(v)
            elif hasattr(v, "isoformat"):
                r[k] = v.isoformat()

    schema_parts = [f"{c} ({df.dtypes[c].name})" for c in columns]
    schema_summary = ", ".join(schema_parts) if schema_parts else "(no columns)"

    return ParsedData(
        records=records,
        columns=columns,
        row_count=len(records),
        format=ext.lstrip("."),
        schema_summary=schema_summary,
    )


def _normalize_json_records(data: Any) -> List[Dict[str, Any]]:
    """Turn JSON/GeoJSON into a list of flat(ish) record dicts."""
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, dict):
                out.append(_flatten_value(item))
            else:
                out.append({"_value": item})
        return out
    if isinstance(data, dict):
        if "features" in data and isinstance(data["features"], list):
            return [_flatten_value(f.get("properties") or f) for f in data["features"]]
        if "data" in data and isinstance(data["data"], list):
            return [_flatten_value(x) for x in data["data"]]
        return [_flatten_value(data)]
    return []


def _flatten_value(v: Any) -> Dict[str, Any]:
    """One-level flatten for display; nested dicts are JSON-serialized."""
    if not isinstance(v, dict):
        return {"_value": v}
    out: Dict[str, Any] = {}
    for key, val in v.items():
        if isinstance(val, (dict, list)) and not isinstance(val, (str, bytes)):
            try:
                out[str(key)] = json.dumps(val, ensure_ascii=False)[:500]
            except (TypeError, ValueError):
                out[str(key)] = str(val)[:500]
        else:
            if hasattr(val, "isoformat"):
                out[str(key)] = val.isoformat()
            elif pd.isna(val) if hasattr(pd, "isna") else (val is None):
                out[str(key)] = None
            else:
                out[str(key)] = val
    return out


def _parse_records(
    path: str,
    max_rows: int,
    ext: str,
) -> ParsedData:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if ext in (".jsonl", ".ndjson"):
            records = []
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(_flatten_value(obj) if isinstance(obj, dict) else {"_value": obj})
                except json.JSONDecodeError:
                    continue
            if not records:
                return ParsedData(
                    records=[],
                    columns=[],
                    row_count=0,
                    format=ext.lstrip("."),
                    schema_summary="(empty or invalid JSONL)",
                )
            columns = list(records[0].keys()) if records else []
            schema_summary = ", ".join(columns) if columns else "(no keys)"
            return ParsedData(
                records=records,
                columns=columns,
                row_count=len(records),
                format="jsonl",
                schema_summary=schema_summary,
            )

        # JSON/GeoJSON: load entire file into memory — skip if too large (avoids OOM on big geo files)
        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = 0
        if file_size > MAX_JSON_FILE_BYTES:
            logger.warning(
                "JSON/GeoJSON file too large to parse safely: %s (%d MB); skipping to avoid OOM",
                path,
                file_size // (1024 * 1024),
            )
            return ParsedData(
                records=[],
                columns=[],
                row_count=0,
                format=ext.lstrip("."),
                schema_summary=f"(file too large: {file_size // (1024*1024)} MB; skipped)",
            )

        raw = json.load(f)

    records = _normalize_json_records(raw)[:max_rows]
    if not records:
        return ParsedData(
            records=[],
            columns=[],
            row_count=0,
            format=ext.lstrip("."),
            schema_summary="(empty or unsupported structure)",
        )
    columns = list(records[0].keys()) if records else []
    schema_summary = ", ".join(columns) if columns else "(no keys)"
    return ParsedData(
        records=records,
        columns=columns,
        row_count=len(records),
        format=ext.lstrip("."),
        schema_summary=schema_summary,
    )


def parse_into_records(
    path: str,
    resource: Optional[Dict[str, Any]] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    resource_id: str = "",
    metadata: str = "",
) -> Optional[ParsedData]:
    """
    If the resource is structured, parse it into a normalized list of records
    plus schema summary. Returns None if structure is unsuitable or parsing fails.
    """
    kind = detect_structure(path, resource)
    if kind == "unsuitable":
        logger.debug("parse_into_records: unsuitable structure for %s", path)
        return None

    ext = _extension_from_path(path)
    try:
        if kind == "tabular":
            parsed = _parse_tabular(path, max_rows, ext)
        else:
            parsed = _parse_records(path, max_rows, ext)
    except Exception as e:
        logger.warning("parse_into_records failed for %s: %s", path, e)
        return None

    parsed.resource_id = resource_id
    parsed.metadata = metadata
    return parsed


def build_technical_context(
    parsed_list: List[ParsedData],
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    unstructured_blocks: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Build a single technical context string for the RLM: schema + preview for
    each parsed resource, plus optional unstructured (extracted text) blocks from
    resources that were not machine-parseable (e.g. PDF, DOCX). The model can
    use both to answer the subquery.
    """
    parts: List[str] = []
    for i, p in enumerate(parsed_list):
        block = [
            f"## Structured resource {i + 1}",
            f"Format: {p.format}",
            f"Rows: {p.row_count}",
            f"Columns / schema: {p.schema_summary}",
        ]
        if p.metadata:
            block.append(f"Metadata: {p.metadata}")
        block.append("Preview (first {} rows):".format(min(preview_rows, p.row_count)))
        block.append(p.to_preview_json(max_rows=preview_rows))
        parts.append("\n".join(block))

    if unstructured_blocks:
        unstructured_parts = ["## Unstructured resources (extracted text)"]
        unstructured_parts.append(
            "The following resources were not tabular/record-based; their text was "
            "extracted with the same pipeline as the general (RAG) agent so you can "
            "still use them to inform your answer."
        )
        for b in unstructured_blocks:
            meta = (b.get("metadata") or "").strip()
            content = (b.get("content") or "").strip()
            if not content:
                continue
            if meta:
                unstructured_parts.append(f"\n[Source: {meta}]\n{content}")
            else:
                unstructured_parts.append(f"\n{content}")
        parts.append("\n".join(unstructured_parts))

    if not parts:
        return "(No structured or unstructured data available.)"
    return "\n\n---\n\n".join(parts)
