"""
Detect whether a resource is structured enough for computation and normalize it
into a machine-usable object (dataframe / records) for the Technical Agent.

Pipeline: download → detect structure → parse into records → build technical context.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
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
    total_rows_in_file: Optional[int] = None  # estimate for full resource, if available

    def _find_sort_column(self) -> Optional[str]:
        """Return the numeric column with the highest max value across a sample."""
        best_col: Optional[str] = None
        best_max: float = 0.0
        _sample = self.records[:500]
        for col in self.columns:
            vals: list[float] = []
            for r in _sample:
                raw = r.get(col)
                if raw is None:
                    continue
                try:
                    v = float(str(raw).replace(",", ".").strip())
                    vals.append(v)
                except (ValueError, TypeError):
                    pass
            if vals and max(vals) > best_max:
                best_max = max(vals)
                best_col = col
        return best_col

    def _sorted_preview(self, max_rows: int, best_col: Optional[str]) -> List[Dict[str, Any]]:
        """Return records sorted by best_col descending, truncated to max_rows."""
        if len(self.records) <= max_rows:
            return self.records
        if best_col:
            def _key(r: Dict[str, Any]) -> float:
                raw = r.get(best_col)
                if raw is None:
                    return 0.0
                try:
                    return float(str(raw).replace(",", ".").strip())
                except (ValueError, TypeError):
                    return 0.0
            return sorted(self.records, key=_key, reverse=True)[:max_rows]
        return self.records[:max_rows]

    def to_preview_json(self, max_rows: int = DEFAULT_PREVIEW_ROWS) -> str:
        """
        Return up to max_rows records as JSON. When truncating, sort by the
        numeric column with the highest maximum value (descending) so that
        extreme-value rows appear in the preview rather than alphabetical head.
        """
        best_col = self._find_sort_column()
        preview = self._sorted_preview(max_rows, best_col)
        return json.dumps(preview, ensure_ascii=False, indent=2)

    def to_compact_preview_table(
        self, max_rows: int = DEFAULT_PREVIEW_ROWS, max_cols: int = 8
    ) -> str:
        """
        Compact pipe-separated table of the top max_rows records using key columns only.
        Far more token-efficient than full JSON when records have many columns.
        """
        best_col = self._find_sort_column()
        preview = self._sorted_preview(max_rows, best_col)
        if not preview:
            return "(empty)"

        all_cols = list(preview[0].keys()) if preview else self.columns

        if len(all_cols) <= max_cols:
            cols = all_cols
        else:
            priority: list[tuple[int, int, str]] = []
            for i, col in enumerate(all_cols):
                key = col.lower()
                if col == best_col:
                    p = 0
                elif any(k in key for k in ("lib", "nom", "name", "label", "intitule")):
                    p = 1
                elif any(k in key for k in ("codgeo", "code", "geo", "id")):
                    p = 2
                elif any(k in key for k in ("dep", "dept", "reg")):
                    p = 3
                elif any(k in key for k in ("pop", "total", "mun", "municipal")):
                    p = 4
                else:
                    p = 99
                priority.append((p, i, col))
            priority.sort(key=lambda x: (x[0], x[1]))
            cols = [col for _, _, col in priority[:max_cols]]

        def _fmt(v: Any) -> str:
            if v is None:
                return ""
            # Only convert float x.0 -> int for actual float/int types, not strings
            # (string "06088" must keep leading zero for geographic codes).
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v)[:60]

        rows_out = [" | ".join(cols)]
        for row in preview:
            rows_out.append(" | ".join(_fmt(row.get(col)) for col in cols))
        return "\n".join(rows_out)



def _extension_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return ext


def _count_file_lines(path: str) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def estimate_total_rows(path: str, resource: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """
    Estimate total data rows in a resource file (best effort, no full parse).
    For CSV/TSV subtracts one line for a header when present.
    """
    ext = _extension_from_path(path)
    fmt = _resource_format(resource)
    if not ext and fmt:
        ext = f".{fmt.split('/')[-1]}" if "/" not in fmt else f".{fmt}"
    try:
        if ext in (".csv", ".tsv"):
            lines = _count_file_lines(path)
            return max(0, lines - 1) if lines > 0 else 0
        if ext in (".jsonl", ".ndjson"):
            return _count_file_lines(path)
        if ext == ".json":
            return _estimate_json_record_count(path)
        if ext in (".xlsx", ".xls"):
            return _estimate_excel_row_count(path, ext)
    except Exception as e:
        logger.debug("estimate_total_rows failed for %s: %s", path, e)
    return None


def _estimate_json_record_count(path: str) -> Optional[int]:
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size > 10 * 1024 * 1024:
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        features = data.get("features")
        if isinstance(features, list):
            return len(features)
    return None


def _estimate_excel_row_count(path: str, ext: str) -> Optional[int]:
    if ext == ".xlsx":
        try:
            import openpyxl
        except ImportError:
            return None
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.active
            max_row = ws.max_row or 0
        finally:
            wb.close()
        return max(0, max_row - 1) if max_row > 0 else 0
    return None


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
        # Auto-detect delimiter (comma, semicolon, tab, pipe, etc.) then cap columns.
        try:
            header_df = pd.read_csv(
                path, nrows=0, sep=None, engine="python",
                encoding="utf-8", on_bad_lines="skip",
            )
            ncols = len(header_df.columns)
            detected_sep = header_df._constructor_sliced  # not available; use sniffer below
        except Exception:
            header_df = None
            ncols = None
        # Determine actual separator from the first line.
        detected_sep = ","
        try:
            import csv as _csv
            with open(path, "r", encoding="utf-8", errors="replace") as _f:
                sample = _f.read(4096)
            detected_sep = _csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except Exception:
            pass
        try:
            header_df2 = pd.read_csv(
                path, nrows=0, sep=detected_sep,
                encoding="utf-8", on_bad_lines="skip",
            )
            ncols = len(header_df2.columns)
            usecols = list(range(min(MAX_TABULAR_COLUMNS, ncols)))
            read_csv_kw["usecols"] = usecols
        except Exception as e:
            logger.debug("Could not get CSV header for usecols: %s", e)
        read_csv_kw["sep"] = detected_sep
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


def _is_zip(path: str) -> bool:
    """Return True if the file starts with a ZIP magic header (PK\\x03\\x04)."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _best_structured_member(zip_path: str) -> Optional[str]:
    """
    Return the name of the most useful structured file inside a ZIP.
    Prefers CSV/TSV by size descending, then JSON variants, then Excel.
    Returns None if no structured member found.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            candidates = []
            for m in zf.infolist():
                if m.is_dir():
                    continue
                ext = os.path.splitext(m.filename.lower())[1]
                if ext in STRUCTURED_EXTENSIONS:
                    if ext in (".csv", ".tsv"):
                        priority = 0
                    elif ext in (".json", ".jsonl", ".ndjson", ".geojson"):
                        priority = 1
                    else:
                        priority = 2
                    candidates.append((priority, -m.file_size, m.filename))
            if not candidates:
                return None
            candidates.sort()
            return candidates[0][2]
    except Exception as e:
        logger.debug("_best_structured_member failed for %s: %s", zip_path, e)
        return None


def _extract_member(zip_path: str, member_name: str, dest_dir: str) -> Optional[str]:
    """Extract one member from a ZIP; return its path on disk, or None on failure."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(member_name, dest_dir)
        extracted = os.path.join(dest_dir, member_name)
        return extracted if os.path.exists(extracted) else None
    except Exception as e:
        logger.debug("_extract_member failed for %s/%s: %s", zip_path, member_name, e)
        return None


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
    ZIP archives are transparently unpacked: the largest structured member (CSV,
    JSON, XLSX, …) is extracted and parsed in place of the archive itself.
    """
    kind = detect_structure(path, resource)

    # ZIP handling: peek inside and use the best structured member.
    if kind == "unsuitable" and (_extension_from_path(path) == ".zip" or _is_zip(path)):
        member = _best_structured_member(path)
        if member:
            dest_dir = os.path.dirname(path) or "."
            extracted = _extract_member(path, member, dest_dir)
            if extracted:
                logger.info("ZIP resource: using extracted member %r", member)
                kind = detect_structure(extracted)
                if kind != "unsuitable":
                    path = extracted

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
    parsed.total_rows_in_file = estimate_total_rows(path, resource)
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


def build_resource_context(
    *,
    dataset: Dict[str, Any],
    resource: Dict[str, Any] | None,
    url: str,
    parsed: ParsedData | None = None,
    unstructured_text: str | None = None,
    resource_selector_reasoning: str = "",
    dataset_selector_reasoning: str = "",
    max_rows_loaded: int = 0,
) -> str:
    """
    Human-readable context for the technical RLM (metadata + schema; not the full records).
    Records are passed separately as the REPL variable `records`.
    """
    title = (dataset.get("title") or dataset.get("name") or "Unknown").strip()
    org_raw = dataset.get("organization")
    if isinstance(org_raw, dict):
        org = (org_raw.get("name") or org_raw.get("title") or "").strip()
    else:
        org = (org_raw or "").strip() if isinstance(org_raw, str) else ""

    lines = [
        f"Dataset: {title}",
    ]
    if org:
        lines.append(f"Organization: {org}")
    ds_desc = (dataset.get("description") or "").strip()
    if ds_desc:
        lines.append(f"Dataset description: {ds_desc[:800]}{'…' if len(ds_desc) > 800 else ''}")

    if resource:
        res_title = (resource.get("title") or "").strip()
        if res_title:
            lines.append(f"Resource title: {res_title}")
        fmt = (resource.get("format") or "").strip()
        if fmt:
            lines.append(f"Resource format: {fmt}")
        mime = (resource.get("mime") or "").strip()
        if mime:
            lines.append(f"Resource MIME: {mime}")
        size = resource.get("size")
        if size is not None:
            lines.append(f"Resource size (bytes): {size}")

    lines.append(f"Resource URL: {url[:300]}{'…' if len(url) > 300 else ''}")

    if dataset_selector_reasoning:
        lines.append(f"Dataset selector reasoning: {dataset_selector_reasoning.strip()}")
    if resource_selector_reasoning:
        lines.append(f"Resource selector reasoning: {resource_selector_reasoning.strip()}")

    if parsed and parsed.records:
        lines.append(f"Rows loaded into REPL variable `records`: {parsed.row_count}")
        if max_rows_loaded and parsed.row_count >= max_rows_loaded:
            lines.append(
                f"(Capped at {max_rows_loaded} rows from file; file may contain more.)"
            )
        lines.append(f"Columns / schema: {parsed.schema_summary}")
        lines.append(
            "Use Python on `records` (list of dicts). Example: "
            "`import pandas as pd; df = pd.DataFrame(records)` — pandas may not be installed; "
            "use plain Python if import fails."
        )
        lines.append(
            "Preview (top 100 rows sorted by largest numeric column desc;"
            " use directly if REPL is unavailable; key columns shown):\n"
            "(codgeo/name/population columns selected for readability)"
        )
        lines.append(parsed.to_compact_preview_table(max_rows=100))
    elif unstructured_text:
        lines.append(
            "Structured parse unavailable; extracted text snippet is included below "
            "(full text may be truncated)."
        )
        lines.append(unstructured_text[:20_000])
        if len(unstructured_text) > 20_000:
            lines.append("\n[Extracted text truncated for resource_context.]")
    else:
        lines.append("No rows were loaded into `records` (empty list).")

    return "\n".join(lines)
