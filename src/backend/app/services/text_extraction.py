"""
Download data.gouv resources and extract text for RAG.

Download: uses resource metadata (format/mime) from the API when available to set
the file extension, then Content-Disposition, URL path, Content-Type, and finally
content-based detection (filetype) so saved files have a proper extension.

Extraction: with unstructured[all-docs] we attempt all types unstructured supports
(.doc, .docx, .pdf, .xls, .xlsx, .csv, .tsv, .ppt, .pptx, .odt, .epub, .rtf, .rst,
.md, .org, .html, .xml, .txt, .eml, .msg, and images). GTFS ZIP uses custom logic
(static schedule .txt/.csv inside the archive). Other ZIP-based formats (e.g. XLSX,
DOCX, EPUB, KMZ) are handled by saving with the correct extension from resource
metadata so unstructured or fallbacks process them as single files; we do not
unpack generic ZIPs of mixed documents here.

Meaningful output:
- Documents (PDF, Word, etc.) and spreadsheets: text is extracted and useful for RAG.
- Images: meaningful only if OCR is available (e.g. tesseract; unstructured may use
  it when the image extra is installed). Without OCR, image partition may return little.
- Some formats (e.g. .odt, .rtf, .epub) may need system tools (pandoc, LibreOffice);
  if missing, extraction can fail and we fall back to a short "[Unsupported...]" message.

Weight of unstructured[all-docs]: pulls in many deps (e.g. PyTorch, opencv, pdfminer,
pikepdf, python-pptx, python-docx, openpyxl, pypandoc, unstructured-inference). Install
size and memory use are significant. For a lighter setup use a subset, e.g.
unstructured[pdf,docx,xlsx,csv,html,xml,md,ppt,pptx,tsv,rtf].
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import time
import zipfile
from typing import Any, Dict, List, Optional, Union
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from pypdf import PdfReader

logger = logging.getLogger(__name__)
DEFAULT_DOWNLOAD_RETRY_ATTEMPTS = 3
DEFAULT_DOWNLOAD_RETRY_BACKOFF_S = 1.5

# Map data.gouv resource "format" / mime to file extension (lowercase, with leading dot).
# Used when the server does not provide a filename so we still save with the right extension.
FORMAT_MIME_TO_EXT: Dict[str, str] = {
    "csv": ".csv",
    "application/csv": ".csv",
    "text/csv": ".csv",
    "json": ".json",
    "application/json": ".json",
    "geojson": ".geojson",
    "application/geo+json": ".geojson",
    "application/vnd.geo+json": ".geojson",
    "jsonl": ".jsonl",
    "ndjson": ".ndjson",
    "application/jsonlines": ".jsonl",
    "application/x-ndjson": ".ndjson",
    "pdf": ".pdf",
    "application/pdf": ".pdf",
    "xlsx": ".xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "xls": ".xls",
    "application/vnd.ms-excel": ".xls",
    "docx": ".docx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "doc": ".doc",
    "application/msword": ".doc",
    "html": ".html",
    "text/html": ".html",
    "htm": ".htm",
    "xml": ".xml",
    "application/xml": ".xml",
    "text/xml": ".xml",
    "txt": ".txt",
    "text/plain": ".txt",
    "md": ".md",
    "text/markdown": ".md",
    "application/zip": ".zip",
    "zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "protobuf": ".pb",
    "application/x-protobuf": ".pb",
    "application/octet-stream": "",
    "tsv": ".tsv",
    "text/tab-separated-values": ".tsv",
    "ppt": ".ppt",
    "pptx": ".pptx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "odt": ".odt",
    "application/vnd.oasis.opendocument.text": ".odt",
    "epub": ".epub",
    "application/epub+zip": ".epub",
    "rtf": ".rtf",
    "application/rtf": ".rtf",
    "text/rtf": ".rtf",
    "rst": ".rst",
    "text/x-rst": ".rst",
    "org": ".org",
    "eml": ".eml",
    "message/rfc822": ".eml",
    "msg": ".msg",
    "application/vnd.ms-outlook": ".msg",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/heic": ".heic",
}

_CD_FILENAME_RE = re.compile(r'filename\*?=(?:UTF-8\'?\')?"?([^";]+)"?')


def _safe_filename(name: str) -> str:
    name = unquote(name)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return name or "resource"


def _filename_from_content_disposition(cd: str) -> Optional[str]:
    if not cd:
        return None
    m = _CD_FILENAME_RE.search(cd)
    return m.group(1).strip() if m else None


def _filename_from_url(url: str) -> Optional[str]:
    path = urlparse(url).path
    if not path:
        return None
    base = os.path.basename(path.rstrip("/"))
    return base or None


def _extension_from_resource(resource: Optional[Dict[str, Any]]) -> Optional[str]:
    """Infer file extension from data.gouv resource metadata (format, mime, extras)."""
    if not resource:
        return None
    fmt = (resource.get("format") or "").strip().lower()
    if fmt and fmt in FORMAT_MIME_TO_EXT:
        return FORMAT_MIME_TO_EXT[fmt]
    mime = (resource.get("mime") or "").strip().lower()
    if mime:
        # exact match
        if mime in FORMAT_MIME_TO_EXT:
            return FORMAT_MIME_TO_EXT[mime]
        # without params (e.g. "application/vnd...; charset=utf-8")
        base_mime = mime.split(";")[0].strip()
        if base_mime in FORMAT_MIME_TO_EXT:
            return FORMAT_MIME_TO_EXT[base_mime]
    extras = resource.get("extras") or {}
    analysis_mime = (extras.get("analysis:mime-type") or "").strip().lower()
    if analysis_mime and analysis_mime in FORMAT_MIME_TO_EXT:
        return FORMAT_MIME_TO_EXT[analysis_mime]
    return None


def _guess_extension_from_content(path: str) -> Optional[str]:
    """Guess extension from file content (magic bytes). Returns e.g. '.pdf' or None."""
    try:
        import filetype
    except ImportError:
        return None
    kind = filetype.guess(path)
    if kind is None:
        return None
    ext = kind.extension
    return f".{ext}" if ext and not ext.startswith(".") else (f".{ext}" if ext else None)


def download_file(
    url: str,
    out_dir: str,
    resource: Optional[Dict[str, Any]] = None,
    timeout_s: int = 60,
    retry_attempts: int = DEFAULT_DOWNLOAD_RETRY_ATTEMPTS,
    retry_backoff_s: float = DEFAULT_DOWNLOAD_RETRY_BACKOFF_S,
) -> str:
    """
    Download URL to out_dir; return local path. Retries transient request errors.

    Filename/extension is chosen in this order:
    1. Content-Disposition header (if present)
    2. URL path
    3. data.gouv resource metadata (format/mime) when resource= is provided
    4. Content-Type header (mimetypes)
    5. After download: content-based guess (filetype) if still no extension
    """
    os.makedirs(out_dir, exist_ok=True)

    last_exc: Exception | None = None
    path: Optional[str] = None
    max_attempts = max(1, retry_attempts)
    base_backoff = max(0.1, retry_backoff_s)

    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout_s, allow_redirects=True) as r:
                r.raise_for_status()

                cd = r.headers.get("Content-Disposition", "")
                name = _filename_from_content_disposition(cd)
                if not name:
                    name = _filename_from_url(r.url)
                if not name:
                    name = _filename_from_url(url)

                name = _safe_filename(name or "resource")

                root, ext = os.path.splitext(name)
                if not ext:
                    # Prefer extension from data.gouv resource metadata
                    res_ext = _extension_from_resource(resource)
                    if res_ext:
                        name = root + res_ext
                    else:
                        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                        guessed_ext = mimetypes.guess_extension(ctype)
                        if guessed_ext:
                            name = root + guessed_ext

                path_candidate = os.path.join(out_dir, name)
                if os.path.exists(path_candidate):
                    base, ext2 = os.path.splitext(path_candidate)
                    i = 2
                    while os.path.exists(f"{base}__{i}{ext2}"):
                        i += 1
                    path_candidate = f"{base}__{i}{ext2}"

                with open(path_candidate, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                path = path_candidate

                # If we still have no extension, guess from content (magic bytes)
                root_final, ext_final = os.path.splitext(path)
                if not ext_final or ext_final.lower() in (".bin", ".dat", ""):
                    content_ext = _guess_extension_from_content(path)
                    if content_ext:
                        new_path = root_final + content_ext
                        if new_path != path and not os.path.exists(new_path):
                            os.rename(path, new_path)
                            path = new_path
                            logger.debug("download_file: renamed to %s using content guess", path)
            break
        except requests.RequestException as e:
            last_exc = e
            if attempt >= max_attempts:
                raise
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "download_file retry: attempt=%d/%d url=%s delay=%.1fs error=%s",
                attempt,
                max_attempts,
                url[:120] + ("..." if len(url) > 120 else ""),
                delay,
                e,
            )
            time.sleep(delay)

    if path is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("download_file failed before creating output path")
    if last_exc is not None and not os.path.exists(path):
        raise last_exc

    size = os.path.getsize(path)
    logger.debug(
        "download_file: url=%s -> path=%s size_bytes=%d",
        url[:100] + ("..." if len(url) > 100 else ""),
        path,
        size,
    )
    return path


def _extract_with_unstructured(path: str, max_chars: int = 200_000) -> Optional[str]:
    """
    Extract text using the unstructured library if available. Returns None if
    the library is not installed, the format is not supported, or extraction fails.
    """
    try:
        from unstructured.partition.auto import partition
    except ImportError:
        try:
            from unstructured.partition.auto import partition_auto as partition
        except ImportError:
            logger.debug("unstructured not installed, skipping partition")
            return None
    try:
        elements = partition(filename=path)
    except Exception as e:
        logger.debug("unstructured partition failed for %s: %s", path, e)
        return None
    if not elements:
        return None
    texts: List[str] = []
    for el in elements:
        t = getattr(el, "text", None) or str(el)
        if t and t.strip():
            texts.append(t.strip())
    out = "\n\n".join(texts)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n\n[Text truncated for length.]"
    return out or None


def _extract_gtfs_zip_docs(
    path: str,
    *,
    max_rows: int,
    max_members: int = 250,
) -> List[str]:
    """Extract GTFS zip into a list of docs (one per internal .txt/.csv file)."""

    def _safe_member(name: str) -> bool:
        name = name.replace("\\", "/")
        if name.startswith("/") or name.startswith("../") or "/../" in name:
            return False
        return True

    def _decode_line(b: bytes) -> str:
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return b.decode(enc)
            except UnicodeDecodeError:
                continue
        return b.decode("utf-8", errors="replace")

    docs: List[str] = []

    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist()[:max_members]:
            if getattr(info, "is_dir", lambda: info.filename.endswith("/"))():
                continue

            name = info.filename
            if not _safe_member(name):
                continue

            name_lower = name.lower()
            if not (name_lower.endswith(".txt") or name_lower.endswith(".csv")):
                continue

            lines: List[str] = []
            try:
                with zf.open(info, "r") as fp:
                    max_lines = max_rows + 1
                    for i, raw_line in enumerate(fp):
                        if i >= max_lines:
                            break
                        lines.append(_decode_line(raw_line).rstrip("\r\n"))
            except Exception:
                continue

            if not lines:
                continue

            text = "\n".join(lines).strip()
            docs.append(f"[GTFS file: {name} | preview_lines={len(lines)}]\n{text}")

    return docs


def extract_text_from_file(
    path: str,
    max_rows: int = 200,
    resource: Optional[Dict[str, Any]] = None,
) -> Union[str, List[str]]:
    """
    Extract text from a downloaded file for RAG.

    Supports: CSV, JSON/JSONL, TXT/MD, PDF, DOCX, XLSX, HTML, XML, ZIP (GTFS and
    other ZIP-based formats by extension). Uses the unstructured library when
    available for broad format support; falls back to pandas/pypdf/json for known types.
    Returns a single string or, for GTFS ZIP, a list of strings (one per internal file).
    """
    lower = path.lower()
    logger.debug("extract_text_from_file: path=%s max_rows=%d", path, max_rows)

    # Custom handling that must run first (GTFS ZIP, GTFS-RT)
    if lower.endswith(".zip"):
        r = resource or {}
        extras = r.get("extras") or {}
        fmt = str(r.get("format") or "")
        mime = str(r.get("mime") or "")
        desc = str(r.get("description") or "")
        analysis_mime = str(extras.get("analysis:mime-type") or "")
        combined = f"{fmt} {mime} {desc} {analysis_mime}".lower()
        is_gtfs = "gtfs" in combined

        if is_gtfs:
            docs = _extract_gtfs_zip_docs(path, max_rows=max_rows, max_members=250)
            if docs:
                return docs
            return f"[GTFS ZIP detected but no .txt/.csv extracted: {os.path.basename(path)}]"
        # Non-GTFS zip: try unstructured, then generic unsupported
        unstructured_out = _extract_with_unstructured(path, max_chars=200_000)
        if unstructured_out:
            return f"[ZIP content extracted]\n{unstructured_out}"
        return f"[Unsupported binary format for text extraction: {os.path.basename(path)}]"

    if lower.endswith(".bin") or lower.endswith(".pb") or lower.endswith(".protobuf"):
        unstructured_out = _extract_with_unstructured(path, max_chars=200_000)
        if unstructured_out:
            return unstructured_out
        return f"[Unsupported binary format for text extraction: {os.path.basename(path)}]"

    # Try unstructured first for all other files (CSV, DOCX, XLSX, HTML, XML, PDF, etc.)
    unstructured_out = _extract_with_unstructured(path, max_chars=200_000)
    if unstructured_out:
        return unstructured_out

    # Fallback: our own handling for CSV/JSON/JSONL (to respect max_rows) and simple text/PDF
    if lower.endswith(".csv"):
        df = pd.read_csv(path, nrows=max_rows)
        return f"[CSV preview: first {len(df)} rows]\n" + df.to_csv(index=False)

    if lower.endswith(".json") or lower.endswith(".jsonl") or lower.endswith(".geojson"):
        if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
            lines = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= max_rows:
                        break
                    lines.append(line.strip())
            return "[JSONL preview]\n" + "\n".join(lines)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
        txt = json.dumps(obj, ensure_ascii=False, indent=2)
        return txt[:200_000]

    if lower.endswith(".txt") or lower.endswith(".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:200_000]

    if lower.endswith(".pdf"):
        reader = PdfReader(path)
        pages = []
        for p in reader.pages[:20]:
            pages.append(p.extract_text() or "")
        return "[PDF extracted text]\n" + "\n\n".join(pages)[:200_000]

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:200_000]
    except Exception:
        return f"[Unsupported binary format for text extraction: {os.path.basename(path)}]"
