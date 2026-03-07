"""
Download data.gouv resources and extract text for RAG.
Mirrors the logic in notebooks/generate_questions_manual_w_gtfs_v1.1.ipynb:
CSV, JSON/JSONL, TXT/MD, PDF, ZIP (GTFS), and GTFS-RT protobuf.
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


def download_file(
    url: str,
    out_dir: str,
    timeout_s: int = 60,
    retry_attempts: int = DEFAULT_DOWNLOAD_RETRY_ATTEMPTS,
    retry_backoff_s: float = DEFAULT_DOWNLOAD_RETRY_BACKOFF_S,
) -> str:
    """Download URL to out_dir; return local path. Retries transient request errors."""
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


def _meta_text(resource: Optional[Dict[str, Any]]) -> str:
    r = resource or {}
    extras = r.get("extras") or {}
    parts = [
        str(r.get("format") or ""),
        str(r.get("mime") or ""),
        str(r.get("description") or ""),
        str(extras.get("analysis:mime-type") or ""),
    ]
    return " ".join(parts).lower()


def _looks_like_gtfs_rt(resource: Optional[Dict[str, Any]]) -> bool:
    t = _meta_text(resource)
    return any(
        k in t
        for k in ["gtfs-rt", "gtfsrt", "gtfs realtime", "realtime", "protobuf", "x-protobuf"]
    )


def extract_gtfs_rt_preview(path: str, max_entities: int = 50) -> str:
    """Decode GTFS-RT protobuf into JSON preview. Requires gtfs-realtime-bindings."""
    try:
        from google.transit import gtfs_realtime_pb2
    except Exception:
        return (
            "[GTFS-RT detected but decoder not installed]\n"
            "Install: pip install gtfs-realtime-bindings protobuf"
        )

    with open(path, "rb") as f:
        data = f.read()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    out: Dict[str, Any] = {
        "header": {
            "gtfs_realtime_version": feed.header.gtfs_realtime_version,
            "incrementality": (
                int(feed.header.incrementality) if feed.header.HasField("incrementality") else None
            ),
            "timestamp": (
                int(feed.header.timestamp) if feed.header.HasField("timestamp") else None
            ),
        },
        "entity_count": len(feed.entity),
        "entities_preview": [],
    }

    for ent in feed.entity[:max_entities]:
        e: Dict[str, Any] = {"id": ent.id or None}
        if ent.is_deleted:
            e["is_deleted"] = True

        if ent.HasField("trip_update"):
            tu = ent.trip_update
            e["trip_update"] = {
                "trip": {
                    "trip_id": tu.trip.trip_id or None,
                    "route_id": tu.trip.route_id or None,
                    "direction_id": (
                        tu.trip.direction_id if tu.trip.HasField("direction_id") else None
                    ),
                    "start_time": tu.trip.start_time or None,
                    "start_date": tu.trip.start_date or None,
                },
                "vehicle": {
                    "id": tu.vehicle.id or None,
                    "label": tu.vehicle.label or None,
                },
                "timestamp": int(tu.timestamp) if tu.HasField("timestamp") else None,
                "delay": int(tu.delay) if tu.HasField("delay") else None,
                "stop_time_updates": [],
            }
            for stu in tu.stop_time_update[:50]:
                e["trip_update"]["stop_time_updates"].append(
                    {
                        "stop_id": stu.stop_id or None,
                        "arrival": (
                            {
                                "time": (
                                    int(stu.arrival.time)
                                    if stu.arrival.HasField("time")
                                    else None
                                ),
                                "delay": (
                                    int(stu.arrival.delay)
                                    if stu.arrival.HasField("delay")
                                    else None
                                ),
                            }
                            if stu.HasField("arrival")
                            else None
                        ),
                        "departure": (
                            {
                                "time": (
                                    int(stu.departure.time)
                                    if stu.departure.HasField("time")
                                    else None
                                ),
                                "delay": (
                                    int(stu.departure.delay)
                                    if stu.departure.HasField("delay")
                                    else None
                                ),
                            }
                            if stu.HasField("departure")
                            else None
                        ),
                    }
                )

        if ent.HasField("vehicle"):
            vp = ent.vehicle
            e["vehicle_position"] = {
                "trip": (
                    {
                        "trip_id": vp.trip.trip_id or None,
                        "route_id": vp.trip.route_id or None,
                    }
                    if vp.HasField("trip")
                    else None
                ),
                "vehicle": (
                    {
                        "id": vp.vehicle.id or None,
                        "label": vp.vehicle.label or None,
                    }
                    if vp.HasField("vehicle")
                    else None
                ),
                "timestamp": int(vp.timestamp) if vp.HasField("timestamp") else None,
                "position": {
                    "latitude": (
                        vp.position.latitude if vp.HasField("position") else None
                    ),
                    "longitude": (
                        vp.position.longitude if vp.HasField("position") else None
                    ),
                    "bearing": (
                        vp.position.bearing
                        if vp.HasField("position") and vp.position.HasField("bearing")
                        else None
                    ),
                    "speed": (
                        vp.position.speed
                        if vp.HasField("position") and vp.position.HasField("speed")
                        else None
                    ),
                },
                "current_status": (
                    int(vp.current_status) if vp.HasField("current_status") else None
                ),
                "stop_id": vp.stop_id or None,
            }

        if ent.HasField("alert"):
            al = ent.alert
            e["alert"] = {
                "cause": int(al.cause) if al.HasField("cause") else None,
                "effect": int(al.effect) if al.HasField("effect") else None,
                "header_text": (
                    al.header_text.translation[0].text
                    if al.header_text.translation
                    else None
                ),
                "description_text": (
                    al.description_text.translation[0].text
                    if al.description_text.translation
                    else None
                ),
            }

        out["entities_preview"].append(e)

    txt = json.dumps(out, ensure_ascii=False, indent=2)
    return "[GTFS-RT decoded preview]\n" + txt[:200_000]


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
    Extract text from a downloaded file for RAG. Handles CSV, JSON/JSONL, TXT/MD,
    PDF, ZIP (GTFS), and .bin/.pb (GTFS-RT). Returns a single string or, for
    GTFS ZIP, a list of strings (one per internal file).
    """
    lower = path.lower()
    logger.debug("extract_text_from_file: path=%s max_rows=%d", path, max_rows)

    if lower.endswith(".csv"):
        df = pd.read_csv(path, nrows=max_rows)
        return f"[CSV preview: first {len(df)} rows]\n" + df.to_csv(index=False)

    if lower.endswith(".json") or lower.endswith(".jsonl"):
        if lower.endswith(".jsonl"):
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
        return f"[Unsupported binary format for text extraction: {os.path.basename(path)}]"

    if lower.endswith(".bin") or lower.endswith(".pb") or lower.endswith(".protobuf"):
        if _looks_like_gtfs_rt(resource):
            return extract_gtfs_rt_preview(path, max_entities=max_rows)
        return f"[Unsupported binary format for text extraction: {os.path.basename(path)}]"

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:200_000]
    except Exception:
        return f"[Unsupported binary format for text extraction: {os.path.basename(path)}]"
