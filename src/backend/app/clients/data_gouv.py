from __future__ import annotations

import logging
import re
import time
import requests
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import tiktoken

DATASETS_URL = "https://www.data.gouv.fr/api/1/datasets/"
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
MAX_DESC_TOKENS = 7000


logger = logging.getLogger(__name__)
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_S = 1.5


def _collapse_ws(s: str) -> str:
    return " ".join(s.split())


def truncate_desc_tokens(desc: str, max_tokens: int = MAX_DESC_TOKENS) -> str:
    tokens = _TIKTOKEN_ENC.encode(desc)
    if len(tokens) <= max_tokens:
        return desc
    truncated = _TIKTOKEN_ENC.decode(tokens[:max_tokens])
    return truncated + " …"


def extract_resource_urls(ds: Dict[str, Any]) -> List[Dict[str, Any]]:
    """From a dataset payload, return list of {url, resource} for each resource."""
    out: List[Dict[str, Any]] = []
    for res in ds.get("resources") or []:
        u = res.get("url")
        if not u:
            continue
        out.append({"url": u, "resource": res})
    return out


@dataclass
class DatasetRecord:
    id: str
    title: str
    description: str
    tags: List[str]
    organization: Optional[str] = None
    url: Optional[str] = None

    def to_embedding_text(self) -> str:
        # Keep your “title/description/org” focus + basic cleanup
        title = _collapse_ws(self.title or "")
        desc = _collapse_ws(self.description or "")
        desc = truncate_desc_tokens(desc)
        org = _collapse_ws(self.organization or "")
        parts = []
        if title:
            parts.append(title)
        if desc:
            parts.append(desc)
        if org:
            parts.append(org)
        return "\n".join(parts)


class DataGouvDatasetsClient:
    def __init__(
        self,
        base_url: str = DATASETS_URL,
        timeout_s: int = 60,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
    ):
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff_s = max(0.1, retry_backoff_s)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get_json_with_retry(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                with self.session.get(url, params=params, timeout=self.timeout_s) as resp:
                    resp.raise_for_status()
                    return resp.json()
            except requests.RequestException as e:
                last_exc = e
                if attempt >= self.retry_attempts:
                    break
                delay = self.retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "data.gouv request retry: attempt=%d/%d url=%s params=%s delay=%.1fs error=%s",
                    attempt,
                    self.retry_attempts,
                    url,
                    params,
                    delay,
                    e,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Fetch a single dataset by ID (full payload including resources)."""
        url = f"{self.base_url.rstrip('/')}/{dataset_id}/"
        logger.info("Fetching dataset: dataset_id=%s", dataset_id)
        return self._get_json_with_retry(url)

    def get_resource(self, dataset_id: str, resource_id: str) -> Dict[str, Any]:
        """
        Fetch a single resource by dataset ID and resource ID (rid).
        Uses GET /datasets/{dataset}/resources/{rid}/ so full metadata (e.g. size) is available.
        See: https://guides.data.gouv.fr/api-de-data.gouv.fr/reference/datasets
        """
        url = f"{self.base_url.rstrip('/')}/{dataset_id}/resources/{resource_id}/"
        return self._get_json_with_retry(url)

    def fetch_page(
        self,
        page: int = 1,
        page_size: int = 50,
        q: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if q:
            params["q"] = q
        logger.info(
            "Fetching data.gouv page: page=%d, page_size=%d, q=%s",
            page,
            page_size,
            q,
        )
        return self._get_json_with_retry(self.base_url, params=params)

    @staticmethod
    def _pick_str(d: Dict[str, Any], *keys: str) -> str:
        for k in keys:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    @staticmethod
    def _org_name(d: Dict[str, Any]) -> Optional[str]:
        org = d.get("organization")
        if isinstance(org, dict):
            name = org.get("name") or org.get("title")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return None

    def iter_datasets(
        self,
        mode: str = "single_page",
        page: int = 1,
        page_size: int = 50,
        q: Optional[str] = None,
        hard_limit: Optional[int] = None,
    ) -> Iterator[DatasetRecord]:
        fetched = 0
        p = page
        while True:
            payload = self.fetch_page(page=p, page_size=page_size, q=q)
            items = payload.get("data", [])
            logger.info(
                "Page fetched from data.gouv: page=%d, items=%d, fetched_total=%d, mode=%s, hard_limit=%s",
                p,
                len(items),
                fetched,
                mode,
                hard_limit,
            )
            for it in items:
                rec = DatasetRecord(
                    id=str(it.get("id") or ""),
                    title=self._pick_str(it, "title", "name"),
                    description=self._pick_str(it, "description", "notes"),
                    tags=[t for t in (it.get("tags") or []) if isinstance(t, str)],
                    organization=self._org_name(it),
                    url=self._pick_str(it, "page", "uri", "url"),
                )
                if rec.id:
                    yield rec
                    fetched += 1
                    if hard_limit and fetched >= hard_limit:
                        return
            if mode != "all_pages":
                return
            # pagination
            next_page = payload.get("next_page")
            if not next_page:
                return
            p += 1
