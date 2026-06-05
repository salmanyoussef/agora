from __future__ import annotations

from typing import Any, Iterator, List, Optional, Set
import logging

from app.clients.data_gouv import DataGouvDatasetsClient, DatasetRecord
from app.embeddings.azure import get_embedding_client
from app.weaviate.store import WeaviateStore


logger = logging.getLogger(__name__)

SYNC_BATCH_SIZE = 128


def _record_to_props(rec: DatasetRecord, content: str) -> dict[str, Any]:
    return {
        "dataset_id": rec.id,
        "title": rec.title,
        "description": rec.description,
        "organization": rec.organization or "",
        "content": content,
        "url": rec.url or "",
        "tags": rec.tags or [],
    }


def _upsert_batch(
    batch_records: List[DatasetRecord],
    emb_client: Any,
    store: WeaviateStore,
) -> int:
    if not batch_records:
        return 0
    texts = [r.to_embedding_text() for r in batch_records]
    embeddings, _ = emb_client.embed_texts(texts)
    rows = [
        (_record_to_props(rec, content), emb)
        for rec, content, emb in zip(batch_records, texts, embeddings)
    ]
    return store.upsert_many(rows)


def iter_sync_data_gouv(
    page_size: int = 100,
    prune_stale: bool = True,
) -> Iterator[dict[str, Any]]:
    """
    Walk the full data.gouv.fr catalogue and upsert into Weaviate.
    Yields SSE-friendly event dicts for progress UI.
    """
    logger.info("Starting data.gouv sync: page_size=%d, prune_stale=%s", page_size, prune_stale)
    yield {
        "event": "status",
        "message": "Starting sync from data.gouv.fr (full catalogue)…",
    }

    dg = DataGouvDatasetsClient()
    emb_client = get_embedding_client()
    store = WeaviateStore()

    before_count = store.count()
    yield {
        "event": "status",
        "message": f"Weaviate currently has {before_count:,} indexed datasets.",
    }

    seen_ids: Set[str] = set()
    total_synced = 0
    batch_index = 0
    batch: List[DatasetRecord] = []

    for rec in dg.iter_datasets(mode="all_pages", page=1, page_size=page_size):
        seen_ids.add(rec.id)
        batch.append(rec)
        if len(batch) >= SYNC_BATCH_SIZE:
            upserted = _upsert_batch(batch, emb_client, store)
            total_synced += upserted
            batch_index += 1
            batch = []
            yield {
                "event": "progress",
                "message": f"Synced {total_synced:,} datasets (batch {batch_index})…",
                "synced": total_synced,
                "batch": batch_index,
            }

    if batch:
        upserted = _upsert_batch(batch, emb_client, store)
        total_synced += upserted
        batch_index += 1
        yield {
            "event": "progress",
            "message": f"Synced {total_synced:,} datasets (batch {batch_index})…",
            "synced": total_synced,
            "batch": batch_index,
        }

    removed = 0
    if prune_stale:
        yield {
            "event": "status",
            "message": "Removing datasets no longer on data.gouv.fr…",
        }
        removed = store.delete_stale_except(seen_ids)

    after_count = store.count()
    logger.info(
        "Completed data.gouv sync: synced=%d, removed=%d, weaviate_count=%d",
        total_synced,
        removed,
        after_count,
    )
    yield {
        "event": "done",
        "synced": total_synced,
        "removed": removed,
        "catalogue_count": len(seen_ids),
        "weaviate_count": after_count,
        "message": (
            f"Sync complete — {after_count:,} datasets in Weaviate "
            f"({total_synced:,} upserted"
            + (f", {removed:,} removed" if removed else "")
            + ")."
        ),
    }


def ingest_data_gouv(
    mode: str = "single_page",
    page: int = 1,
    page_size: int = 50,
    q: Optional[str] = None,
    hard_limit: Optional[int] = None,
) -> int:
    logger.info(
        "Starting data.gouv ingestion: mode=%s, page=%d, page_size=%d, q=%s, hard_limit=%s",
        mode,
        page,
        page_size,
        q,
        hard_limit,
    )
    dg = DataGouvDatasetsClient()
    emb_client = get_embedding_client()
    store = WeaviateStore()

    total_ingested = 0
    batch_index = 0
    batch: List[DatasetRecord] = []

    for rec in dg.iter_datasets(mode=mode, page=page, page_size=page_size, q=q, hard_limit=hard_limit):
        batch.append(rec)
        if len(batch) >= SYNC_BATCH_SIZE:
            logger.info(
                "Processing ingestion batch: index=%d, batch_size=%d, total_ingested_so_far=%d",
                batch_index,
                len(batch),
                total_ingested,
            )
            inserted = _upsert_batch(batch, emb_client, store)
            total_ingested += inserted
            logger.info(
                "Finished ingestion batch: index=%d, attempted=%d, upserted=%d",
                batch_index,
                len(batch),
                inserted,
            )
            batch_index += 1
            batch = []

    if batch:
        inserted = _upsert_batch(batch, emb_client, store)
        total_ingested += inserted

    logger.info("Completed data.gouv ingestion: total_ingested=%d", total_ingested)
    return total_ingested


def search_datasets(query_text: str, k: int = 5, alpha: float = 0.5):
    """Return (hits, embed_usage). embed_usage is for pipeline cost tracking (None if no usage)."""
    emb_client = get_embedding_client()
    embeddings, embed_usage = emb_client.embed_texts([query_text])
    q_emb = embeddings[0] if embeddings else None
    if q_emb is None:
        return [], embed_usage
    store = WeaviateStore()
    hits = store.search(query_text=query_text, query_vector=q_emb, k=k, alpha=alpha)
    return hits, embed_usage
