from __future__ import annotations

from typing import Optional, List
import logging

from app.clients.data_gouv import DataGouvDatasetsClient
from app.embeddings.azure import get_embedding_client
from app.weaviate.store import WeaviateStore


logger = logging.getLogger(__name__)


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

    batch_size = 128
    total_ingested = 0
    batch_index = 0

    def process_batch(batch_records: List, idx: int) -> int:
        if not batch_records:
            return 0
        logger.info(
            "Processing ingestion batch: index=%d, batch_size=%d, total_ingested_so_far=%d",
            idx,
            len(batch_records),
            total_ingested,
        )
        texts = [r.to_embedding_text() for r in batch_records]
        embeddings = emb_client.embed_texts(texts)

        rows = []
        for rec, content, emb in zip(batch_records, texts, embeddings):
            props = {
                "dataset_id": rec.id,
                "title": rec.title,
                "description": rec.description,
                "organization": rec.organization or "",
                "content": content,
                "url": rec.url or "",
                "tags": rec.tags or [],
            }
            rows.append((props, emb))

        inserted = store.upsert_many(rows)
        logger.info(
            "Finished ingestion batch: index=%d, attempted=%d, inserted=%d",
            idx,
            len(rows),
            inserted,
        )
        return inserted

    batch: List = []
    for rec in dg.iter_datasets(mode=mode, page=page, page_size=page_size, q=q, hard_limit=hard_limit):
        batch.append(rec)
        if len(batch) >= batch_size:
            total_ingested += process_batch(batch, batch_index)
            batch_index += 1
            batch = []

    if batch:
        total_ingested += process_batch(batch, batch_index)

    logger.info("Completed data.gouv ingestion: total_ingested=%d", total_ingested)
    return total_ingested


def search_datasets(query_text: str, k: int = 5, alpha: float = 0.5):
    emb_client = get_embedding_client()
    q_emb = emb_client.embed_texts([query_text])[0]

    store = WeaviateStore()
    return store.search(query_text=query_text, query_vector=q_emb, k=k, alpha=alpha)
