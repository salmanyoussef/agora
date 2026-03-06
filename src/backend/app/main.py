from __future__ import annotations

from fastapi import FastAPI
import logging
from pydantic import BaseModel, Field

from app.pipelines.retrieval import ingest_data_gouv
from app.agents.orchestrator import AgentOrchestrator

from app.weaviate.store import WeaviateStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(title="Data.gouv dataset retrieval (Weaviate)")

class SearchRequest(BaseModel):
    question: str
    k: int = Field(default=5, ge=1, le=50)

class IngestRequest(BaseModel):
    mode: str = "single_page"
    page: int = 1
    page_size: int = 50
    q: str | None = None
    hard_limit: int | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest")
def ingest(req: IngestRequest):
    logger.info(
        "HTTP /ingest called: mode=%s, page=%d, page_size=%d, q=%s, hard_limit=%s",
        req.mode,
        req.page,
        req.page_size,
        req.q,
        req.hard_limit,
    )
    n = ingest_data_gouv(
        mode=req.mode,
        page=req.page,
        page_size=req.page_size,
        q=req.q,
        hard_limit=req.hard_limit,
    )
    logger.info("HTTP /ingest completed: ingested=%d", n)
    return {"ingested": n}


@app.post("/search")
def search(req: SearchRequest):

    orchestrator = AgentOrchestrator()

    result = orchestrator.run(req.question, k=req.k)

    return result.model_dump()

@app.get("/debug/count")
def debug_count():
    store = WeaviateStore()
    return {"collection": store.collection_name, "count": store.count()}

@app.get("/debug/sample")
def debug_sample(limit: int = 20):
    store = WeaviateStore()
    return {"collection": store.collection_name, "items": store.sample(limit=limit)}
