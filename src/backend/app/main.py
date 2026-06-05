from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agents.orchestrator import AgentOrchestrator, _stream_run
from app.pipelines.retrieval import ingest_data_gouv, iter_sync_data_gouv
from app.weaviate.store import WeaviateStore

_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
_level = getattr(logging, _log_level, logging.INFO)
logging.basicConfig(
    level=_level,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(title="Agora — French Open Data Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend from src/frontend
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")

    @app.get("/")
    def index():
        return FileResponse(_FRONTEND_DIR / "index.html")

class SearchRequest(BaseModel):
    question: str
    k: int = Field(default=5, ge=1, le=50)
    use_only_general_agent: bool | None = Field(
        default=None,
        description="If True, use RAG (general) only; if False, use both RAG and technical. None = use server default.",
    )

class IngestRequest(BaseModel):
    mode: str = "single_page"
    page: int = 1
    page_size: int = 50
    q: str | None = None
    hard_limit: int | None = None


class SyncRequest(BaseModel):
    page_size: int = Field(default=100, ge=10, le=200)
    prune_stale: bool = Field(
        default=True,
        description="Remove Weaviate datasets that no longer exist on data.gouv.fr.",
    )


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


def _sse_ingest_sync(page_size: int, prune_stale: bool):
    stream = iter_sync_data_gouv(page_size=page_size, prune_stale=prune_stale)
    try:
        for payload in stream:
            yield f"data: {json.dumps(payload)}\n\n"
    except GeneratorExit:
        stream.close() if hasattr(stream, "close") else None
        raise
    except Exception as e:
        logger.exception("Ingest sync stream failed")
        yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"


@app.post("/ingest/sync/stream")
def ingest_sync_stream(req: SyncRequest):
    """Stream a full data.gouv.fr → Weaviate sync via Server-Sent Events."""
    logger.info(
        "HTTP /ingest/sync/stream called: page_size=%d, prune_stale=%s",
        req.page_size,
        req.prune_stale,
    )
    return StreamingResponse(
        _sse_ingest_sync(req.page_size, req.prune_stale),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/search")
def search(req: SearchRequest):

    orchestrator = AgentOrchestrator()

    result = orchestrator.run(req.question, k=req.k)

    return result.model_dump()


def _sse_stream(question: str, k: int, use_only_general_agent: bool | None = None):
    """Yield Server-Sent Events: one event per orchestrator step.
    When the client disconnects, the response is closed and this generator
    receives GeneratorExit; we close the inner _stream_run generator so
    the pipeline and any held connections are released.
    """
    orchestrator = AgentOrchestrator()
    stream_run = _stream_run(orchestrator, question, k=k, use_only_general_agent=use_only_general_agent)
    try:
        for payload in stream_run:
            yield f"data: {json.dumps(payload)}\n\n"
    except GeneratorExit:
        stream_run.close()
        raise
    finally:
        try:
            stream_run.close()
        except Exception:
            pass


@app.post("/search/stream")
def search_stream(req: SearchRequest):
    """Stream search progress in real time via Server-Sent Events (SSE).
    Connect with EventSource or fetch with stream; each event is JSON:
    - event: 'status' | 'plan' | 'user_message' | 'technical_repl' | 'evidence' | 'done'
    - message (for status, user_message)
    - plan (for plan)
    - response (for done, full AgentResponse as dict)
    """
    return StreamingResponse(
        _sse_stream(req.question, k=req.k, use_only_general_agent=req.use_only_general_agent),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/debug/count")
def debug_count():
    store = WeaviateStore()
    return {"collection": store.collection_name, "count": store.count()}

@app.get("/debug/sample")
def debug_sample(limit: int = 20):
    store = WeaviateStore()
    return {"collection": store.collection_name, "items": store.sample(limit=limit)}
