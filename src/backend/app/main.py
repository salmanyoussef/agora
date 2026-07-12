from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agents.orchestrator import AgentOrchestrator, _stream_run
from app.pipelines.retrieval import ingest_data_gouv, iter_sync_data_gouv
from app.services.budget import EXHAUSTED_MESSAGE, budget_tracker
from app.weaviate.store import WeaviateStore

_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
_level = getattr(logging, _log_level, logging.INFO)
logging.basicConfig(
    level=_level,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _client_ip(request: Request) -> str:
    """Rate-limit key: first hop of X-Forwarded-For when behind a proxy (Render/ACA)."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)


# Per-IP throttling on the search endpoints (public demo hardening).
SEARCH_RATE_LIMIT = os.environ.get("AGORA_RATE_LIMIT", "3/minute;20/hour;60/day")

# Cap simultaneous pipeline runs so a burst can't OOM the instance.
_MAX_CONCURRENT = int(os.environ.get("AGORA_MAX_CONCURRENT", "3") or 3)
_run_slots = threading.Semaphore(_MAX_CONCURRENT)

# Admin token gates ingestion/debug endpoints when set (always set it in public deploys).
_ADMIN_TOKEN = os.environ.get("AGORA_ADMIN_TOKEN") or None

limiter = Limiter(key_func=_client_ip)

app = FastAPI(title="Agora — French Open Data Q&A")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _require_admin(request: Request) -> None:
    if _ADMIN_TOKEN is None:
        return  # not configured (local/dev use)
    if request.headers.get("x-admin-token") != _ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin token required")

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


@app.get("/budget")
def budget_status():
    """Public read-only view of demo budget consumption."""
    return budget_tracker.status()


@app.post("/ingest")
def ingest(req: IngestRequest, request: Request):
    _require_admin(request)
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
def ingest_sync_stream(req: SyncRequest, request: Request):
    """Stream a full data.gouv.fr → Weaviate sync via Server-Sent Events."""
    _require_admin(request)
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
@limiter.limit(SEARCH_RATE_LIMIT)
def search(req: SearchRequest, request: Request):
    if budget_tracker.exhausted:
        raise HTTPException(status_code=503, detail=EXHAUSTED_MESSAGE)
    if not _run_slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Demo is busy — please retry in a minute.")
    try:
        orchestrator = AgentOrchestrator()
        result = orchestrator.run(req.question, k=req.k)
    finally:
        _run_slots.release()
    budget_tracker.record_run(getattr(result, "pipeline_cost_usd", None))
    return result.model_dump()


def _sse_stream(question: str, k: int, use_only_general_agent: bool | None = None):
    """Yield Server-Sent Events: one event per orchestrator step.
    When the client disconnects, the response is closed and this generator
    receives GeneratorExit; we close the inner _stream_run generator so
    the pipeline and any held connections are released.
    """
    if not _run_slots.acquire(blocking=False):
        yield f"data: {json.dumps({'event': 'error', 'message': 'Demo is busy — please retry in a minute.'})}\n\n"
        return
    orchestrator = AgentOrchestrator()
    stream_run = _stream_run(orchestrator, question, k=k, use_only_general_agent=use_only_general_agent)
    try:
        for payload in stream_run:
            if payload.get("event") == "done":
                response = payload.get("response") or {}
                budget_tracker.record_run(response.get("pipeline_cost_usd"))
            yield f"data: {json.dumps(payload)}\n\n"
    except GeneratorExit:
        stream_run.close()
        raise
    finally:
        _run_slots.release()
        try:
            stream_run.close()
        except Exception:
            pass


@app.post("/search/stream")
@limiter.limit(SEARCH_RATE_LIMIT)
def search_stream(req: SearchRequest, request: Request):
    """Stream search progress in real time via Server-Sent Events (SSE).
    Connect with EventSource or fetch with stream; each event is JSON:
    - event: 'status' | 'plan' | 'user_message' | 'technical_repl' | 'evidence' | 'done'
    - message (for status, user_message)
    - plan (for plan)
    - response (for done, full AgentResponse as dict)
    """
    if budget_tracker.exhausted:
        def _exhausted():
            yield f"data: {json.dumps({'event': 'error', 'message': EXHAUSTED_MESSAGE})}\n\n"
        return StreamingResponse(_exhausted(), media_type="text/event-stream")
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
def debug_count(request: Request):
    _require_admin(request)
    store = WeaviateStore()
    return {"collection": store.collection_name, "count": store.count()}

@app.get("/debug/sample")
def debug_sample(request: Request, limit: int = 20):
    _require_admin(request)
    store = WeaviateStore()
    return {"collection": store.collection_name, "items": store.sample(limit=limit)}
