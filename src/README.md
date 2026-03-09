# Agora — Application layer

Backend (FastAPI), frontend, and agent pipeline for French open data Q&A.

- **Weaviate** runs locally in Docker (self-hosted); dataset metadata is embedded with **Azure OpenAI** (BYOV) and stored in Weaviate.
- **Backend** is FastAPI; it runs a multi-step **agent pipeline**: planner → hybrid search → dataset selector → RAG or technical agent per dataset → synthesis.
- **Streaming** and non-streaming search endpoints return a single, source-attributed answer.

## Quickstart

1. Copy env example (from repo root):

```bash
cp .env.example .env
```

2. Start Weaviate:

```bash
cd infra
docker compose up -d
```

3. Choose a backend setup (see below for **Dev** vs **Prod** flows).

## Agent pipeline (overview)

1. **Planner** — Derives intent and subqueries from the user question.
2. **Search** — Hybrid (vector + keyword) search over Weaviate for dataset hits.
3. **Dataset selector** — Chooses which datasets to use and, when in **Technical** mode, whether each dataset should be handled by RAG (general) or technical extraction.
4. **RAG (general) or technical agent** — Per selected dataset:
   - **General (RAG only)** — Download → extract text → chunk → embed → retrieve top-k chunks → LLM evidence. No RLM, no structured computation.
   - **Technical (per-resource choice)** — For each resource: if structured (CSV, JSON, XLSX, etc.), parse into records and run the **Technical agent** (RLM explores schema + preview in a sandboxed REPL); if not structured (e.g. PDF), use the same text extraction as the General agent and include it in the technical context. So Technical mode uses **both** general-style extraction and technical analysis depending on what the pipeline decides is best for each resource.
5. **Synthesis** — Combines evidence (with `[Source: dataset]` labels) into one final answer in the same language as the question.

**Frontend mode:** The UI lets you pick **General** or **Technical** before asking. General forces RAG-only for all datasets; Technical lets the selector choose RAG vs technical per dataset and runs the Technical agent where appropriate.

Embeddings use a **shared singleton client** (`get_embedding_client()`) to avoid connection leaks; chat/synthesis uses DSPy with Azure OpenAI. **Usage and cost:** The backend logs LLM token usage per agent and a pipeline grand total, plus embedding usage (search + RAG chunk retrieval only, not Weaviate ingestion); when the configured models have known pricing (e.g. gpt-5-mini, text-embedding-3-small), it logs an estimated cost at the end of each run.

## Backend setup – development

### Option A: Backend in Docker (recommended for dev)

From repo root:

```bash
cd src/backend

# Build image (uses pyproject.toml for deps)
docker build -t agora-backend .

# Run backend in dev mode with autoreload, attached to the Weaviate network
docker run --rm \
  --name agora-backend-dev \
  --env-file ../.env \
  -e WEAVIATE_URL=http://weaviate:8080 \
  -e WEAVIATE_GRPC_HOST=weaviate \
  --network infra_default \
  -p 8000:8000 \
  -v "$(pwd)":/app \
  agora-backend \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: Backend locally in a venv

From repo root:

```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
uvicorn app.main:app --reload
```

### Ingest + search (same for both dev options)

In another terminal (from repo root):

```bash
cd src/backend
python scripts/ingest_data_gouv.py --mode single_page --page 1 --page-size 50
```

Then test **non-streaming** search:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question":"Je cherche des données sur la qualité de l'\''air à Paris","k":5}'
```

**Streaming** search (SSE: plan, status, user_message, done with full response):

```bash
curl -X POST http://localhost:8000/search/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Quels jeux de données sur la qualité de l'\''air à Paris ?","k":5}'
```

The frontend (see **`src/frontend/`**) uses the streaming endpoint and shows progress plus the final answer. Before submitting, you choose **General** (RAG only) or **Technical** (RAG + technical per resource); see **General vs Technical** in the root README.

### Ingest the French data.gouv catalogue (15k sample vs full ~73k)

This is a long-running, Azure-OpenAI‑heavy operation. The `--hard-limit` is **arbitrary** and mainly used to catch issues early before you run a full ingestion.

As of March 2026, data.gouv exposes **~73,000 datasets**. Embedding all of them with the current embedding model costs on the order of **\$0.03** (very cheap, but still worth being aware of).

- **From a local venv** (backend running either locally or in Docker):

  ```bash
  cd src/backend
  python scripts/ingest_data_gouv.py \
    --mode all_pages \
    --page 1 \
    --page-size 100 \
    --hard-limit 15000
  ```

- **From Docker only** (no local Python needed):

  ```bash
  cd src/backend
  docker run --rm \
    --env-file ../.env \
    -e WEAVIATE_URL=http://weaviate:8080 \
    -e WEAVIATE_GRPC_HOST=weaviate \
    --network infra_default \
    agora-backend \
    python scripts/ingest_data_gouv.py \
      --mode all_pages \
      --page 1 \
      --page-size 100 \
      --hard-limit 15000
  ```

The `--mode all_pages` flag walks all pages from data.gouv, and `--hard-limit` caps the total number of datasets ingested. You can raise it up to ~73k if you want to embed the full catalogue. However, make sure `--hard-limit` is greater than the total number of datasets. In practice (as of March 2026), you can set `--hard-limit` to 800000 to ensure that ALL datasets get embedded. You can also optionally remove the `--hard-limit` flag altogether.

### Optional: admin UI (Streamlit)

From `src`:

```bash
cd src
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install streamlit
python -m streamlit run admin_gui/app.py
```

The UI assumes `BACKEND_URL=http://localhost:8000` (override via env if needed).

## Test scripts (backend)

From `src/backend` (with venv active and Weaviate + backend running):

- **Full pipeline** (one question → plan, search, select, RAG/technical, synthesis):

  ```bash
  python -m scripts.run_full_workflow_test "Quels jeux de données sur la qualité de l'air à Paris ?" --k 3
  ```

- **General (RAG) agent only** (subquery + dataset ID(s)):

  ```bash
  python -m scripts.run_general_agent_test "qualité de l'air Paris" --dataset-ids <id1> [id2 ...]
  ```

## Backend setup – Production (Docker)

From repo root:

```bash
cd src/backend
docker build -t agora-backend .

docker run -d \
  --name agora-backend \
  --env-file ../.env \
  -e WEAVIATE_URL=http://weaviate:8080 \
  -e WEAVIATE_GRPC_HOST=weaviate \
  --network infra_default \
  -p 8000:8000 \
  agora-backend
```

## Notes

- **Do not** commit `.env`.
- Embeddings use a shared Azure client (`app.embeddings.azure.get_embedding_client()`) to avoid connection leaks.
- Backend Docker image is built from `pyproject.toml`; `requirements.txt` is not used for the image but may be used for pinned versions later.
