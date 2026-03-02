# AGORA project

This repo is a starting point for migrating the Colab notebook workflow to a local, GitHub-friendly setup:

- **Weaviate** runs locally in Docker (self-hosted).
- **Backend** is FastAPI.
- Embeddings are generated with **Azure OpenAI** (same as notebook) - specificaly the text-3-embeddings-small model.
- Dataset vectors are **self-provided** (BYOV) - using the above mentioned embeddings model - and stored in Weaviate.

## Quickstart

1) Copy env example:

```bash
cp .env.example .env
```

2) Start Weaviate:

```bash
cd infra
docker compose up -d
```

3) Choose a backend setup (see below for **Dev** vs **Prod** flows).

## Backend setup – development

### Option A: Backend in Docker (recommended for dev)

From `src/backend`:

```bash
cd backend

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

From `src/backend`:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
uvicorn app.main:app --reload
```

### Ingest + search (same for both dev options)

In another terminal:

```bash
cd backend
python scripts/ingest_data_gouv.py --mode single_page --page 1 --page-size 50
```

Then test search:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question":"Je cherche des données sur la qualité de l\'air à Paris","k":5}'
```

### Ingest the French data.gouv catalogue (15k sample vs full ~73k)

This is a long-running, Azure-OpenAI‑heavy operation. The `--hard-limit` is **arbitrary** and mainly used to catch issues early before you run a full ingestion.

As of March 2026, data.gouv exposes **~73,000 datasets**. Embedding all of them with the current embedding model costs on the order of **\$0.03** (very cheap, but still worth being aware of).

- **From a local venv** (backend running either locally or in Docker):

  ```bash
  cd backend
  python scripts/ingest_data_gouv.py \
    --mode all_pages \
    --page 1 \
    --page-size 100 \
    --hard-limit 15000
  ```

- **From Docker only** (no local Python needed):

  ```bash
  cd backend
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

## Backend setup – Production (Docker)

From `src/backend`:

```bash
cd backend
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
- If you previously used Neon/pgvector: this repo removes it entirely and replaces it with Weaviate.
- requirements.txt is NOT BEING USED to build the backend docker image, but it remains in the project's backend/ directory to be used later if there is need for pinned versions.
