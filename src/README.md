# data-gouv → Weaviate (local) project

This repo is a starting point for migrating the Colab notebook workflow to a local, GitHub-friendly setup:

- **Weaviate** runs locally in Docker (self-hosted).
- **Backend** is FastAPI.
- Embeddings are generated with **Azure OpenAI** (same as notebook).
- Dataset vectors are **self-provided** (BYOV) and stored in Weaviate.

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

3) Start the backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
uvicorn app.main:app --reload
```

4) Ingest datasets (example: first page):

```bash
cd backend
python scripts/ingest_data_gouv.py --mode single_page --page 1 --page-size 50
```

5) Search:

```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"question":"Je cherche des données sur la qualité de l\'air à Paris","k":5}'
```

## Notes

- **Do not** commit `.env`.
- If you previously used Neon/pgvector: this repo removes it entirely and replaces it with Weaviate.
