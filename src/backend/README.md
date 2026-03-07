# Agora backend

FastAPI backend for Agora: agent pipeline (planner, hybrid search, dataset selector, RAG/technical agents, synthesis) over French open data (data.gouv.fr). Embeddings and chat use Azure OpenAI; vectors live in Weaviate.

## Quick reference

- **Run API (dev):** from this directory: `uvicorn app.main:app --reload`
- **Ingest:** `python scripts/ingest_data_gouv.py --mode single_page --page 1 --page-size 50` (see `src/README.md` for full ingestion)
- **Test full pipeline:** `python -m scripts.run_full_workflow_test "Your question" --k 3`
- **Test RAG agent only:** `python -m scripts.run_general_agent_test "subquery" --dataset-ids <id1> [id2 ...]`

Full setup (Weaviate, Docker, env, streaming, production): **[`src/README.md`](../README.md)**.
