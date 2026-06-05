# Agora backend

FastAPI backend for Agora: agent pipeline (planner, hybrid search, dataset selector, RAG/technical agents, synthesis) over French open data (data.gouv.fr). Chat and embeddings use **OpenAI** or **Azure OpenAI** via `LLM_PROVIDER` in `src/.env`; vectors live in Weaviate.

## Quick reference

- **Run API (dev):** from this directory: `uvicorn app.main:app --reload`
- **Ingest:** `python scripts/ingest_data_gouv.py --mode single_page --page 1 --page-size 50` (see `src/README.md` for full ingestion)
- **Test full pipeline:** `python -m scripts.run_full_workflow_test "Your question" --k 3`
- **Test RAG agent only:** `python -m scripts.run_general_agent_test "subquery" --dataset-ids <id1> [id2 ...]`

Full setup (Weaviate, Docker, env, streaming, production): **[`src/README.md`](../README.md)**.

## Modes: General vs Technical

- **General** (frontend choice or `USE_ONLY_GENERAL_AGENT=True`) — Every dataset is processed with the **RAG agent only**: download → extract text → chunk → embed → retrieve top-k chunks → LLM answer. No technical/computation path, no RLM.
- **Technical** (frontend choice or `USE_ONLY_GENERAL_AGENT=False`) — The **dataset selector** picks datasets for **RAG** or **technical** mode. For technical runs: a **resource selector** agent chooses **one** resource per dataset (preferring CSV/JSON/XLSX for computation), then the **technical agent** downloads it, parses up to **50,000 rows** into a REPL variable `records`, and runs **DSPy RLM** with metadata in `resource_context`. Non-tabular resources fall back to extracted text in `resource_context` only.

## Technical agent & RLM

Pipeline for each technical dataset: **ResourceSelectorAgent** (metadata-only) → download one resource → parse → **DSPy RLM** with `records` (list of dicts, up to 50k rows) and `resource_context` (dataset/resource/schema text). The model writes Python in a sandboxed REPL to filter, aggregate, etc., and calls `llm_query()` when needed, then `SUBMIT(answer)`. Requires **Deno** for the WASM sandbox; otherwise falls back to `dspy.Predict`. Setup: `agora-setup-repl` or `uv run python -m app.scripts.setup_repl` (see `[tool.agora.repl]` in `pyproject.toml`).

## LLM provider (plug-and-play)

Set `LLM_PROVIDER` in `src/.env` to switch backends without code changes:

| `LLM_PROVIDER` | Required env vars | Chat | Embeddings |
|----------------|-------------------|------|------------|
| `openai` | `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBED_MODEL` | `api.openai.com` | OpenAI SDK |
| `azure` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_CHAT_DEPLOYMENT`, `AZURE_OPENAI_EMBED_DEPLOYMENT` | Azure deployment | Azure OpenAI SDK |

Restart the backend after changing provider. Both credential sets can stay in `.env`; only the active provider’s vars are required.

## Usage and cost tracking

- **LLM usage** — Logged per agent (planner, selector, general_rag, technical_rlm, synthesis) and as a **pipeline grand total**. Cost uses `KNOWN_MODEL_PRICING` keyed by **model name** (OpenAI) or **deployment name** (Azure) when they match (e.g. `gpt-5-mini`).
- **Embedding usage** — Logged for **pipeline** use only: search (one embed per subquery) and General agent chunk retrieval. Weaviate ingestion is not counted. Priced via `KNOWN_EMBED_PRICING` on embed model/deployment name (e.g. `text-embedding-3-small`).
