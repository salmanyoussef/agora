# Agora backend

FastAPI backend for Agora: agent pipeline (planner, hybrid search, dataset selector, RAG/technical agents, synthesis) over French open data (data.gouv.fr). Embeddings and chat use Azure OpenAI; vectors live in Weaviate.

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

## Usage and cost tracking

- **LLM usage** — Logged per agent (planner, selector, general_rag, technical_rlm, synthesis) and as a **pipeline grand total**. If the chat deployment has known pricing (e.g. gpt-5-mini in `KNOWN_MODEL_PRICING`), an estimated cost in USD is logged at the end.
- **Embedding usage** — Logged for **pipeline** use only: search (one embed per subquery) and General agent chunk retrieval. Weaviate ingestion is not counted. If the embed deployment has known pricing (e.g. text-embedding-3-small), an estimated cost is logged after the embedding grand total.
