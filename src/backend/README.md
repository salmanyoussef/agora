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
- **Technical** (frontend choice or `USE_ONLY_GENERAL_AGENT=False`) — The **dataset selector** decides per dataset whether to run **RAG** or **technical** extraction. The Technical agent: detects structure (tabular/records), parses into dataframe/records, builds technical context (schema + preview), and runs **DSPy RLM** (Recursive Language Model) so the model explores the data in a sandboxed Python REPL (code + `llm_query`). Resources that are not structured (e.g. PDF) get the same text extraction as the General agent and are included in the technical context. So Technical mode uses **both** general and technical analysis depending on what the pipeline chooses for each resource.

## Technical agent & RLM

The Technical agent uses **DSPy RLM** when available: the model writes Python in a sandboxed REPL to explore the technical context (filter, aggregate, etc.) and calls `llm_query()` for semantic extraction, then `SUBMIT(answer)`. This requires **Deno** (for the WASM sandbox). If Deno is not installed, the agent falls back to a single `dspy.Predict` call. To enable full RLM: run `agora-setup-repl` or `uv run python -m app.scripts.setup_repl` (see `[tool.agora.repl]` in `pyproject.toml` for prerequisites, e.g. `unzip` on Unix).

## Usage and cost tracking

- **LLM usage** — Logged per agent (planner, selector, general_rag, technical_rlm, synthesis) and as a **pipeline grand total**. If the chat deployment has known pricing (e.g. gpt-5-mini in `KNOWN_MODEL_PRICING`), an estimated cost in USD is logged at the end.
- **Embedding usage** — Logged for **pipeline** use only: search (one embed per subquery) and General agent chunk retrieval. Weaviate ingestion is not counted. If the embed deployment has known pricing (e.g. text-embedding-3-small), an estimated cost is logged after the embedding grand total.
