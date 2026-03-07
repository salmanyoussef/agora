# Agora

**Agora** is a question-answering pipeline over French open government data ([data.gouv.fr](https://www.data.gouv.fr)). You ask a question in natural language; Agora plans subqueries, searches the dataset catalogue, selects relevant datasets, retrieves and extracts content (RAG or technical), and synthesizes a single, attributed answer.

---

## What Agora does

- **Plan** — Splits your question into intent and subqueries.
- **Search** — Hybrid (vector + keyword) search over indexed dataset metadata (Weaviate).
- **Select** — Per-dataset choice of RAG vs technical extraction.
- **Answer** — For each selected dataset: download resources, extract text, run RAG (chunk + embed + retrieve) or technical extraction, then **synthesize** one final answer with clear source attribution.

Embeddings use **Azure OpenAI**; the chat/synthesis model is configured via DSPy (Azure OpenAI chat). Vectors are stored in **Weaviate** (self-hosted, BYOV).

---

## Repository layout

| Path | Description |
|------|-------------|
| **`src/`** | Application: backend (FastAPI), frontend, pipelines, agents, embeddings. See **`src/README.md`** for runbooks and API details. |
| **`notebooks/`** | Research, prototyping, and experiments (embedding/retrieval, validation). |

Design: prototype in `notebooks/`, then refactor into production modules in `src/`, keeping a clear split between experimentation and deployment.

---

## Quick links

- **Run the app, ingest, search, streaming:** [**`src/README.md`**](src/README.md)
- **Backend-only quickref:** [**`src/backend/README.md`**](src/backend/README.md)

---

## Core capabilities

- Semantic + keyword search over French open data catalogue
- Multi-step agent pipeline (planner → search → selector → RAG/technical → synthesis)
- Streaming progress and final answer via SSE
- RAG with chunking, embedding-based retrieval, and per-resource fairness
- Source-attributed synthesis (dataset/source labels in the answer)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
