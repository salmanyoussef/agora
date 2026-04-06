# Agora

**Agora** is a question-answering pipeline over French open government data ([data.gouv.fr](https://www.data.gouv.fr)). You ask a question in natural language; Agora plans subqueries, searches the dataset catalogue, selects relevant datasets, retrieves and extracts content (RAG or technical), and synthesizes a single, attributed answer.

**Framework intent.** This repository is meant to serve as a **reusable baseline** for other scholarly work and projects that use open government data (OGD): the same **systemic, agentic** pattern---planner, hybrid catalogue search, selector, dual evidence paths, synthesis---can be re-targeted to other national or regional portals, extended with new agents or evaluations, or adapted for accessibility-focused deployments. The goal is easier, more **accessible** access to open data through grounded natural-language interfaces, not a one-off demo tied to a single catalogue.

---

## What Agora does

- **Plan** — Splits your question into intent and subqueries.
- **Search** — Hybrid (vector + keyword) search over indexed dataset metadata (Weaviate).
- **Select** — Per-dataset choice of RAG vs technical extraction (when Technical mode is enabled).
- **Answer** — For each selected dataset: download resources, then either **RAG only** (General mode) or **RAG + technical per resource** (Technical mode); finally **synthesize** one answer with clear source attribution.

Embeddings use **Azure OpenAI**; the chat/synthesis model is configured via DSPy (Azure OpenAI chat). Vectors are stored in **Weaviate** (self-hosted, BYOV).

### General vs Technical mode (frontend)

The UI lets you choose **General** or **Technical** before submitting a question:

- **General** — The pipeline uses **only RAG**: download → extract text → chunk → embed → semantic retrieval → LLM answer. No RLM (reasoning language model), no technical/computation path. Best when you want a quick, document-style answer from dataset contents.
- **Technical** — The pipeline uses **both** general and technical analysis **per resource**: for each relevant dataset/resource, the **dataset selector** decides whether to run **RAG** (general) or **technical** extraction. Technical resources are parsed into structured data (tables/records), explored via an RLM (sandboxed Python REPL), and can include fallback text extraction for non-structured files (e.g. PDFs). So by choosing Technical, you get the best of both: RAG where a sample of text is enough, and structured exploration + computation where the data is tabular or record-based.

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
- **Mode choice (General vs Technical)** — Frontend toggle: General = RAG only; Technical = per-resource RAG or technical (selector-driven)
- Multi-step agent pipeline (planner → search → selector → RAG/technical → synthesis)
- **Technical agent** — Structure detection (tabular/records), parse to dataframe/records, RLM (DSPy) exploration in a sandboxed REPL; unsuitable resources get the same text extraction as the General agent
- Streaming progress and final answer via SSE
- RAG with chunking, embedding-based retrieval, and per-resource fairness
- Source-attributed synthesis (dataset/source labels in the answer)
- **LLM and embedding usage + cost estimates** — Per-agent token logs and pipeline grand total with optional pricing (e.g. gpt-5-mini, text-embedding-3-small)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
