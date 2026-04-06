# Agora

**Agora** implements the **AGORA protocol**: a fixed, auditable pipeline from a natural-language **question** to a single **grounded answer** over a national open-data **catalogue** ([data.gouv.fr](https://www.data.gouv.fr)). This README summarizes the *scientific* view (objects, stages, state); runbooks and APIs live under **`src/`**.

**Framework intent.** The same staged pattern can be re-targeted to other API-backed portals: swap ingestion and prompts, keep the protocol shape. The goal is **accessible**, **source-attributed** access to open government data, not a one-off demo.

---

## The AGORA protocol (formal view)

**Objects.**

| Symbol | Meaning |
|--------|---------|
| **D** | Finite set of **dataset identifiers** exposed by the portal catalogue. |
| **I** | **Indexed catalogue**: metadata records with dense + lexical (hybrid) features for search. |
| **Q** | User **question** (natural language). |

A **run** maps **Q → A** (final answer) in **five stages** with a single orchestrator—no ad hoc message graph between agents.

**Stages (control flow).**

1. **Plan.** Produce a plan **P = (ι, {q₁,…,qₖ})**: intent **ι** and retrieval-oriented **subqueries** **qₗ** (often **k = 1**).
2. **Retrieve.** For each **ℓ**, hybrid search over **I** returns ranked hits **Hₗ ⊆ D** (scores + metadata). Hits may be deduplicated across **ℓ**.
3. **Select.** Output **S = {(d, m_d, r_d)}** where **d ∈ D**, **m_d ∈ {RAG, TECH}**, optional rationale **r_d**. Possibly **S = ∅** if nothing is relevant.
4. **Execute evidence.** For each **(d, m_d) ∈ S**, download portal resources for **d**, extract content, and form an evidence block **e_d** (passage-style for **RAG**, structured / tool-assisted for **TECH**).
5. **Synthesize.** Map **(Q, P, {e_d}, refs(S)) → A**: one answer with citations, using portal titles and URLs from **refs(S)**.

**Coordination vs cognition.**

- **Agents reason** — Planner, selector, General, Technical, and synthesis each apply their own policy (LLM and/or tools) to local inputs and emit typed artifacts (plans, **S**, evidence blocks).
- **Protocol coordinates** — The orchestrator enforces order, deduplicates hits across subqueries, routes each **d** to **RAG** or **TECH**, merges evidence in a **fixed order**, and forwards citation fields so **A** points to real catalogue rows.

**Protocol state (between stages).**  
Write **θ₁ = P** after planning; **θ₂ = (P, {Hₗ})** after retrieval; **θ₃ = (P, S)** after selection; **θ₄ = (P, {e_d})** after execution. Synthesis consumes **θ₄** plus **refs(S)**.  
**Aggregation** stacks evidence in deterministic order—schematically **E = ⨁_{d ∈ S↓} e_d** with **S↓** the ordered selection—so every span of **A** stays attributable to some **d ∈ D**.

---

## What this repository implements

In code, the five stages correspond to: **plan** → **search** (Weaviate hybrid index over **I**) → **select** → **execute** (General and/or Technical agents) → **synthesize**. Embeddings use **Azure OpenAI**; vectors live in **Weaviate** (self-hosted, bring-your-own vectors); agents are wired with **DSPy**.

- **Plan** — **Q → P** (intent + subqueries).
- **Search** — Each **qₗ → Hₗ** over indexed metadata.
- **Select** — **(Q, P, {Hₗ}) → S** with per-dataset **RAG** vs **TECH** when Technical mode is enabled (General mode forces **RAG** for all selected **d**).
- **Answer** — For each **d** in **S**: download resources, **e_d** via RAG path and/or technical path; then **P, {e_d}, refs → A** with attribution.

### General vs Technical mode (UI)

- **General** — Only the **RAG** path for every selected dataset: download → extract → chunk → embed → answer from context. No technical / RLM path.
- **Technical** — The selector assigns **m_d ∈ {RAG, TECH}** per dataset; **TECH** uses structured parsing and an RLM-style loop (e.g. sandboxed REPL) when the question needs lookup or aggregation over tables/records.

---

## Repository layout

| Path | Description |
|------|-------------|
| **`src/`** | Application: backend (FastAPI), frontend, pipelines, agents, embeddings. See **`src/README.md`**. |
| **`notebooks/`** | Research, prototyping, and experiments. |

Prototype in `notebooks/`, ship stable pieces in `src/`.

---

## Quick links

- **Run, ingest, search, streaming:** [**`src/README.md`**](src/README.md)
- **Backend quickref:** [**`src/backend/README.md`**](src/backend/README.md)

---

## Core capabilities

- Hybrid (vector + keyword) search over the French open-data catalogue (**I**)
- **General vs Technical** UI modes (override to all-**RAG** vs selector-driven **RAG**/**TECH**)
- Linear multi-agent pipeline consistent with the protocol above
- Technical path: tabular/record parsing, RLM-style exploration in a sandbox; unstructured files fall back to text extraction
- Streaming progress (SSE) and source-attributed **A**
- RAG with chunking and embedding-based chunk retrieval where enabled
- Per-stage LLM/embedding usage and optional cost estimates

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
