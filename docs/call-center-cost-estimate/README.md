# Call center — API cost estimate (technical mode)

Business-case folder for presenting variable LLM/API cost if every agent assist runs the **full technical pipeline** at **$0.10 per query**.

## Assumptions

| Parameter | Value |
|-----------|------:|
| Agents | 200 |
| Calls per agent per day | 40 |
| Share of calls using the tool | 50% |
| Queries per assisted call (average) | 2 |
| Mode | 100% technical |
| Cost per query (all-in API) | **$0.10** |

**Queries per day**

`200 × 40 × 0.5 × 2 =` **8,000 queries/day**

## Cost summary (API only)

| Period | Queries | Cost @ $0.10 |
|--------|--------:|-------------:|
| Per day | 8,000 | **$800** |
| Per month (22 working days) | 176,000 | **$17,600** |
| Per month (30 calendar days) | 240,000 | **$24,000** |
| Per year (252 working days) | 2,016,000 | **$201,600** |
| Per year (365 calendar days) | 2,920,000 | **$292,000** |

## Unit economics

| Metric | Value |
|--------|------:|
| Cost per agent per month (22 days) | **$88** |
| Cost per assisted call (2 queries) | **$0.20** |
| Cost per call (all calls, blended) | **$0.10** |

## What this includes / excludes

**Included (in the $0.10/query assumption):** chat LLM (planner, selectors, RAG/technical/RLM, synthesis) and embeddings for search/RAG, as one bundled unit cost.

**Not included:** Weaviate/hosting, corpus ingestion & re-indexing, engineering, support, enterprise contract minimums, rate-limit overages, or General-only mode (typically cheaper).

## Files

- `presentation.html` — open in a browser for a slide-style deck (print to PDF for meetings).
- `diagrams.md` — Mermaid source for flow and cost waterfall (paste into Notion, Confluence, or [mermaid.live](https://mermaid.live)).

## Interactive canvas

Open **`call-center-api-cost.canvas.tsx`** from the Cursor Canvases panel (project `canvases/` folder) for charts and tables beside the chat.
