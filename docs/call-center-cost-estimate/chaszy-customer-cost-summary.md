# Chaszy + Agora-style pipeline — customer cost summary

**Context:** [Chasms.com](https://www.chasms.com/) offers **Chaszy**, an AI assistant for support agents (included in **Agent Pro** at **$29/month** per seat). This note estimates what it could cost **customers** and **Chasms** if Chaszy used an **Agora-like pipeline** for **advanced reporting** (plan → search knowledge → **technical** analysis on tabular/structured resources → synthesized answer), not just quick KB snippets.

**Planning assumption for technical runs:** **$0.10 per pipeline query** (same as the call-center model in this folder).  
**Lighter path (RAG / lookup only):** typically **~3–5× cheaper** than technical in pilot systems; use **~$0.02–0.03/query** for budgeting until you measure General-only runs.

---

## What changes for Chaszy users

| Today (typical KB chat) | With “advanced reporting” pipeline |
|-------------------------|-------------------------------------|
| Short answers from articles | Multi-step: search corpus, pick dataset/resource, download/parse, **compute in REPL** (counts, trends, comparisons), then narrative report |
| Low token use per turn | Higher tokens + optional many RLM steps |
| Good for “what’s the policy?” | Good for “which SKU had the most returns last quarter by region?” |

Each **advanced report** ≈ **one full pipeline query** in the cost model below.

---

## Volume assumptions (per agent seat)

Aligned with the call-center scenario in `README.md`, scaled to **one** Agent Pro user:

| Parameter | Conservative | Typical | Heavy |
|-----------|-------------:|--------:|------:|
| Calls / day | 40 | 40 | 40 |
| Calls using Chaszy advanced report | 20% | 50% | 50% |
| Advanced queries / assisted call | 1 | 2 | 2 |
| **Queries / day** | **8** | **40** | **40** |
| Working days / month | 22 | 22 | 22 |
| **Queries / month** | **176** | **880** | **880** |

Formula: `queries/day = calls × usage_rate × queries_per_call`

---

## Variable API cost per agent / month (technical @ $0.10)

| Profile | Queries / month | API cost / agent / month |
|---------|----------------:|-------------------------:|
| Conservative | 176 | **$17.60** |
| Typical (same as call-center model) | 880 | **$88.00** |
| Heavy (same queries, 100% technical) | 880 | **$88.00** |

**Per day (typical):** 40 queries × $0.10 = **$4.00 / agent / day**

**Per advanced report (typical, 2 queries/call):** **$0.20** per assisted call  
**Per call (blended, 50% adoption):** **$0.10** per call

---

## What this means for Chasms **customers** (support organizations)

Customers do **not** pay OpenAI directly unless Chasms passes costs through. Effective cost depends on **pricing model**:

### A. Included in subscription (Chasms absorbs API)

| Chasms plan | Seat price | Typical API COGS (technical) | Rough API % of revenue |
|-------------|----------:|-----------------------------:|------------------------:|
| Agent Pro Monthly | $29 | up to **$88** | up to **~300%** |
| Agent Pro Annual | ~$24.17/mo | up to **$88** | up to **~360%** |

At **typical** technical usage, **API alone exceeds** the $29 seat price. That implies:

- Advanced reporting must be **metered**, **capped**, or **General-first** with technical on demand; or  
- Priced on **Enterprise** tiers with higher seat fees and usage bundles.

### B. Usage pass-through (customer pays marginal cost)

Illustrative **add-on** to customer (technical @ $0.10, 22-day month):

| Profile | Extra / agent / month |
|---------|----------------------:|
| Conservative | **~$18** |
| Typical | **~$88** |

Or **per report**: **$0.10** (1 query) to **$0.20** (2 queries) per assisted call.

### C. Hybrid (included allowance + overage)

Example packaging for enterprise buyers:

| Tier | Included advanced reports / seat / month | Overage |
|------|------------------------------------------:|---------|
| Starter | 50 (~$5 COGS) | $0.12 / report |
| Pro | 200 (~$20 COGS) | $0.10 / report |
| Unlimited* | — | Requires **$99+** seat or hard caps |

\*Unlimited technical at $0.10 is not viable on a $29 seat without heavy subsidization.

---

## Platform-level view (Chasms as vendor)

If **90,000 professionals / month** use Chasms (per their marketing) and a **small fraction** adopt advanced reporting:

| Adoption | Users on advanced | Queries/user/mo (typical) | Monthly API (technical) |
|----------|------------------:|--------------------------:|------------------------:|
| 1% | 900 | 880 | **~$79,200** |
| 5% | 4,500 | 880 | **~$396,000** |
| 10% | 9,000 | 880 | **~$792,000** |

Scale **dominates** economics — metering and mode selection (General vs technical) are product decisions, not afterthoughts.

---

## Organization examples (monthly API only)

| Customer size | Seats | Usage profile | Queries / month | API @ $0.10 |
|---------------|------:|---------------|----------------:|------------:|
| Small shop | 10 | Conservative | 1,760 | **$176** |
| Mid CC | 50 | Typical | 44,000 | **$4,400** |
| Large CC | 200 | Typical | 176,000 | **$17,600** |
| Enterprise | 1,000 | Typical | 880,000 | **$88,000** |

Add **~15–25%** buffer for retries, longer reports, and price changes. Add **hosting + embedding ingestion** separately (not in $0.10).

---

## Comparison: General (lookup) vs Technical (reporting)

For **documentation-style** answers (policy, device steps), prefer **General (RAG)**:

| Mode | Planning $/query | Typical / agent / month (880 q) |
|------|-----------------:|--------------------------------:|
| General (illustrative) | $0.025 | **~$22** |
| Technical | $0.10 | **$88** |

**Advanced reporting** should default to **technical only when needed** (CSV/metrics/compute); otherwise customer cost stays closer to **$20–30/seat/month** API, which can fit inside or near Pro pricing.

---

## Recommended commercial framing for Chaszy

1. **Base Chaszy** — KB Q&A (General / low cost), aligned with current Pro value.  
2. **Chaszy Reports** (or similar) — Technical pipeline, **metered** with visible row counts and cost transparency (as in Agora UI).  
3. **Enterprise** — Pooled queries, SLAs, custom corpus, **$X/seat + $Y per 1,000 advanced reports**.  
4. **Pilot** — Measure real **p50/p90 $/query** on customer KB (PDF-heavy vs CSV-heavy corpora change totals a lot).

---

## One-slide takeaway for a business meeting

> If every Chaszy **advanced report** runs the full **technical** pipeline at **$0.10**, a typical agent (~40 advanced queries/day) costs about **$88/month in API** — **3×** the current **$29** Pro seat. Customers only “feel” that if usage is uncapped; Chasms should bundle **light** usage in Pro and sell **metered advanced reporting** for analytics-style questions, or keep those answers on the cheaper **General** path where possible.

---

## Related files in this folder

- `README.md` — 200-agent call-center scenario (same $0.10 technical assumption)  
- `presentation.html` — slide deck for internal / investor-style review  
- `diagrams.md` — Mermaid funnel and sensitivity charts  

**Disclaimer:** Chasms/Chaszy list prices from [chasms.com/pricing](https://www.chasms.com/pricing) (May 2026). API rates are **planning** assumptions; validate with production telemetry before customer contracts.
