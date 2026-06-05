# Diagrams — call center API cost

Paste into Mermaid Live Editor or any Markdown viewer with Mermaid support.

## 1. Volume funnel (queries per day)

```mermaid
flowchart TB
  A["200 agents"] --> B["40 calls / agent / day"]
  B --> C["8,000 calls / day"]
  C --> D["50% use assist tool"]
  D --> E["4,000 assisted calls / day"]
  E --> F["× 2 queries / call"]
  F --> G["8,000 queries / day"]
  G --> H["× $0.10 / query"]
  H --> I["$800 API cost / day"]
```

## 2. Monthly cost waterfall (22 working days)

```mermaid
flowchart LR
  subgraph daily ["Per day"]
    Q1["8,000 queries"]
    C1["$800"]
    Q1 --> C1
  end
  subgraph monthly ["Per month × 22 days"]
    Q2["176,000 queries"]
    C2["$17,600"]
    Q2 --> C2
  end
  subgraph annual ["Per year × 252 days"]
    Q3["2.02M queries"]
    C3["$201,600"]
    Q3 --> C3
  end
  daily --> monthly --> annual
```

## 3. Cost split (illustrative — if unbundling later)

```mermaid
pie title Hypothetical split of $0.10/query (illustrative only)
  "Technical RLM + agents" : 70
  "Synthesis + planning" : 20
  "Embeddings + search" : 10
```

*Pie is for discussion only; your pilot should replace with measured splits.*

## 4. Sensitivity — monthly API cost (22 days)

```mermaid
xychart-beta
  title "Monthly API cost vs. usage rate (200 agents, 40 calls, 2 queries, $0.10)"
  x-axis [25%, 50%, 75%, 100%]
  y-axis "USD (thousands)" 0 --> 36
  bar [8.8, 17.6, 26.4, 35.2]
```

*At 50% usage → $17.6k/month. At 25% → $8.8k; at 100% → $35.2k.*
