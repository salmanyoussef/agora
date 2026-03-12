# Agora pipeline — agent structure and data flow

This diagram reflects the current setup in `src/backend/app/agents/orchestrator.py` and related agents.

## Mermaid flowchart

```mermaid
flowchart TB
    subgraph input[" "]
        Q["User question"]
    end

    subgraph orchestrator["Orchestrator"]
        direction TB
        Q --> Planner
        Planner -->|"QueryPlan (intent + subqueries)"| Loop
        Loop -->|"per subquery"| Search
        Search -->|"hits"| Selector
        Selector -->|"DatasetSelection (dataset_id, execution_mode: rag|technical, reasoning)"| Branch
        Branch -->|"per selected dataset"| GeneralOrTechnical
        GeneralOrTechnical -->|"ExecutionResult (evidence)"| Collect
        Collect -->|"evidence_blocks"| Synthesis
        Synthesis -->|"answer"| Response
    end

    subgraph agents["Agents"]
        Planner["PlannerAgent<br>(ChainOfThought)<br>question → intent + subqueries"]
        Selector["DatasetSelectorAgent<br>(ChainOfThought)<br>question, subquery, hits → selected_datasets<br>with execution_mode + reasoning"]
        General["GeneralAgent<br>(RAG)<br>subquery, hit, reasoning → evidence"]
        Technical["TechnicalAgent<br>(RLM or Predict)<br>subquery, hit, reasoning → evidence"]
        Synthesis["SynthesisAgent<br>(ChainOfThought)<br>question, evidence, context → answer"]
    end

    subgraph general_internals["GeneralAgent internal"]
        direction LR
        G1["download resources<br>(data.gouv)"] --> G2["extract text"]
        G2 --> G3["chunk"]
        G3 --> G4["embed + top-k retrieval"]
        G4 --> G5["AnswerFromContext<br>(Predict)<br>→ evidence"]
    end

    subgraph technical_internals["TechnicalAgent internal"]
        direction LR
        T1["download resources"] --> T2["detect structure"]
        T2 --> T3["parse into records<br>(CSV/JSON/XLSX…)"]
        T3 --> T4["build_technical_context<br>(schema + preview)"]
        T4 --> T5["RLM explores (REPL)<br>or Predict fallback<br>→ evidence"]
    end

    Q --> Planner
    Planner -.->|"plan"| Loop
    %% search_datasets is a retrieval utility (non-agent)
    Search["search_datasets<br>(retrieval, non-agent)<br>query → embed → WeaviateStore.search<br>(hybrid vector + BM25)<br>→ hits"]
    Loop --> Search
    Search --> Selector
    Selector --> Branch
    Branch -->|"execution_mode=rag<br>or use_only_general_agent"| General
    Branch -->|"execution_mode=technical"| Technical
    General --> Collect
    Technical --> Collect
    Collect --> Synthesis
    Synthesis --> Response["AgentResponse<br>(answer, plan, evidence, hits, usage)"]
    Response --> output["Final answer to user"]

    General -.-> general_internals
    Technical -.-> technical_internals
```

## Simplified sequence (who calls whom)

```mermaid
sequenceDiagram
    participant User
    participant Orch as Orchestrator
    participant Planner as PlannerAgent
    participant Search as search_datasets
    participant Store as WeaviateStore
    participant Selector as DatasetSelectorAgent
    participant General as GeneralAgent
    participant Technical as TechnicalAgent
    participant Synthesis as SynthesisAgent

    User->>Orch: question, k, use_only_general_agent
    Orch->>Planner: run(question)
    Planner-->>Orch: QueryPlan (intent, subqueries)

    loop For each subquery
        Orch->>Search: search_datasets(subquery, k)
        Search->>Store: hybrid search
        Store-->>Search: hits
        Search-->>Orch: hits, embed_usage

        Orch->>Selector: run(question, subquery, hits)
        Selector-->>Orch: DatasetSelection (selected_datasets: id, mode, reasoning)

        loop For each selected dataset
            alt execution_mode = rag OR use_only_general_agent
                Orch->>General: run(subquery, [hit], reasoning)
                General-->>Orch: ExecutionResult (evidence)
            else execution_mode = technical
                Orch->>Technical: run(subquery, [hit], reasoning)
                Technical-->>Orch: ExecutionResult (evidence)
            end
        end
    end

    Orch->>Orch: evidence_text = join(evidence_blocks)
    Orch->>Synthesis: run(question, evidence_text, context)
    Synthesis-->>Orch: answer, lm_usage
    Orch-->>User: AgentResponse (answer, plan, evidence, hits, usage)
```

## Data types at boundaries

| Step | Input | Output |
|------|--------|--------|
| **PlannerAgent** | `question: str` | `QueryPlan` (intent, subqueries[], lm_usage) |
| **search_datasets** | `query_text: str`, `k`, `alpha` | `(hits: list[dict], embed_usage)` |
| **DatasetSelectorAgent** | `question`, `subquery`, `hits` | `DatasetSelection` (selected_datasets: dataset_id, execution_mode, reasoning; lm_usage) |
| **GeneralAgent** | `subquery`, `datasets` (hits), `dataset_reasoning` | `ExecutionResult` (mode=rag, subquery, evidence, lm_usage, embed_usage) |
| **TechnicalAgent** | `subquery`, `hits`, `dataset_reasoning` | `ExecutionResult` (mode=technical, subquery, evidence, lm_usage) |
| **SynthesisAgent** | `question`, `evidence` (concatenated blocks), `context` | `(answer: str, lm_usage)` |
| **Orchestrator** | `question`, `k`, `use_only_general_agent` | `AgentResponse` (answer, plan, evidence[], hits, user_messages, lm_usage_grand_total, embed_usage_grand_total) |

## One-line pipeline summary

**Question → Planner (subqueries) → [per subquery: search_datasets (hits) → DatasetSelector (rag/technical per dataset) → GeneralAgent or TechnicalAgent (evidence)] → concatenate evidence → SynthesisAgent (answer) → AgentResponse.**
