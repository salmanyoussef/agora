from __future__ import annotations

import logging
import tempfile
import time
from typing import Any, Dict, List, Optional

import dspy

from app.clients.data_gouv import DataGouvDatasetsClient, extract_resource_urls
from app.models.execution_result import ExecutionResult
from app.services.dspy_setup import configure_dspy, log_last_lm_call
from app.services.text_extraction import download_file, extract_text_from_file

logger = logging.getLogger(__name__)

# Context size knobs for RAG (general) agent
MAX_DATASETS_FOR_RAG = 3
MAX_RESOURCES_PER_DATASET = 5
MAX_ROWS_PER_RESOURCE = 200
MAX_CONTEXT_CHARS = 0  # 0 = no truncation; set to e.g. 8000 to cap total context length

# Chunking + retrieval: split extracted text into chunks, embed and select top-k by similarity to subquery
RAG_CHUNK_MAX_CHARS = 800
RAG_CHUNK_OVERLAP_CHARS = 100
RAG_TOP_K_CHUNKS = 15
RAG_USE_CHUNK_RETRIEVAL = True  # set False to pass full context without chunk retrieval
# Per-resource fairness: cap how many chunks we take from any single resource
RAG_MAX_CHUNKS_PER_RESOURCE = 5


def _resource_metadata_str(
    dataset: Dict[str, Any],
    resource: Optional[Dict[str, Any]],
    url: str,
) -> str:
    """Build a short metadata line for the LLM: dataset title, org, resource format/description, url."""
    title = (dataset.get("title") or dataset.get("name") or "Unknown dataset").strip()
    org_raw = dataset.get("organization")
    if isinstance(org_raw, dict):
        org = (org_raw.get("name") or org_raw.get("title") or "").strip()
    elif isinstance(org_raw, str):
        org = org_raw.strip()
    else:
        org = ""
    parts = [f"Dataset: {title}"]
    if org:
        parts.append(f"Organization: {org}")
    if resource:
        fmt = (resource.get("format") or "").strip()
        desc = (resource.get("description") or "").strip()[:200]
        if fmt:
            parts.append(f"Format: {fmt}")
        if desc:
            parts.append(f"Description: {desc}")
    parts.append(f"URL: {url[:120]}{'…' if len(url) > 120 else ''}")
    return " | ".join(parts)


def _chunk_evidence(evidence_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split each block's content into chunks; return list of {resource_id, metadata, chunk}."""
    out: List[Dict[str, Any]] = []
    for block in evidence_blocks:
        content = (block.get("content") or "").strip()
        if not content:
            continue
        resource_id = block.get("resource_id") or "unknown"
        metadata = block.get("metadata") or ""
        # Split on double newline then by size (same logic as before)
        parts = [p.strip() for p in content.split("\n\n") if p.strip()]
        current: List[str] = []
        current_len = 0
        for p in parts:
            if current_len + len(p) + 2 <= RAG_CHUNK_MAX_CHARS:
                current.append(p)
                current_len += len(p) + 2
            else:
                if current:
                    out.append({
                        "resource_id": resource_id,
                        "metadata": metadata,
                        "chunk": "\n\n".join(current),
                    })
                if RAG_CHUNK_OVERLAP_CHARS > 0 and current:
                    overlap_text = "\n\n".join(current)[-RAG_CHUNK_OVERLAP_CHARS:]
                    current = [overlap_text] if overlap_text.strip() else []
                    current_len = len(overlap_text)
                else:
                    current = []
                    current_len = 0
                if len(p) <= RAG_CHUNK_MAX_CHARS:
                    current.append(p)
                    current_len = len(p)
                else:
                    for i in range(0, len(p), RAG_CHUNK_MAX_CHARS - RAG_CHUNK_OVERLAP_CHARS):
                        chunk = p[i : i + RAG_CHUNK_MAX_CHARS]
                        if chunk.strip():
                            out.append({
                                "resource_id": resource_id,
                                "metadata": metadata,
                                "chunk": chunk,
                            })
        if current:
            out.append({
                "resource_id": resource_id,
                "metadata": metadata,
                "chunk": "\n\n".join(current),
            })
    return out


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _retrieve_relevant_chunks(
    chunks: List[Dict[str, Any]],
    subquery: str,
    top_k: int,
    embed_client: Any,
    max_per_resource: int = RAG_MAX_CHUNKS_PER_RESOURCE,
) -> List[Dict[str, Any]]:
    """Embed subquery and chunks, rank by similarity, return top-k with per-resource cap."""
    if not chunks:
        return []
    chunk_texts = [c["chunk"] for c in chunks]
    try:
        texts = [subquery] + chunk_texts
        embeddings = embed_client.embed_texts(texts)
    except Exception as e:
        logger.warning("Chunk retrieval embedding failed, using all chunks: %s", e)
        return chunks[:top_k]
    if len(embeddings) != len(texts):
        return chunks[:top_k]
    query_emb = embeddings[0]
    chunk_embs = embeddings[1:]
    scored = [
        (_cosine_similarity(query_emb, ce), i)
        for i, ce in enumerate(chunk_embs)
    ]
    scored.sort(key=lambda x: -x[0])
    # Take top-k but cap how many we take from any single resource (fairness)
    selected: List[Dict[str, Any]] = []
    per_resource_count: Dict[str, int] = {}
    for _, i in scored:
        if len(selected) >= top_k:
            break
        c = chunks[i]
        rid = c.get("resource_id") or "unknown"
        n = per_resource_count.get(rid, 0)
        if n >= max_per_resource:
            continue
        selected.append(c)
        per_resource_count[rid] = n + 1
    return selected


_EMBED_CLIENT: Optional[Any] = None


def _get_embed_client() -> Optional[Any]:
    """Return a shared Azure embedding client (lazy singleton). Reuse avoids leaking SSL connections."""
    global _EMBED_CLIENT
    if _EMBED_CLIENT is not None:
        return _EMBED_CLIENT
    try:
        from app.embeddings.azure import get_embedding_client
        _EMBED_CLIENT = get_embedding_client()
        return _EMBED_CLIENT
    except Exception as e:
        logger.debug("Embed client not available: %s", e)
        return None


class AnswerFromContext(dspy.Signature):
    """
    You are the RAG (retrieval-augmented) step in a multi-step question-answering pipeline
    over French open government data.

    Pipeline role: The user asked a question. A planner split it into subqueries; a search
    found relevant datasets; a selector chose which datasets to use. For each selected
    dataset, resources were downloaded and their text extracted. You receive the chunks
    or excerpts from those resources that were judged most relevant to this subquery. Your
    job is to answer the subquery using only this context—no external knowledge. If the
    context does not support a full answer, say so and summarize what it does contain.
    Answer in the same language as the question (typically French).

    What your output is used for: Your answer is not shown directly to the user. It becomes
    one evidence block that is combined with evidence from other datasets (and other
    subqueries). A synthesis step then reads all these blocks and produces a single
    final answer for the user. So write a clear, self-contained summary that the synthesis
    can cite or merge—factual, attributable to the context, and concise enough to be
    combined with other evidence without redundancy.
    """

    pipeline_role = dspy.InputField(
        desc="Short reminder: you are the RAG step; you answer from the provided context only."
    )
    context = dspy.InputField(
        desc="Relevant chunks or excerpts from one or more dataset resources. Each section is prefixed with [Source: Dataset title | Organization | Format | Description | URL] so you know the provenance of the content."
    )
    question = dspy.InputField(desc="The subquery to answer from the context.")
    focus = dspy.InputField(
        desc="Why this dataset was chosen for this subquery (selector reasoning)."
    )
    answer = dspy.OutputField(
        desc="A concise, self-contained evidence block based only on the context (same language as the question). This will be merged with other evidence and passed to a synthesis step to produce the final user answer—so be factual, attributable, and avoid redundancy. Say when the context is insufficient."
    )


PIPELINE_ROLE_REMINDER = (
    "You are the RAG step in a pipeline over French open data. "
    "Answer using only the provided context; no external knowledge."
)


def _rag_answer(
    question: str,
    context: str,
    focus: str = "",
    pipeline_role: str = "",
) -> str:
    configure_dspy()
    focus_text = focus.strip() if focus else "(No specific focus provided.)"
    role_text = (pipeline_role or PIPELINE_ROLE_REMINDER).strip()
    logger.info(
        "GeneralAgent RAG call started: question_len=%d context_len=%d focus_len=%d",
        len(question),
        len(context),
        len(focus_text),
    )
    started_at = time.perf_counter()
    pred = dspy.Predict(AnswerFromContext)(
        pipeline_role=role_text,
        context=context,
        question=question,
        focus=focus_text,
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info("GeneralAgent RAG call completed in %.1f ms", elapsed_ms)
    try:
        usage = pred.get_lm_usage()
        if usage is not None:
            logger.debug("RAG Prediction.get_lm_usage(): %s", usage)
    except Exception as e:
        logger.debug("RAG get_lm_usage not available: %s", e)
    log_last_lm_call(caller="general_rag")
    logger.info("GeneralAgent DSPy response trace (last call):")
    dspy.inspect_history(n=1)
    return pred.answer or "No answer could be produced from the given context."


class GeneralAgent:
    def run(
        self,
        subquery: str,
        datasets: List[Dict[str, Any]],
        dataset_reasoning: str = "",
    ) -> ExecutionResult:
        logger.info(
            "GeneralAgent.run: subquery=%r datasets_count=%d reasoning_len=%d",
            subquery[:200] + ("..." if len(subquery) > 200 else ""),
            len(datasets),
            len(dataset_reasoning),
        )
        for d in datasets[:MAX_DATASETS_FOR_RAG]:
            logger.info(
                "GeneralAgent candidate dataset | id=%s | title=%s | tags=%s",
                d.get("dataset_id") or d.get("id"),
                d.get("title"),
                d.get("tags"),
            )
        client = DataGouvDatasetsClient()
        evidence_blocks: List[Dict[str, Any]] = []
        dataset_ids_to_try = [d.get("dataset_id") or d.get("id") for d in datasets[:MAX_DATASETS_FOR_RAG]]
        logger.debug("GeneralAgent: dataset_ids_to_try=%s", dataset_ids_to_try)

        with tempfile.TemporaryDirectory(prefix="agora_general_") as out_dir:
            logger.debug("GeneralAgent: download out_dir=%s", out_dir)
            for dataset in datasets[:MAX_DATASETS_FOR_RAG]:
                dataset_id = dataset.get("dataset_id") or dataset.get("id")
                if not dataset_id:
                    logger.debug("GeneralAgent: skipping dataset with no id: %s", dataset)
                    continue
                try:
                    ds = client.get_dataset(dataset_id)
                    url_items = extract_resource_urls(ds)
                    logger.info(
                        "GeneralAgent: dataset_id=%s resources_count=%d",
                        dataset_id,
                        len(url_items),
                    )
                except Exception as e:
                    logger.warning("Failed to fetch dataset %s: %s", dataset_id, e)
                    evidence_blocks.append({
                        "resource_id": dataset_id,
                        "metadata": f"Dataset {dataset_id} fetch failed",
                        "content": str(e),
                    })
                    continue

                for res_idx, item in enumerate(url_items[:MAX_RESOURCES_PER_DATASET]):
                    url = item.get("url")
                    resource = item.get("resource")
                    if not url:
                        continue
                    resource_id = f"{dataset_id}_{res_idx}"
                    metadata_str = _resource_metadata_str(ds, resource, url)
                    try:
                        path = download_file(url, out_dir, resource=resource)
                        logger.debug(
                            "GeneralAgent: downloaded dataset_id=%s url=%s -> path=%s",
                            dataset_id,
                            url[:80] + ("..." if len(url) > 80 else ""),
                            path,
                        )
                        text = extract_text_from_file(
                            path, max_rows=MAX_ROWS_PER_RESOURCE, resource=resource
                        )
                        if isinstance(text, list):
                            for doc in text:
                                evidence_blocks.append({
                                    "resource_id": resource_id,
                                    "metadata": metadata_str,
                                    "content": doc,
                                })
                            logger.debug(
                                "GeneralAgent: extracted list of %d docs from %s",
                                len(text),
                                path,
                            )
                        else:
                            evidence_blocks.append({
                                "resource_id": resource_id,
                                "metadata": metadata_str,
                                "content": text,
                            })
                            logger.debug(
                                "GeneralAgent: extracted single doc len=%d from %s",
                                len(text),
                                path,
                            )
                    except Exception as e:
                        logger.warning("Download/extract failed for %s: %s", url, e)
                        evidence_blocks.append({
                            "resource_id": resource_id,
                            "metadata": metadata_str,
                            "content": f"[Download/extract failed] {e}",
                        })

        def _format_context_from_blocks(blocks: List[Dict[str, Any]], content_key: str = "content") -> str:
            """Format blocks for LLM: each block has [Source: metadata] then content."""
            parts = []
            for b in blocks:
                meta = (b.get("metadata") or "").strip()
                body = (b.get(content_key) or "").strip()
                if not body:
                    continue
                if meta:
                    parts.append(f"[Source: {meta}]\n{body}")
                else:
                    parts.append(body)
            return "\n\n---\n\n".join(parts) if parts else ""

        # Optionally chunk and retrieve most relevant chunks by embedding similarity (with per-resource cap)
        if RAG_USE_CHUNK_RETRIEVAL and evidence_blocks:
            chunks = _chunk_evidence(evidence_blocks)
            embed_client = _get_embed_client()
            if embed_client and len(chunks) > 1:
                selected = _retrieve_relevant_chunks(
                    chunks,
                    subquery,
                    top_k=RAG_TOP_K_CHUNKS,
                    embed_client=embed_client,
                    max_per_resource=RAG_MAX_CHUNKS_PER_RESOURCE,
                )
                context = _format_context_from_blocks(selected, content_key="chunk")
                logger.info(
                    "GeneralAgent: chunk retrieval: %d chunks -> top %d (per-resource cap %d), context_len=%d",
                    len(chunks),
                    len(selected),
                    RAG_MAX_CHUNKS_PER_RESOURCE,
                    len(context),
                )
            else:
                context = _format_context_from_blocks(evidence_blocks)
                if embed_client and len(chunks) == 1:
                    logger.debug("GeneralAgent: single chunk, no retrieval")
        else:
            context = _format_context_from_blocks(evidence_blocks)
        if MAX_CONTEXT_CHARS and len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated for length.]"
        if not context.strip():
            context = "(No content could be extracted from the selected datasets.)"
        logger.info(
            "GeneralAgent: evidence_blocks=%d context_len=%d",
            len(evidence_blocks),
            len(context),
        )

        rag_answer = _rag_answer(
            subquery,
            context,
            focus=dataset_reasoning,
            pipeline_role=PIPELINE_ROLE_REMINDER,
        )
        logger.info(
            "GeneralAgent: RAG answer len=%d",
            len(rag_answer),
        )

        return ExecutionResult(
            mode="rag",
            subquery=subquery,
            evidence=rag_answer,
        )
