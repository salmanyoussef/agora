from __future__ import annotations

import logging
from typing import Any, Optional, List

import dspy

from app.settings import settings

logger = logging.getLogger(__name__)

# Token keys that we sum when merging usage (OpenAI/Azure style).
_USAGE_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")


def _sum_usage_entry(entry: dict[str, Any]) -> dict[str, int]:
    """Flatten a single LM usage entry to numeric token counts."""
    out: dict[str, int] = {}
    for key in _USAGE_TOKEN_KEYS:
        val = entry.get(key)
        if isinstance(val, (int, float)):
            out[key] = int(val)
    if "total_tokens" not in out and "prompt_tokens" in out and "completion_tokens" in out:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def format_lm_usage(usage: dict[str, Any] | None) -> str:
    """Format DSPy get_lm_usage() dict for log messages (per-LM and totals)."""
    if not usage or not isinstance(usage, dict):
        return "no usage data"
    parts: List[str] = []
    total_prompt = total_completion = 0
    for lm_name, entry in usage.items():
        if not isinstance(entry, dict):
            continue
        summed = _sum_usage_entry(entry)
        p = summed.get("prompt_tokens", 0) or summed.get("input_tokens", 0)
        c = summed.get("completion_tokens", 0) or summed.get("output_tokens", 0)
        t = summed.get("total_tokens", p + c)
        total_prompt += p
        total_completion += c
        parts.append(f"{lm_name}: prompt={p} completion={c} total={t}")
    if parts:
        parts.append(f"TOTAL: prompt={total_prompt} completion={total_completion} total={total_prompt + total_completion}")
    return " | ".join(parts) if parts else "no token counts"


def merge_lm_usage(usage_dicts: List[dict[str, Any] | None]) -> dict[str, dict[str, int]]:
    """Merge multiple get_lm_usage() results into one usage-by-LM dict with summed tokens."""
    merged: dict[str, dict[str, int]] = {}
    for u in usage_dicts:
        if not u or not isinstance(u, dict):
            continue
        for lm_name, entry in u.items():
            if not isinstance(entry, dict):
                continue
            summed = _sum_usage_entry(entry)
            if lm_name not in merged:
                merged[lm_name] = {k: 0 for k in _USAGE_TOKEN_KEYS if k in summed}
            for key, val in summed.items():
                merged[lm_name][key] = merged[lm_name].get(key, 0) + val
    for entry in merged.values():
        if "total_tokens" not in entry and "prompt_tokens" in entry and "completion_tokens" in entry:
            entry["total_tokens"] = entry["prompt_tokens"] + entry["completion_tokens"]
    return merged


def log_lm_usage(caller: str, usage: dict[str, Any] | None) -> None:
    """Log LLM token usage at INFO for a single agent/call."""
    if usage:
        logger.info("LLM usage [%s]: %s", caller, format_lm_usage(usage))
    else:
        logger.info("LLM usage [%s]: no usage data", caller)


# Pricing for known models (USD per 1M tokens). Only used when deployment matches.
# gpt-5-mini: https://openai.com/api/pricing/ (example rates)
KNOWN_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5-mini": {
        "input_per_1M": 0.25,
        "output_per_1M": 2.00,
    },
}


def _is_priced_model(lm_key: str) -> bool:
    """True if this LM key corresponds to our configured chat deployment and we have pricing."""
    deployment = getattr(settings, "azure_openai_chat_deployment", "") or ""
    if not deployment:
        return False
    # DSPy LM name is typically "azure/<deployment>"
    if lm_key == deployment or lm_key == f"azure/{deployment}" or lm_key.endswith(f"/{deployment}"):
        return deployment in KNOWN_MODEL_PRICING
    return False


def get_llm_cost_append(grand_total_usage: dict[str, Any] | None) -> str:
    """
    If we have pricing for the configured chat model, return a string to append to the
    LLM GRAND TOTAL log line (e.g. " | cost ~0.05 USD"). Otherwise return "".
    """
    if not grand_total_usage or not isinstance(grand_total_usage, dict):
        return ""
    deployment = getattr(settings, "azure_openai_chat_deployment", "") or ""
    pricing = KNOWN_MODEL_PRICING.get(deployment) if deployment else None
    if not pricing:
        return ""
    total_input = total_output = 0
    for lm_name, entry in grand_total_usage.items():
        if not _is_priced_model(lm_name) or not isinstance(entry, dict):
            continue
        summed = _sum_usage_entry(entry)
        total_input += summed.get("prompt_tokens", 0) or summed.get("input_tokens", 0)
        total_output += summed.get("completion_tokens", 0) or summed.get("output_tokens", 0)
    if total_input == 0 and total_output == 0:
        return ""
    cost_input = (total_input / 1_000_000) * pricing["input_per_1M"]
    cost_output = (total_output / 1_000_000) * pricing["output_per_1M"]
    cost_total = cost_input + cost_output
    return f" | cost ~{cost_total:.6f} USD"


def estimate_and_log_pipeline_cost(grand_total_usage: dict[str, Any] | None) -> None:
    """
    If grand_total_usage is for our configured model and we have pricing, the cost is
    included on the GRAND TOTAL line (orchestrator). Log only when we have no pricing.
    """
    if not grand_total_usage or not isinstance(grand_total_usage, dict):
        return
    deployment = getattr(settings, "azure_openai_chat_deployment", "") or ""
    if KNOWN_MODEL_PRICING.get(deployment) is not None:
        return  # Cost is on the grand total line
    logger.info("LLM cost estimate: no pricing for deployment=%r (only gpt-5-mini has defined rates)", deployment)


# --- Embedding usage (pipeline only: search + General Agent chunk retrieval) ---
# Embedding usage dicts: {"prompt_tokens": N, "total_tokens": N} per call.

def format_embed_usage(usage: dict[str, int] | None) -> str:
    """Format a single embedding usage dict for logging."""
    if not usage or not isinstance(usage, dict):
        return "no usage data"
    p = usage.get("prompt_tokens", 0) or 0
    t = usage.get("total_tokens", 0) or 0
    return f"prompt_tokens={p} total_tokens={t}"


def merge_embed_usage(usage_list: List[dict[str, int] | None]) -> dict[str, int]:
    """Sum prompt_tokens and total_tokens across pipeline embedding calls."""
    total_p = total_t = 0
    for u in usage_list:
        if not u or not isinstance(u, dict):
            continue
        total_p += u.get("prompt_tokens", 0) or 0
        total_t += u.get("total_tokens", 0) or 0
    return {"prompt_tokens": total_p, "total_tokens": total_t}


# Embedding model pricing (USD per 1M tokens). Only used when deployment matches.
KNOWN_EMBED_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
}


def get_embedding_cost_append(grand_total_embed_usage: dict[str, int] | None) -> str:
    """
    If we have pipeline embedding usage and our embed deployment has known pricing,
    return a string to append to the embedding GRAND TOTAL log line (e.g. " | cost ~0.00 USD").
    Otherwise return "".
    """
    if not grand_total_embed_usage or not grand_total_embed_usage.get("total_tokens"):
        return ""
    deployment = getattr(settings, "azure_openai_embed_deployment", "") or ""
    per_1M = KNOWN_EMBED_PRICING.get(deployment) if deployment else None
    if per_1M is None:
        return ""
    total_tokens = grand_total_embed_usage.get("total_tokens", 0)
    cost = (total_tokens / 1_000_000) * per_1M
    return f" | cost ~{cost:.6f} USD"


def estimate_and_log_embedding_cost(grand_total_embed_usage: dict[str, int] | None) -> None:
    """
    If we have no pricing for the embed deployment, log that so the user knows why
    the GRAND TOTAL line has no USD. When we have pricing, cost is on the grand total line.
    """
    if not grand_total_embed_usage or not grand_total_embed_usage.get("total_tokens"):
        return
    deployment = getattr(settings, "azure_openai_embed_deployment", "") or ""
    if KNOWN_EMBED_PRICING.get(deployment) is not None:
        return  # Cost is on the grand total line
    logger.info(
        "Embedding cost estimate: no pricing for deployment=%r (known: %s)",
        deployment,
        list(KNOWN_EMBED_PRICING.keys()),
    )


_DSPY_CONFIGURED = False
_LM: Optional[Any] = None
_DSPY_CALLERS: List[str] = []


def configure_dspy() -> None:
    global _DSPY_CONFIGURED, _LM

    if _DSPY_CONFIGURED:
        return

    logger.info(
        "Configuring DSPy with Azure OpenAI chat deployment=%s",
        settings.azure_openai_chat_deployment,
    )

    lm = dspy.LM(
        f"azure/{settings.azure_openai_chat_deployment}",
        api_key=settings.azure_openai_api_key,
        api_base=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_chat_api_version,
        model_type="chat",
        temperature=None,
        max_tokens=64000,
    )
    _LM = lm

    dspy.configure(lm=lm)
    dspy.configure(track_usage=True)
    _DSPY_CONFIGURED = True


def log_last_lm_call(caller: str = "dspy") -> None:
    """Track DSPy call origin for later history inspection."""
    _DSPY_CALLERS.append(caller)


def inspect_dspy_history(n: int = 50) -> None:
    """
    Print DSPy history and a lightweight caller map.
    Call this after running the orchestrator to inspect planner/selector/rag/synthesis calls.
    """
    global _LM
    if _LM is None:
        logger.warning("DSPy history unavailable: LM is not configured yet.")
        return

    history = getattr(_LM, "history", None) or []
    total = len(history)
    if total == 0:
        logger.warning("DSPy history is empty.")
        return

    n = max(1, min(n, total))
    logger.info("Inspecting DSPy history: last %d of %d calls", n, total)

    if _DSPY_CALLERS:
        tail = _DSPY_CALLERS[-n:]
        logger.info("Call origins for last %d DSPy calls:", len(tail))
        start_idx = total - len(tail) + 1
        for i, caller in enumerate(tail, start=start_idx):
            logger.info("  #%d -> %s", i, caller)

    dspy.inspect_history(n=n)