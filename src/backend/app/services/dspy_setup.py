from __future__ import annotations

import logging
from typing import Any, Optional, List

import dspy

from app.settings import settings

logger = logging.getLogger(__name__)

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
        max_tokens=16000,
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