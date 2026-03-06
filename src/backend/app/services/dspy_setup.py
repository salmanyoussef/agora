from __future__ import annotations

import logging
import dspy

from app.settings import settings

logger = logging.getLogger(__name__)

_DSPY_CONFIGURED = False


def configure_dspy() -> None:
    global _DSPY_CONFIGURED

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
        temperature=0.4,
        max_tokens=4000,
    )

    dspy.configure(lm=lm)
    _DSPY_CONFIGURED = True