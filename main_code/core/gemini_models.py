"""Shared Gemini model fallback order and model-specific request settings."""

import os
from typing import Any, Dict, List


DEFAULT_GEMMA_MODELS = ("gemma-4-26b-a4b-it",)
DEFAULT_MODELS = ("gemini-3.5-flash-lite",)

# Google returned 404 for these IDs on the newly created key used by this
# project, explicitly stating they are unavailable to new users.
UNAVAILABLE_MODELS = frozenset({
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
})

# Google AI Studio can take several seconds to wake a model.  Keep the total
# budget bounded, but do not treat the old 4-second read timeout as a failure.
TOTAL_REQUEST_TIMEOUT = 32.0
GEMMA_REQUEST_TIMEOUT = 12.0
MODEL_REQUEST_TIMEOUT = 12.0
GEMINI_MAX_ATTEMPTS = 2
GEMINI_RETRY_BACKOFF_SECONDS = 0.6
GEMINI_RETRY_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def load_gemma_models() -> List[str]:
    """Load the first-choice school-hosted Gemma model(s)."""
    configured = os.getenv("GEMMA_MODELS", "")
    if not configured.strip():
        return list(DEFAULT_GEMMA_MODELS)
    models = [part.strip() for part in configured.replace(";", ",").split(",")]
    return list(dict.fromkeys(model for model in models if model)) or list(DEFAULT_GEMMA_MODELS)


def load_models() -> List[str]:
    """Load GEMINI_MODELS override or use the shared default fallback order."""
    configured = os.getenv("GEMINI_MODELS", "")
    if not configured.strip():
        return list(DEFAULT_MODELS)
    models = [part.strip() for part in configured.replace(";", ",").split(",")]
    deduped = list(dict.fromkeys(
        model for model in models if model and model not in UNAVAILABLE_MODELS
    ))
    return deduped or list(DEFAULT_MODELS)


def configure_model_thinking(payload: Dict[str, Any], model: str) -> None:
    """Apply a supported low-latency thinking setting for a model family."""
    generation_config = payload.setdefault("generationConfig", {})
    generation_config.pop("thinkingConfig", None)

    if model == "gemini-3.5-flash-lite":
        generation_config["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}
    elif model.startswith("gemini-3."):
        generation_config["thinkingConfig"] = {"thinkingLevel": "LOW"}
    elif model.startswith("gemini-2.5-"):
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}
