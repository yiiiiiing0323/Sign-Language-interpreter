"""Provider ordering for real-time translation requests."""

import os
from typing import List


# The project's Gemma is hosted by CGU/MercuryAPI and does not use Google
# quota. Try it first, then the CGU backup model, and finally Google Gemini.
# An explicit LLM_PROVIDER_ORDER in .env still overrides this default.
DEFAULT_PROVIDER_ORDER = ("gemma", "cgu", "gemini")


def load_provider_order() -> List[str]:
    """Load a comma-separated provider order, ignoring unsupported names."""
    configured = os.getenv("LLM_PROVIDER_ORDER", "")
    values = configured.split(",") if configured.strip() else DEFAULT_PROVIDER_ORDER
    order = list(dict.fromkeys(
        value.strip().lower()
        for value in values
        if value.strip().lower() in {"gemma", "cgu", "gemini"}
    ))
    return order or list(DEFAULT_PROVIDER_ORDER)
