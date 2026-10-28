"""REST client for structured JSON requests to the CGU LLM Gateway."""

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests


CGU_API_BASE_URL = "https://air.cgu.edu.tw/cgullmapi/v1"
DEFAULT_CGU_MODELS = ("gpt-6-luna",)
CGU_REQUEST_TIMEOUT = 8.0


class CGUGatewayError(RuntimeError):
    """Raised when the CGU Gateway cannot return the requested response."""


def _safe_error_detail(response: Any, api_key: str) -> str:
    """Return a short API error summary without exposing credentials."""
    try:
        error = response.json().get("error", {})
    except (ValueError, TypeError, AttributeError):
        return ""
    if not isinstance(error, dict):
        return ""

    details = []
    for field in ("code", "type", "message"):
        value = error.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        value = " ".join(value.split())
        if api_key:
            value = value.replace(api_key, "[redacted]")
        value = re.sub(r"\b(?:sk|rk)-[A-Za-z0-9_-]{8,}\b", "[redacted]", value)
        value = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [redacted]", value)
        details.append(f"{field}={value}")
    return "; ".join(details)[:240]


def load_cgu_api_key() -> str:
    return os.getenv("CGU_API_KEY", "").strip()


def load_cgu_models() -> List[str]:
    """Return CGU models in priority order, defaulting to gpt-6-luna."""
    configured = os.getenv("CGU_MODELS", "")
    if not configured.strip():
        return list(DEFAULT_CGU_MODELS)
    models = [part.strip() for part in configured.replace(";", ",").split(",")]
    return list(dict.fromkeys(model for model in models if model)) or list(DEFAULT_CGU_MODELS)


def _extract_output_text(response_data: Dict[str, Any]) -> str:
    """Extract message text from a Responses-compatible gateway result."""
    direct_text = response_data.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    text_parts = []
    for output_item in response_data.get("output", []):
        if output_item.get("type") != "message":
            continue
        for content_item in output_item.get("content", []):
            if content_item.get("type") == "refusal":
                raise CGUGatewayError("CGU Gateway 拒絕處理這次翻譯要求")
            if content_item.get("type") == "output_text":
                text = content_item.get("text", "")
                if text:
                    text_parts.append(text)

    if not text_parts:
        raise CGUGatewayError("CGU Gateway 回應中沒有可用的文字輸出")
    return "".join(text_parts).strip()


def generate_json(
    *,
    api_key: str,
    model: str,
    instructions: str,
    input_text: str,
    schema_name: str,
    schema: Dict[str, Any],
    timeout: float,
    session: Optional[Any] = None,
    max_output_tokens: int = 300,
) -> Dict[str, Any]:
    """Call the CGU Responses-compatible endpoint and parse a JSON object."""
    if not api_key:
        raise CGUGatewayError("找不到 CGU_API_KEY")

    client = session or requests
    payload = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "reasoning": {"effort": "none"},
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }
    try:
        response = client.post(
            f"{CGU_API_BASE_URL}/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise CGUGatewayError(f"CGU Gateway 網路請求失敗：{type(exc).__name__}") from exc

    if response.status_code != 200:
        detail = _safe_error_detail(response, api_key)
        suffix = f"：{detail}" if detail else ""
        raise CGUGatewayError(f"CGU Gateway 回傳 HTTP {response.status_code}{suffix}")

    try:
        response_data = response.json()
        if response_data.get("status") == "incomplete":
            raise CGUGatewayError("CGU Gateway 回應未完成或輸出遭截斷")
        parsed = json.loads(_extract_output_text(response_data))
    except CGUGatewayError:
        raise
    except (ValueError, TypeError, KeyError) as exc:
        raise CGUGatewayError("CGU Gateway 回應不是有效的 JSON 物件") from exc

    if not isinstance(parsed, dict):
        raise CGUGatewayError("CGU Gateway JSON 回應不是物件")
    return parsed
