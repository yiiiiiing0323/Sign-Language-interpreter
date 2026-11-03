"""REST client for structured JSON requests to the CGU LLM Gateway."""

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests


# MercuryAPI documents both paths as aliases.  The mercuryapi path is the
# current public entry point; keep an environment override for older setups.
DEFAULT_CGU_API_BASE_URL = "https://air.cgu.edu.tw/mercuryapi/v1"
DEFAULT_CGU_MODELS = ("gpt-6-luna",)
CGU_REQUEST_TIMEOUT = 15.0


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
    return os.getenv("CGU_API_KEY", os.getenv("MERCURY_API_KEY", "")).strip()


def load_cgu_base_url() -> str:
    """Return the OpenAI-compatible MercuryAPI base URL."""
    return os.getenv("CGU_API_BASE_URL", DEFAULT_CGU_API_BASE_URL).strip().rstrip("/")


def load_cgu_models() -> List[str]:
    """Return CGU models in priority order, defaulting to gpt-6-luna."""
    configured = os.getenv("CGU_MODELS", "")
    if not configured.strip():
        return list(DEFAULT_CGU_MODELS)
    models = [part.strip() for part in configured.replace(";", ",").split(",")]
    return list(dict.fromkeys(model for model in models if model)) or list(DEFAULT_CGU_MODELS)


def _extract_output_text(response_data: Dict[str, Any]) -> str:
    """Extract text from Responses or Chat Completions JSON."""
    direct_text = response_data.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    choices = response_data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                if parts:
                    return "".join(parts).strip()

    text_parts = []
    output_items = response_data.get("output", [])
    if isinstance(output_items, dict):
        output_items = [output_items]
    for output_item in output_items:
        if not isinstance(output_item, dict):
            continue
        if output_item.get("type") in {"output_text", "text", "text_delta"}:
            text = output_item.get("text", "")
            if isinstance(text, dict):
                text = text.get("value", "")
            if text:
                text_parts.append(str(text))
            continue
        if output_item.get("type") != "message":
            continue
        content_items = output_item.get("content", [])
        if isinstance(content_items, dict):
            content_items = [content_items]
        if isinstance(content_items, str):
            text_parts.append(content_items)
            continue
        for content_item in content_items:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "refusal":
                raise CGUGatewayError("CGU Gateway 拒絕處理這次翻譯要求")
            if content_item.get("type") in {"output_text", "text", "text_delta"}:
                text = content_item.get("text", "")
                if isinstance(text, dict):
                    text = text.get("value", "")
                if text:
                    text_parts.append(str(text))

    if not text_parts:
        raise CGUGatewayError("CGU Gateway 回應中沒有可用的文字輸出")
    return "".join(text_parts).strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Parse plain JSON, fenced JSON, or JSON surrounded by short prose."""
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise CGUGatewayError("CGU Gateway JSON 回應不是物件")
    return parsed


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
    """Call MercuryAPI's OpenAI-compatible endpoint and parse a JSON object."""
    if not api_key:
        raise CGUGatewayError("找不到 CGU_API_KEY")

    client = session or requests
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    prompt = (
        f"{instructions.strip()}\n\n"
        f"只回傳一個符合此 JSON Schema 的 JSON 物件，不要 Markdown、解釋或其他文字。"
        f" schema_name={schema_name}; schema={schema_text}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text},
        ],
        "max_tokens": max_output_tokens,
    }
    try:
        response = client.post(
            f"{load_cgu_base_url()}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise CGUGatewayError(f"CGU Gateway 網路請求失敗：{type(exc).__name__}") from exc

    if not 200 <= response.status_code < 300:
        detail = _safe_error_detail(response, api_key)
        suffix = f"：{detail}" if detail else ""
        raise CGUGatewayError(f"CGU Gateway 回傳 HTTP {response.status_code}{suffix}")

    try:
        response_data = response.json()
        if not isinstance(response_data, dict):
            raise CGUGatewayError("CGU Gateway 回應不是 JSON 物件")
        if response_data.get("status") == "incomplete":
            raise CGUGatewayError("CGU Gateway 回應未完成或輸出遭截斷")
        parsed = _parse_json_object(_extract_output_text(response_data))
    except CGUGatewayError:
        raise
    except (ValueError, TypeError, KeyError) as exc:
        raise CGUGatewayError("CGU Gateway 回應不是有效的 JSON 物件") from exc

    return parsed
