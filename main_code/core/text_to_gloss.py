"""Text-to-Gloss conversion for the second-stage input path.

This module is intentionally independent from the camera A/B pipeline.  It
uses the project's Excel ``中文`` column as a first-pass Gloss allow-list and
asks Gemini for a structured conversion from ordinary Traditional Chinese to
the project's Gloss order.

The model is not allowed to invent an animation vocabulary.  The returned
Gloss sequence is validated locally before it is exposed to the caller.
"""

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd
try:
    import requests
except ImportError:  # pragma: no cover - production requirements include requests
    requests = None
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - requirements.txt includes python-dotenv
    load_dotenv = None

from core.word_normalization import word_alternatives


logger = logging.getLogger(__name__)

if load_dotenv is not None:
    # Keep standalone use of this module consistent with main.py.  Existing
    # environment variables still win by python-dotenv's default behavior.
    load_dotenv()


DEFAULT_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
)

# These are semantic markers produced by the text converter.  They describe
# grammar/non-manual information and therefore do not have to be rows in the
# camera-rule Excel sheet.
DEFAULT_SYSTEM_MARKERS = {
    "Q_YN",
    "Q_WH",
    "ASPECT_COMPLETE",
    "ASPECT_EXPERIENCE",
    "ASPECT_PROGRESSIVE",
    "ASPECT_CONTINUOUS",
    "NEG_NOT",
    "NEG_NONE",
    "NEG_DONT",
    "NEG_CANNOT",
    "完成",
    "進行",
    "持續",
    "過",
    "不",
    "沒有",
    "不要",
    "不能",
}


class TextToGlossError(RuntimeError):
    """Raised when no model/key attempt can produce a valid response."""


class TextToGlossConverter:
    """Convert ordinary Traditional Chinese into a validated Gloss sequence.

    The converter deliberately keeps the API call and local validation in one
    small module so that ``main.py`` can later connect a text box or a speech
    recognizer without changing the existing A/B camera pipeline.
    """

    def __init__(
        self,
        excel_path: str = "database.xlsx",
        sheet_name: str = "工作表3",
        chinese_column: str = "中文",
        api_keys: Optional[Sequence[str]] = None,
        models: Optional[Sequence[str]] = None,
        timeout: float = 15.0,
        use_real_api: bool = True,
        session: Optional[Any] = None,
    ):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.chinese_column = chinese_column
        self.timeout = timeout
        self.use_real_api = use_real_api
        self.models = tuple(models or self.load_models())
        self.session = session or (requests.Session() if requests is not None else None)
        self.system_markers = set(DEFAULT_SYSTEM_MARKERS)
        self.gloss_vocabulary = self.load_gloss_vocabulary(
            excel_path,
            sheet_name=sheet_name,
            chinese_column=chinese_column,
        )
        self.allowed_glosses = self.gloss_vocabulary | self.system_markers
        self.api_keys = self._dedupe(api_keys if api_keys is not None else self.load_api_keys())

    @staticmethod
    def _dedupe(values: Iterable[str]) -> List[str]:
        result = []
        seen = set()
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                result.append(text)
                seen.add(text)
        return result

    @classmethod
    def load_api_keys(cls) -> List[str]:
        """Load keys in deterministic rotation order without logging values.

        Supported .env names:
        - GEMINI_API_KEY (the existing project setting)
        - GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
        - GEMINI_API_KEYS=key_a,key_b,key_c
        """
        values = []
        grouped = os.getenv("GEMINI_API_KEYS", "")
        if grouped:
            values.extend(part.strip() for part in grouped.replace(";", ",").split(","))

        base = os.getenv("GEMINI_API_KEY", "").strip()
        if base:
            values.append(base)

        indexed = []
        for name, value in os.environ.items():
            if name.startswith("GEMINI_API_KEY_") and name[len("GEMINI_API_KEY_"):].isdigit():
                indexed.append((int(name.rsplit("_", 1)[1]), value))
        values.extend(value for _, value in sorted(indexed) if value)
        return cls._dedupe(values)

    @classmethod
    def load_models(cls) -> List[str]:
        """Load an optional comma-separated model fallback list from .env."""
        configured = os.getenv("GEMINI_MODELS", "")
        if not configured.strip():
            return list(DEFAULT_MODELS)
        models = [part.strip() for part in configured.replace(";", ",").split(",")]
        return cls._dedupe(models) or list(DEFAULT_MODELS)

    @staticmethod
    def _find_column(columns: Iterable[Any], preferred: str) -> Optional[Any]:
        columns = list(columns)
        if preferred in columns:
            return preferred
        preferred_text = str(preferred).strip()
        for column in columns:
            if str(column).strip() == preferred_text:
                return column
        return None

    @classmethod
    def load_gloss_vocabulary(
        cls,
        excel_path: str,
        sheet_name: str = "工作表3",
        chinese_column: str = "中文",
    ) -> Set[str]:
        """Read standard words and alternatives from the Excel 中文 column.

        Values such as ``我_A`` and ``爸爸/父親`` are normalized into usable
        alternatives. Blank rows, IDs, and rule expressions are ignored.
        """
        try:
            frame = pd.read_excel(excel_path, sheet_name=sheet_name)
        except Exception as exc:
            raise TextToGlossError(f"無法讀取 Gloss 詞彙來源：{excel_path}: {exc}") from exc

        column = cls._find_column(frame.columns, chinese_column)
        if column is None:
            raise TextToGlossError(
                f"Excel 工作表 {sheet_name!r} 找不到詞彙欄位 {chinese_column!r}；"
                f"目前欄位為 {list(frame.columns)!r}"
            )

        vocabulary: Set[str] = set()
        for value in frame[column].tolist():
            if value is None or pd.isna(value):
                continue
            text = str(value).strip()
            if not text or text.lower() == "nan":
                continue
            for alternative in word_alternatives(text, strip_suffix=True):
                if alternative:
                    vocabulary.add(alternative)
        return vocabulary

    @property
    def supported_glosses(self) -> List[str]:
        return sorted(self.gloss_vocabulary)

    def build_system_prompt(self) -> str:
        vocabulary = "、".join(self.supported_glosses)
        return f"""你是本專題的台灣手語 Gloss 語序轉換器。

請把輸入的繁體中文句子轉換成第一版專題規格使用的手語 Gloss 順序。
這個功能只負責「一般中文 → Gloss」，不負責 Blender 動畫播放。

【轉換規則】
1. 先分析時間、情境、地點、主題、主詞、受詞、動詞、體貌、否定與疑問。
2. 第一版預設順序為：[時間/情境] [地點/主題] [主詞] [受詞] [動詞] [體貌/否定] [疑問標記]。
3. 時間、地點或情境通常前置，但不存在的角色不可自行補入。
4. 依主題—評論概念轉換，不要只做逐字排序。
5. 不、沒有、不要、不能、已經、過、正在、完成等語意不可直接刪除。
6. 是非問句使用 Q_YN；疑問詞問句使用 Q_WH。
7. 只能使用下方的專案 Gloss 詞彙或系統保留標記。
8. 不可自行創造詞彙、同義詞、解釋文字或不存在的 Gloss。
9. 若輸入詞無法對應，放入 unsupported_words，不要默默改成其他詞。
10. 保留 gloss_sequence 的順序；sentence 只產生自然中文顯示句，不可用中文句子的順序覆蓋 Gloss 順序。

【專案目前可用 Gloss 詞彙】
{vocabulary}

【系統保留標記】
Q_YN、Q_WH、ASPECT_COMPLETE、ASPECT_EXPERIENCE、ASPECT_PROGRESSIVE、
ASPECT_CONTINUOUS、NEG_NOT、NEG_NONE、NEG_DONT、NEG_CANNOT、完成、進行、持續、過、
不、沒有、不要、不能

【輸出格式】
只能回傳 JSON，不要使用 Markdown code block：
{{
  "input_sentence": "原始輸入句子",
  "gloss_sequence": ["Gloss1", "Gloss2"],
  "gloss_text": "Gloss1 Gloss2",
  "sentence": "自然中文句子",
  "unsupported_words": []
}}
"""

    def build_payload(self, sentence: str) -> Dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": self.build_system_prompt()}]},
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "請依照規則轉換以下一般中文句子，不要回傳額外說明：\n"
                                f"{sentence.strip()}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 300,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "input_sentence": {"type": "STRING"},
                        "gloss_sequence": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "gloss_text": {"type": "STRING"},
                        "sentence": {"type": "STRING"},
                        "unsupported_words": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                    "required": [
                        "input_sentence",
                        "gloss_sequence",
                        "gloss_text",
                        "sentence",
                        "unsupported_words",
                    ],
                },
            },
        }

    def validate_result(self, payload: Dict[str, Any], input_sentence: str) -> Dict[str, Any]:
        """Validate and normalize an LLM response against the Excel allow-list."""
        if not isinstance(payload, dict):
            raise TextToGlossError("Text-to-Gloss 回應不是 JSON 物件")

        raw_sequence = payload.get("gloss_sequence", [])
        if not isinstance(raw_sequence, list):
            raise TextToGlossError("gloss_sequence 必須是陣列")

        accepted: List[str] = []
        unsupported: List[str] = []
        for raw_token in raw_sequence:
            token = str(raw_token).strip()
            if not token:
                continue
            if token in self.allowed_glosses:
                accepted.append(token)
            else:
                unsupported.append(token)

        given_unsupported = payload.get("unsupported_words", [])
        if isinstance(given_unsupported, list):
            unsupported.extend(str(word).strip() for word in given_unsupported if str(word).strip())

        # Preserve order while removing duplicate error reports.
        unsupported = list(dict.fromkeys(unsupported))
        if not accepted and not unsupported:
            raise TextToGlossError("LLM 沒有產生任何 Gloss 或未支援詞")

        sentence = str(payload.get("sentence", "")).strip()
        if not sentence:
            sentence = input_sentence.strip()

        return {
            "input_sentence": str(payload.get("input_sentence") or input_sentence).strip(),
            "gloss_sequence": accepted,
            "gloss_text": " ".join(accepted),
            "sentence": sentence,
            "unsupported_words": unsupported,
            "validated": True,
        }

    def convert(self, sentence: str) -> Dict[str, Any]:
        """Call Gemini with model/key fallback and return validated JSON."""
        sentence = str(sentence or "").strip()
        if not sentence:
            raise ValueError("輸入句子不可為空")
        if not self.use_real_api:
            raise TextToGlossError("目前設定為離線模式，未執行 Gemini API")
        if not self.api_keys:
            raise TextToGlossError("找不到 GEMINI_API_KEY 或 GEMINI_API_KEYS")
        if self.session is None:
            raise TextToGlossError("目前 Python 環境沒有安裝 requests，請先安裝 requirements.txt")

        errors = []
        payload = self.build_payload(sentence)
        for key_index, api_key in enumerate(self.api_keys, start=1):
            for model in self.models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                try:
                    response = self.session.post(
                        url,
                        params={"key": api_key},
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=self.timeout,
                    )
                    if response.status_code != 200:
                        errors.append(f"key#{key_index}/{model}: HTTP {response.status_code}")
                        if response.status_code not in {400, 401, 403, 404, 429, 500, 503}:
                            continue
                        continue

                    data = response.json()
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    parsed = json.loads(raw_text)
                    result = self.validate_result(parsed, sentence)
                    logger.info(
                        "text_to_gloss_success key_index=%s model=%s gloss_count=%s unsupported=%s",
                        key_index,
                        model,
                        len(result["gloss_sequence"]),
                        len(result["unsupported_words"]),
                    )
                    return result
                except Exception as exc:
                    errors.append(f"key#{key_index}/{model}: {type(exc).__name__}")

        raise TextToGlossError("所有 Text-to-Gloss API 嘗試皆失敗：" + "; ".join(errors))


__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_SYSTEM_MARKERS",
    "TextToGlossConverter",
    "TextToGlossError",
]
