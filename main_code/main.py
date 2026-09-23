import sys
sys.dont_write_bytecode = True

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

print("[追蹤雷達] 1. 成功讀取檔案，開始執行...")

import cv2
import numpy as np
import time
import traceback
import pandas as pd
from PIL import Image, ImageDraw, ImageFont 
import threading  
import requests   
import os
import hashlib
from dotenv import load_dotenv

from core.feature_registry import AI_TENSOR_DIM, AI_TENSOR_LAYOUT_VERSION
from core.logging_config import get_logger, setup_logging
from core.word_normalization import normalize_output_word

# ========================================
# ⭐ 新增：PyTorch 與融合模組
# ========================================
import torch
import torch.nn as nn
import json
from collections import deque

print("[追蹤雷達] 2. 成功載入基礎套件...")
import mediapipe as mp

# 載入獨立出來的 A 流與 B 流
from a_stream import AStreamFeatureExtractor
try:
    from b_stream import BStreamGestureMatcher
    print("[追蹤雷達] 4. 成功找到 a_stream 與 b_stream 模組...")
except ImportError:
    print("⚠️ 找不到 b_stream.py，程式可能無法完整運作。")

# ⭐ 新增：匯入融合模組
from fusion import DecisionFusion
from compound_phrase import CompoundPhraseResolver
from core.gloss_history import AppendOnlyGlossHistory
from core.text_to_gloss import TextToGlossConverter, TextToGlossError

logger = get_logger(__name__)
perf_logger = get_logger("performance")

def resource_path(relative_path):
    """
    回傳專案資源檔的實際路徑。

    為什麼需要這個函式：
    1. 直接用 python main.py 執行時，資料檔通常在專案資料夾。
    2. 用 PyInstaller 打包成 exe 後，資料檔會被複製到 exe 的暫存/輸出資料夾。
    3. 如果程式仍然只寫 open("label_map.json")，exe 很容易因工作目錄不同而找不到檔案。

    這個函式會優先找 PyInstaller 的資源資料夾，再找目前工作目錄，
    讓同一份程式碼可以同時支援開發模式與可執行檔模式。
    """
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    packaged_path = os.path.join(base_path, relative_path)
    if os.path.exists(packaged_path):
        return packaged_path
    return os.path.join(os.getcwd(), relative_path)


def open_camera(camera_indices=(0, 1, 2, 3), width=640, height=480):
    """
    嘗試開啟可用攝影機。

    原本程式只嘗試 camera index 0：
        cv2.VideoCapture(0, cv2.CAP_DSHOW)

    但 Windows 筆電常見狀況包括：
    - 內建鏡頭不是 index 0
    - 其他程式正在占用 index 0
    - CAP_DSHOW 在某些攝影機驅動上失敗，但一般 backend 可用
    - exe 模式下工作目錄不同，使用者只看到程式沒開鏡頭，log 又沒有說明

    因此這裡會依序嘗試多個 index 與 backend。
    每次成功/失敗都寫入 system.log，方便判斷問題是攝影機權限、index 錯誤，
    還是 OpenCV backend 不相容。
    """
    backends = [
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("DEFAULT", 0),
    ]

    for index in camera_indices:
        for backend_name, backend in backends:
            logger.info("嘗試開啟攝影機 index=%s backend=%s", index, backend_name)
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                logger.warning("攝影機開啟失敗 index=%s backend=%s", index, backend_name)
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # 讀一幀確認不是「裝置 opened 但實際無畫面」。
            ok, frame = cap.read()
            if ok and frame is not None:
                logger.info("攝影機開啟成功 index=%s backend=%s", index, backend_name)
                return cap, index, backend_name

            cap.release()
            logger.warning("攝影機可開啟但讀不到畫面 index=%s backend=%s", index, backend_name)

    return None, None, None

# ========================================
# ⭐ 新增：LSTM 模型定義
# ========================================
class SignRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# 中文字體繪製器（保持不變）
def put_chinese_text(img, text, position, text_color=(0, 255, 0), font_size=40):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("msjh.ttc", font_size)
        except IOError:
            font = ImageFont.load_default() 
            
        draw.text(position, text, font=font, fill=text_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
        return img

# ==========================================
# 🌟 LLM 翻譯模組（保持你原有的 Gemini 版本）
# ==========================================
class TranslationWorker:
    def __init__(
        self,
        api_key,
        use_real_api=False,
        text_to_gloss_converter=None,
        history_writer=None,
    ):
        self.use_real_api = use_real_api
        self.api_key = api_key
        self.text_to_gloss_converter = text_to_gloss_converter
        self.history_writer = history_writer
        self.word_buffer = []
        # 跟 word_buffer 一一對應，但保留 _A/_B/_N/_S 尾碼，只給複合詞比對用。
        # word_buffer 本身維持乾淨（顯示、送 Gemini 用），不受這個影響。
        self.raw_word_buffer = []
        self.final_sentence = ""
        self.is_translating = False
        self.last_word_time = time.time()
        self.last_added_word = ""  
        self.last_hands_seen_time = time.time()

        self.debounce_time = 2.0     
        self.translate_delay = 2.0
        self.display_duration = 5.0  
        self.translation_done_time = 0 

        # 保留辨識批次的原始/正規化 Gloss，供第二階段比對與失敗備援。
        self.last_camera_gloss_sequence = []
        self.last_camera_raw_gloss_sequence = []
        self.last_stage1_sentence = ""
        self.last_stage2_result = None
        self.playback_gloss_sequence = []
        self.stage2_status = "idle"
        self.stage2_in_progress = False
        self._stage2_lock = threading.Lock()
        self._stage2_job_id = 0

        self.system_instruction = """
        你是一個專業的台灣手語(TSL)翻譯員，負責將一連串辨識出的手語單字（Glosses）重組並潤飾成一句自然、通順的繁體中文日常用語。
        注意：常見的固定複合詞（如親屬稱謂、專有名詞）已在前端比對規則庫合併過，你收到的詞通常已是合併後的結果，不需要再自己組合。

        【處理規則】
        1. 容錯與去重：輸入由即時影像辨識產生，可能包含短時間重複跳動或無意義雜訊字（如「物」）。請自動過濾重複詞與贅字。
        2. 語序調整：手語常將時間副詞放句首、動詞放句尾，請依中文口語習慣調整語序（例：["昨天","我","學校","去"] -> 我昨天去學校。["明天","雨","下"] -> 明天會下雨。）
        3. 問候語與疑問句：轉換成道地口語（例：["早上","平安","好"] -> 早安，你好。["你","吃飯","已經"] -> 你吃過飯了嗎？）
        4. 語氣與否定：正確處理強調詞與否定語序（例：["爸爸","生氣","很"] -> 爸爸很生氣。["我","不是"] -> 不是我。）
        5. 人稱與狀態句：補上必要的「是」等動詞（例：["我們","學生"] -> 我們是學生。）
        6. 混淆詞：遇到方括號/斜線標記的候選詞（如 [A/B]），依上下文只選一個，絕對不保留括號與斜線（例：["你好","[先生/謝謝]"] -> 你好，先生。["[捷運/火車]","搭","我","學校","去"] -> 我搭捷運去學校。）
        7. 邊界條件：語意極不完整時仍輸出最自然的短句，不要拒答，也不要腦補不存在的背景資訊。

        【輸出格式】
        只能回傳 JSON 物件，格式為 {"sentence": "翻譯後的繁體中文句子"}。
        sentence 欄位只放翻譯後的句子與必要標點，絕對不包含解釋、分析、原始 Gloss 或引言。
        """

    def add_word(self, word, raw_word=None):
        if word is None:
            return
        clean_word = str(word).strip()
        if not clean_word:
            return
        if clean_word != self.last_added_word or (time.time() - self.last_word_time > self.debounce_time):
            self.word_buffer.append(clean_word)
            self.raw_word_buffer.append(str(raw_word).strip() if raw_word else clean_word)
            self.last_added_word = clean_word
            self.last_word_time = time.time()
            self.final_sentence = ""
            self._append_history("camera_gloss", {
                "gloss": clean_word,
                "raw_gloss": str(raw_word).strip() if raw_word else clean_word,
            })
            print(f"[翻譯雷達] 目前收集單字: {self.word_buffer}")

    #6/24
    def replace_tail_words(self, consumed_count, merged_word, raw_word=None, now=None):
        if consumed_count > 0:
            self.word_buffer = self.word_buffer[:-consumed_count]
            self.raw_word_buffer = self.raw_word_buffer[:-consumed_count]

        if merged_word is None:
            return
        clean_word = str(merged_word).strip()
        self.word_buffer.append(clean_word)
        self.raw_word_buffer.append(str(raw_word).strip() if raw_word else clean_word)
        self.last_added_word = clean_word
        self.last_word_time = now if now is not None else time.time()
        self.final_sentence = ""

    def _append_history(self, event, data):
        if self.history_writer is None:
            return
        if not self.history_writer.append(event, data):
            logger.warning("無法寫入 gloss history event=%s", event)

    @staticmethod
    def _compare_gloss_sequences(original_sequence, converted_sequence):
        """Compare content without requiring the two stages to share order."""
        original = [normalize_output_word(word) for word in (original_sequence or [])]
        converted = [normalize_output_word(word) for word in (converted_sequence or [])]
        original = [word for word in original if word]
        converted = [word for word in converted if word]

        original_counts = {}
        converted_counts = {}
        for word in original:
            original_counts[word] = original_counts.get(word, 0) + 1
        for word in converted:
            converted_counts[word] = converted_counts.get(word, 0) + 1

        missing = []
        for word, count in original_counts.items():
            missing.extend([word] * max(0, count - converted_counts.get(word, 0)))
        extra = []
        for word, count in converted_counts.items():
            extra.extend([word] * max(0, count - original_counts.get(word, 0)))

        matched = sum(min(count, converted_counts.get(word, 0)) for word, count in original_counts.items())
        overlap = matched / len(original) if original else None
        return {
            "original_count": len(original),
            "converted_count": len(converted),
            "matched_count": matched,
            "overlap": overlap,
            "missing_from_stage2": missing,
            "extra_in_stage2": extra,
        }

    def _run_stage2(self, sentence, source, fallback_gloss_sequence, job_id):
        try:
            result = self.text_to_gloss_converter.convert(sentence)
            result["source"] = source
            result["fallback_gloss_sequence"] = list(fallback_gloss_sequence or [])
            result["comparison"] = self._compare_gloss_sequences(
                fallback_gloss_sequence,
                result.get("gloss_sequence", []),
            )

            # A valid Stage 2 result is preferred. If it fails or contains
            # unknown terms, the preserved camera sequence remains available.
            if result.get("gloss_sequence") and not result.get("unsupported_words"):
                playback_sequence = list(result["gloss_sequence"])
                fallback_used = False
            elif fallback_gloss_sequence:
                playback_sequence = list(fallback_gloss_sequence)
                fallback_used = True
                result["fallback_reason"] = "stage2_empty_or_unsupported"
            else:
                playback_sequence = []
                fallback_used = False

            result["playback_gloss_sequence"] = playback_sequence
            result["fallback_used"] = fallback_used
            # Every completed request is kept in the append-only history,
            # including a result that became stale because a newer request
            # started before this one returned.
            self._append_history("stage2_result", result)
            with self._stage2_lock:
                if job_id != self._stage2_job_id:
                    return
                self.last_stage2_result = result
                self.playback_gloss_sequence = playback_sequence
                self.stage2_status = "completed_fallback" if fallback_used else "completed"
                self.stage2_in_progress = False

            print(f"[Text-to-Gloss] 完成 source={source} gloss={result.get('gloss_text', '')}")
        except Exception as exc:
            fallback = list(fallback_gloss_sequence or [])
            error_record = {
                "source": source,
                "sentence": sentence,
                "error": str(exc),
                "fallback_gloss_sequence": fallback,
            }
            self._append_history("stage2_error", error_record)
            with self._stage2_lock:
                if job_id != self._stage2_job_id:
                    return
                self.last_stage2_result = {
                    **error_record,
                    "gloss_sequence": [],
                    "gloss_text": "",
                    "unsupported_words": [],
                    "playback_gloss_sequence": fallback,
                    "fallback_used": bool(fallback),
                }
                self.playback_gloss_sequence = fallback
                self.stage2_status = "failed_fallback" if fallback else "failed"
                self.stage2_in_progress = False
            logger.exception("第二階段 Text-to-Gloss 失敗 source=%s", source)

    def _start_stage2(self, sentence, source="camera_stage1", fallback_gloss_sequence=None):
        sentence = str(sentence or "").strip()
        fallback_gloss_sequence = list(fallback_gloss_sequence or [])
        if not sentence:
            return None

        with self._stage2_lock:
            self._stage2_job_id += 1
            job_id = self._stage2_job_id
            self.stage2_status = "pending"
            self.stage2_in_progress = True
            self.last_stage2_result = None

        if self.text_to_gloss_converter is None:
            with self._stage2_lock:
                self.stage2_status = "unavailable_fallback" if fallback_gloss_sequence else "unavailable"
                self.stage2_in_progress = False
                self.playback_gloss_sequence = fallback_gloss_sequence
            self._append_history("stage2_unavailable", {
                "source": source,
                "sentence": sentence,
                "fallback_gloss_sequence": fallback_gloss_sequence,
            })
            return job_id

        threading.Thread(
            target=self._run_stage2,
            args=(sentence, source, fallback_gloss_sequence, job_id),
            daemon=True,
        ).start()
        return job_id

    def submit_text_input(self, sentence, source="text"):
        """Public entry point for future text and speech-to-text UI inputs."""
        return self._start_stage2(sentence, source=source, fallback_gloss_sequence=[])

    def get_playback_gloss_sequence(self):
        with self._stage2_lock:
            return list(self.playback_gloss_sequence)

    def _call_llm(self, words_to_translate, original_gloss_sequence=None):
        self.is_translating = True
        stage1_success = False
        original_gloss_sequence = list(original_gloss_sequence or words_to_translate or [])
        try:
            if not self.use_real_api:
                time.sleep(1) 
                self.final_sentence = f"[測試翻譯] {' '.join(words_to_translate)}"
                self.last_camera_gloss_sequence = list(words_to_translate)
                self.last_camera_raw_gloss_sequence = list(original_gloss_sequence)
                self.last_stage1_sentence = self.final_sentence
                self._append_history("stage1_sentence", {
                    "source": "camera_test_mode",
                    "gloss_sequence": list(words_to_translate),
                    "raw_gloss_sequence": list(original_gloss_sequence),
                    "sentence": self.final_sentence,
                })
                self._start_stage2(
                    self.final_sentence,
                    source="camera_stage1_test_mode",
                    fallback_gloss_sequence=list(words_to_translate),
                )
            else:
                print(f"[LLM] 發送雲端請求: {words_to_translate}...")
                clean_api_key = self.api_key.strip()
                
                fallback_models = [
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-flash-latest"
                ]
                
                headers = {'Content-Type': 'application/json'}
                payload = {
                    "systemInstruction": {
                        "parts": [{"text": self.system_instruction}]
                    },
                    "contents": [
                        {"parts": [{"text": f"請翻譯以下手語單字：{words_to_translate}"}]}
                    ],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 200,
                        "thinkingConfig": {
                            "thinkingBudget": 0
                        },
                        "responseMimeType": "application/json",
                        "responseSchema": {
                            "type": "OBJECT",
                            "properties": {
                                "sentence": {"type": "STRING"}
                            },
                            "required": ["sentence"]
                        }
                    }
                }

                max_retries = len(fallback_models)
                last_status_code = None
                for attempt in range(max_retries):
                    current_model = fallback_models[attempt]
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={clean_api_key}"

                    try:
                        response = requests.post(url, headers=headers, json=payload, timeout=8)
                        last_status_code = response.status_code

                        if response.status_code in [400, 404, 429, 503]:
                            if response.status_code == 400:
                                print(f"⚠️ [{current_model}] 請求格式錯誤 (400)，此模型可能不支援目前的參數，切換... 詳細訊息: {response.text}")
                            elif response.status_code == 404:
                                print(f"⚠️ [{current_model}] 模型不存在 (404)，切換...")
                            elif response.status_code == 429:
                                print(f"⚠️ [{current_model}] 免費額度為 0 (429)，切換...")
                            elif response.status_code == 503:
                                print(f"⚠️ [{current_model}] 伺服器塞車 (503)，切換...")

                            time.sleep(0.5)
                            continue

                        if response.status_code != 200:
                            print(f"\n🚨 [{current_model}] 狀態碼: {response.status_code}, 詳細訊息: {response.text}")
                            self.final_sentence = f"API 錯誤: {response.status_code}"
                            return

                        data = response.json()
                        raw_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                        try:
                            parsed = json.loads(raw_text)
                            self.final_sentence = str(parsed.get("sentence", "")).strip()
                            stage1_success = bool(self.final_sentence)
                        except (json.JSONDecodeError, AttributeError):
                            # 保底：萬一模型沒有依 schema 回傳 JSON，退回舊有的純文字處理
                            self.final_sentence = raw_text
                            stage1_success = bool(self.final_sentence)
                        print(f"[LLM] 翻譯成功 🎉 (使用的模型: {current_model}): {self.final_sentence}")
                        self.last_camera_gloss_sequence = list(words_to_translate)
                        self.last_camera_raw_gloss_sequence = list(original_gloss_sequence)
                        self.last_stage1_sentence = self.final_sentence
                        self._append_history("stage1_sentence", {
                            "source": "camera",
                            "gloss_sequence": list(words_to_translate),
                            "raw_gloss_sequence": list(original_gloss_sequence),
                            "sentence": self.final_sentence,
                        })
                        if stage1_success:
                            self._start_stage2(
                                self.final_sentence,
                                source="camera_stage1",
                                fallback_gloss_sequence=list(words_to_translate),
                            )
                        return

                    except requests.exceptions.Timeout:
                        print(f"⚠️ [{current_model}] 請求超時 (Timeout)，切換備用路線...")
                        continue
                    except requests.exceptions.ConnectionError:
                        print("\n🚨 [LLM 網路錯誤] 無法連線！請檢查學校防火牆或切換手機熱點。")
                        self.final_sentence = "網路連線失敗"
                        return

                print(f"\n🚨 [LLM 錯誤] 所有的 2.5 備用模型都失敗了（最後狀態碼: {last_status_code}）。")
                self.final_sentence = "伺服器忙碌中，請稍後再試"
                
        except Exception as e:
            print(f"\n🚨 [LLM 未知錯誤] 翻譯失敗: {e}")
            self.final_sentence = "發生未知錯誤(看終端機)"
        finally:
            self.is_translating = False
            self.last_added_word = "" 
            self.translation_done_time = time.time() 

    def check_and_translate(self, hands_present=False):
        if self.final_sentence and not self.is_translating:
            if time.time() - self.translation_done_time > self.display_duration:
                self.final_sentence = ""

        if hands_present:
            self.last_hands_seen_time = time.time()

        if len(self.word_buffer) > 0 and not self.is_translating:
            no_hands_trigger = (not hands_present) and (time.time() - self.last_hands_seen_time > self.translate_delay)
            timeout_trigger = (time.time() - self.last_word_time > 5.0)

            if no_hands_trigger or timeout_trigger:
                words_to_translate = list(self.word_buffer)
                original_gloss_sequence = list(self.raw_word_buffer or self.word_buffer)
                self.last_camera_gloss_sequence = list(words_to_translate)
                self.last_camera_raw_gloss_sequence = list(original_gloss_sequence)
                self._append_history("camera_gloss_batch", {
                    "gloss_sequence": words_to_translate,
                    "raw_gloss_sequence": original_gloss_sequence,
                })
                self.word_buffer.clear()
                self.raw_word_buffer.clear()
                threading.Thread(
                    target=self._call_llm,
                    args=(words_to_translate, original_gloss_sequence),
                    daemon=True,
                ).start()

# ==========================================
# ⭐ 主程式 (大幅修改)
# ==========================================
def main():
    # 只保留主控台輸出，不會在專案內建立 log 檔。
    app_base_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    setup_logging(app_base_dir)
    print("[追蹤雷達] 5. 進入 main() 主程式區塊...")
    logger.info("主程式啟動")

    # .env 不建議打包進 exe，請放在 exe 同一層或專案資料夾中。
    load_dotenv(resource_path(".env"))
    api_key = os.getenv("GEMINI_API_KEY")
    
    # ========================================
    # ⭐ 新增：載入 LSTM 模型
    # ========================================
    try:
        # label_map.json 可能有 43 個「手型標籤」。
        # 若模型輸出只有 41 類，下面會自動把 _A / _B 後綴合併成同一個最終詞。
        label_map_path = resource_path("label_map.json")
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw_idx2label = {int(k): v for k, v in json.load(f).items()}

        with open(resource_path("model_contract.json"), "r", encoding="utf-8") as f:
            model_contract = json.load(f)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        TOTAL_DIM = AI_TENSOR_DIM
        SEQ_LEN = 30

        state_dict = torch.load(resource_path("sign_lstm.pth"), map_location=device)
        model_class_count = state_dict["fc.weight"].shape[0]

        with open(label_map_path, "rb") as f:
            label_map_hash = hashlib.sha256(f.read()).hexdigest()
        contract_checks = {
            "input_dim": TOTAL_DIM,
            "sequence_length": SEQ_LEN,
            "class_count": model_class_count,
            "tensor_layout": AI_TENSOR_LAYOUT_VERSION,
            "label_map_sha256": label_map_hash,
        }
        mismatches = {
            key: (model_contract.get(key), actual)
            for key, actual in contract_checks.items()
            if model_contract.get(key) != actual
        }
        if mismatches:
            raise ValueError(f"模型契約不一致: {mismatches}")

        clean_labels = []
        for i in sorted(raw_idx2label):
            clean_word = raw_idx2label[i].split('_')[0]
            if clean_word not in clean_labels:
                clean_labels.append(clean_word)

        if model_class_count == len(clean_labels):
            idx2label = {i: word for i, word in enumerate(clean_labels)}
            if len(raw_idx2label) != len(clean_labels):
                print(f"ℹ️ label_map 有 {len(raw_idx2label)} 個手型標籤，合併 _A/_B 後為 {len(clean_labels)} 個 LSTM 輸出詞")
                logger.info(
                    "label_map 手型標籤已合併 raw=%s clean=%s",
                    len(raw_idx2label),
                    len(clean_labels),
                )
        elif model_class_count == len(raw_idx2label):
            idx2label = raw_idx2label
        else:
            print(f"⚠️ label_map 有 {len(raw_idx2label)} 個手型標籤、合併後 {len(clean_labels)} 個詞，但 LSTM 模型是 {model_class_count} 類；AI 流將使用可對應的前 {model_class_count} 類")
            logger.warning(
                "label_map 類別數與模型不一致 raw=%s clean=%s model=%s",
                len(raw_idx2label),
                len(clean_labels),
                model_class_count,
            )
            idx2label = {
                i: raw_idx2label.get(i, f"class_{i}").split('_')[0]
                for i in range(model_class_count)
            }
        
        model = SignRNN(TOTAL_DIM, 128, 2, model_class_count).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        sequence_buffer = deque(maxlen=SEQ_LEN)
        print(f"✅ LSTM 模型已載入 (裝置: {device})")
        logger.info("LSTM 模型已載入 device=%s input_dim=%s seq_len=%s", device, TOTAL_DIM, SEQ_LEN)
        lstm_enabled = True
    except Exception as e:
        print(f"⚠️ LSTM 模型載入失敗: {e}")
        print("⚠️ 系統將只使用 B 流邏輯判斷")
        logger.exception("LSTM 模型載入失敗，系統改用 B 流邏輯判斷")
        lstm_enabled = False
    
    
    # ========================================
    # 初始化模組
    # ========================================
    a_stream_extractor = AStreamFeatureExtractor()

    # ========================================
    # 初始化模組
    # ========================================
    print("[追蹤雷達] 5-A. 開始初始化 BStreamGestureMatcher...")
    b_stream_matcher = BStreamGestureMatcher(excel_path=resource_path("database.xlsx"))
    print("[追蹤雷達] 5-B. BStreamGestureMatcher 初始化成功！")
    
    # ⭐ 新增：決策融合器
    print("[追蹤雷達] 5-C. 開始初始化 DecisionFusion...")
    decision_fusion = DecisionFusion()
    
    print("[追蹤雷達] 5-D. 開始初始化 CompoundPhraseResolver...")
    compound_resolver = CompoundPhraseResolver(
        resource_path("database.xlsx"),
        sheet_name="工作表3",
        source_col="Unnamed: 0",
        output_col="中文",
    )
    print("[追蹤雷達] 5-E. CompoundPhraseResolver 初始化成功！")

    try:
        print("[追蹤雷達] 5-F. 嘗試執行 Excel X光機 讀取...")
        df = pd.read_excel(resource_path("database.xlsx"), sheet_name="工作表3")
        print("\n" + "="*50)
        print("🔍 [Excel X光機] 檢查 B 流讀取到的規則：")
        for idx, row in df.iterrows():
            word = str(row.get('中文', '')).strip()
            if word in ['你', '我', '平安', '謝謝']:
                condition = row.get('MediaPipe 關鍵特徵', '【找不到這個欄位】')
                print(f"👉 單字: {word}")
                print(f"   真實條件: [{condition}]")
        print("="*50 + "\n")
        print("[追蹤雷達] 5-G. Excel X光機 執行完畢！")
    except Exception as e:
        print(f"⚠️ 無法啟動 Excel X光機: {e}")

    print("[追蹤雷達] 6. 準備初始化 MediaPipe 視覺引擎...")
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh  # ⭐ 新增
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hands = mp_hands.Hands(max_num_hands=2, 
                           min_detection_confidence=0.5,   
                           min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # ⭐ 新增
    logger.info("MediaPipe 視覺引擎初始化完成")
    
    # 保留原本的測試模式預設值；若要啟用第一階段與第二階段的真實 Gemini 呼叫，
    # 請在 .env 設定 GEMINI_USE_REAL_API=1。未設定時不會改變原本的測試流程。
    gemini_use_real_api = os.getenv("GEMINI_USE_REAL_API", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }

    history_writer = None
    try:
        history_path = os.path.join(app_base_dir, "gloss_history.txt")
        history_writer = AppendOnlyGlossHistory(history_path)
        print(f"[Gloss紀錄] 追加寫入：{history_path}")
    except Exception as exc:
        logger.exception("Gloss 歷史紀錄初始化失敗：%s", exc)
        print(f"⚠️ Gloss 歷史紀錄無法啟用：{exc}")

    text_to_gloss_converter = None
    try:
        text_to_gloss_converter = TextToGlossConverter(
            excel_path=resource_path("database.xlsx"),
            use_real_api=gemini_use_real_api,
        )
        print(f"[Text-to-Gloss] 已載入 {len(text_to_gloss_converter.gloss_vocabulary)} 個 Excel 詞彙")
    except Exception as exc:
        logger.exception("Text-to-Gloss 初始化失敗：%s", exc)
        print(f"⚠️ Text-to-Gloss 尚未啟用：{exc}")

    translator = TranslationWorker(
        api_key=api_key,
        use_real_api=gemini_use_real_api,
        text_to_gloss_converter=text_to_gloss_converter,
        history_writer=history_writer,
    )

    # API 驗證（保持不變）
    def _verify_api():
        import requests as _req
        key = os.getenv("GEMINI_API_KEY", "").strip()
        test_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"]
        print("\n🔍 [LLM 驗證] 正在確認可用模型...")
        for m in test_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
            try:
                r = _req.post(url, json={
                    "contents": [{"parts": [{"text": "hi"}]}],
                    "generationConfig": {"maxOutputTokens": 5}
                }, timeout=8)
                if r.status_code == 200:
                    print(f"  ✅ [{m}] 可用")
                    break
                else:
                    print(f"  ❌ [{m}] 狀態碼 {r.status_code}")
            except Exception as e:
                print(f"  ❌ [{m}] 連線失敗: {e}")
        print("🔍 [LLM 驗證] 完成\n")
    

    cap, camera_index, camera_backend = open_camera()
    if cap is None:
        message = (
            "無法開啟攝影機。請確認：1. Windows 相機權限已開啟；"
            "2. 沒有 Zoom/Teams/瀏覽器占用鏡頭；3. 裝置管理員看得到攝影機。"
        )
        print(f"🚨 {message}")
        logger.error(message)
        return
    threading.Thread(target=_verify_api, daemon=True).start()

    cv2.namedWindow('Gesture Recognition Core', cv2.WINDOW_NORMAL)

    print(f"🟢 系統完全啟動成功！攝影機 index={camera_index}, backend={camera_backend}，請對準鏡頭並比劃手勢... (按 'q' 退出)")
    logger.info("系統啟動完成 camera_index=%s backend=%s", camera_index, camera_backend)

    frame_count = 0
    last_perf_log_time = time.time()
    last_perf_frame_count = 0
    
    # ⭐ 新增：詞彙緩衝管理變數
    last_word = None
    last_word_time = 0
    WORD_COOLDOWN = 1.5
    failed_frame_count = 0
    
    while True: 
        try:
            ret, frame = cap.read()
            if not ret:
                failed_frame_count += 1
                if failed_frame_count % 30 == 0:
                    logger.warning("攝影機連續讀取失敗 count=%s", failed_frame_count)
                if failed_frame_count >= 150:
                    logger.error("攝影機連續讀取失敗過多，停止主迴圈")
                    break
                continue
            failed_frame_count = 0
                
            frame_count += 1
            frame_start_time = time.perf_counter()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            img_h, img_w = frame.shape[:2]  # ⭐ 新增：取得影像尺寸
            
            # ========================================
            # MediaPipe 偵測（加入 Face Mesh）
            # ========================================
            pose_results = pose.process(rgb_frame)
            hand_results = hands.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)  # ⭐ 新增

            # 畫骨架
            hands_present = False
            pose_present = bool(pose_results and pose_results.pose_landmarks)
            if pose_present:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if hand_results.multi_hand_landmarks:
                hands_present = True
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            tracking_ready = bool(
                hands_present
                and pose_present
                and hand_results.multi_handedness
            )

            # ========================================
            # ⭐ A流：提取特徵（修改為回傳兩個值）
            # ========================================
            current_features, ai_tensor = a_stream_extractor.extract_features(
                pose_results, 
                hand_results, 
                face_results,  # ⭐ 新增
                img_w, 
                img_h
            )

            if frame_count % 30 == 0: 
                active_features = {k: v for k, v in current_features.items() if (isinstance(v, bool)) or (isinstance(v, float) and v < 90)}
                print("\n📦 [給B流的特徵]:")
                for key, value in active_features.items():
                    if isinstance(value, float) or type(value).__name__ == 'float64':
                        print(f"  '{key}': {value:.3f}")
                    else:
                        print(f"  '{key}': {value}")
                print("-" * 40)

            # ========================================
            # ⭐ A流：LSTM 預測（新增）
            # ========================================
            ai_result = None
            if lstm_enabled and tracking_ready:
                sequence_buffer.append(ai_tensor)
                
                if len(sequence_buffer) == SEQ_LEN:
                    x_in = torch.tensor([list(sequence_buffer)], dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        outputs = model(x_in)
                        probs = torch.softmax(outputs, dim=1)[0]
                        
                        TOP_K = 5
                        top_k_probs, top_k_indices = torch.topk(probs, TOP_K)
                        
                        ai_top_k_results = [
                            (idx2label[idx.item()], prob.item())
                            for idx, prob in zip(top_k_indices, top_k_probs)
                        ]
                        
                        if top_k_probs[0].item() > 0.65:
                            ai_result = (idx2label[top_k_indices[0].item()], top_k_probs[0].item())

                            # 檢查混淆詞：只在最高分已經過 0.65 門檻時才嘗試合併，
                            # 避免低信心的雜訊幀單純因為候選分數彼此接近，就繞過門檻產生看似篤定的合併結果。
                            confusable_candidates = []
                            for word, conf in ai_top_k_results:
                                if conf >= top_k_probs[0].item() * 0.85:
                                    if decision_fusion.get_confusable_words(word):
                                        confusable_candidates.append((word, conf))

                            if len(confusable_candidates) >= 2:
                                words = [w for w, c in confusable_candidates]
                                first_word = words[0]
                                group = decision_fusion.get_confusable_words(first_word)

                                same_group_words = [w for w in words if w in group]

                                if len(same_group_words) >= 2:
                                    merged_word = decision_fusion.merge_confusable_words(same_group_words)
                                    max_conf = max(c for w, c in confusable_candidates if w in same_group_words)
                                    ai_result = (merged_word, max_conf)
                                    print(f"🔀 偵測到混淆詞: {same_group_words} → {merged_word}")
            
            elif lstm_enabled:
                sequence_buffer.clear()

            # ========================================
            # ⭐ B流：邏輯判斷
            # ========================================
            if tracking_ready:
                logic_result = b_stream_matcher.evaluate_frame_with_confidence(current_features)
            else:
                b_stream_matcher.reset_transient_state()
                logic_result = None
            gesture_name = logic_result[0] if logic_result else None

            # ========================================
            # ⭐ 決策融合（新增）
            # ========================================
            final_word, source = decision_fusion.fuse(
                ai_result,
                logic_result,
                logic_kind=b_stream_matcher.last_match_kind,
            )
            final_confidence = decision_fusion.last_confidence
            
            # ========================================
            # ⭐ 詞彙緩衝區管理（新增）
            # ========================================
            current_time = time.time()
            
            if final_word:
                if final_word != last_word or (current_time - last_word_time) > WORD_COOLDOWN:
                    translator.add_word(final_word, raw_word=decision_fusion.last_raw_word)
                    while True:
                        # 用保留 _A/_B/_N/_S 尾碼的 raw_word_buffer 比對，
                        # 這樣「要_N/要_S」「家_A/家_B」這類語意不同的變體才不會被互相蓋掉。
                        compound_match = compound_resolver.resolve_tail(translator.raw_word_buffer)
                        if not compound_match:
                            break
                        translator.replace_tail_words(
                            compound_match.consumed,
                            normalize_output_word(compound_match.output),
                            raw_word=compound_match.output,
                            now=current_time,
                        )
                        print(f"[複合詞] {'+'.join(compound_match.pattern)} -> {compound_match.output}")
                    last_word = final_word
                    last_word_time = current_time
                    print(f"✅ 新增詞彙: {final_word} (來源: {source})")
                    logger.info(
                        "新增詞彙 word=%s source=%s confidence=%.3f ai=%s logic=%s",
                        final_word,
                        source,
                        final_confidence,
                        ai_result,
                        logic_result,
                    )

            if frame_count % 30 == 0:
                elapsed = max(1e-6, current_time - last_perf_log_time)
                fps = (frame_count - last_perf_frame_count) / elapsed
                frame_ms = (time.perf_counter() - frame_start_time) * 1000
                perf_logger.info(
                    "fps=%.2f frame_ms=%.2f hands=%s ai=%s logic=%s final=%s confidence=%.3f b_candidates=%s b_suppressed=%s",
                    fps,
                    frame_ms,
                    hands_present,
                    ai_result,
                    logic_result,
                    final_word,
                    final_confidence,
                    b_stream_matcher.last_candidate_words,
                    b_stream_matcher.last_suppressed_words,
                )
                last_perf_log_time = current_time
                last_perf_frame_count = frame_count

            translator.check_and_translate(hands_present=hands_present)
            
            # ========================================
            # 畫面顯示
            # ========================================
            hand_status = "Unknown"
            if current_features.get('is_open_HAND'): hand_status = "OPEN (Flat)"
            elif current_features.get('is_index_pointing_HAND'): hand_status = "POINTING"
            elif current_features.get('is_L_shape_HAND'): hand_status = "L-SHAPE"
            elif current_features.get('is_fist_HAND'): hand_status = "FIST"

            cv2.putText(frame, f"State: {hand_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # ⭐ 新增：顯示融合來源
            if source:
                color_map = {
                    "BOTH": (0, 255, 0),
                    "SYNONYM": (100, 200, 255),
                    "CONFUSABLE": (255, 100, 100),
                    "UNCERTAIN": (255, 200, 0),
                    "LOGIC": (200, 200, 200),
                    "AI": (200, 200, 200)
                }
                source_color = color_map.get(source, (150, 150, 150))
                cv2.putText(frame, f"Source: {source} {final_confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, source_color, 2)
            
            cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
            
            if translator.is_translating:
                display_text = "AI 翻譯中..."
                text_color = (0, 255, 255) 
            elif translator.final_sentence:
                display_text = translator.final_sentence
                text_color = (50, 255, 50) 
            else:
                display_text = f"收集單字：{' '.join(translator.word_buffer)}"
                text_color = (255, 255, 255) 

            frame = put_chinese_text(frame, display_text, (20, 410), text_color=text_color, font_size=36)

            cv2.imshow('Gesture Recognition Core', frame)
            
            if cv2.getWindowProperty('Gesture Recognition Core', cv2.WND_PROP_VISIBLE) < 1: break
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        except Exception as e:
            error_details = traceback.format_exc()
            print("🚨 發生錯誤，錯誤訊息已直接輸出到主控台")
            print(error_details)
            logger.exception("主迴圈發生錯誤")
            time.sleep(5)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
