import sys
sys.dont_write_bytecode = True

import pandas as pd
import re
import time
import logging

from core.safe_rule_engine import SafeRuleEvaluator
from core.feature_registry import FeatureRegistry
from core.word_normalization import format_candidates


logger = logging.getLogger(__name__)

class BStreamGestureMatcher:
    """
    B 流規則辨識器。

    主要職責：
    1. 從 database.xlsx 讀取手語規則。
    2. 將 A 流輸出的 current_features 套入 Excel 條件。
    3. 支援一般單幀規則與 sequence([...], [...]) 連續動作規則。
    4. 透過安全 AST evaluator 評估規則，不使用 eval()。
    5. 回傳手語詞與規則信心分數，交給 fusion.py 與 AI 流整合。
    """
    STATIC_CONFIDENCE = 0.75
    SEQUENCE_CONFIDENCE = 0.90

    def __init__(self, excel_path="database.xlsx"):
        logger.info("[B流] 系統初始化")
        self.rule_evaluator = SafeRuleEvaluator()
        self.loaded_ok = False
        try:
            # 強制讀取 工作表3
            rules_df = pd.read_excel(excel_path, sheet_name="工作表3")
            self._validate_rules_dataframe(rules_df)
            self.rules_df = rules_df.set_index('ID')
            self.loaded_ok = True
            logger.info("[B流] 成功載入 %s 條手勢規則", len(self.rules_df))
        except Exception as e:
            logger.exception("[B流] 讀取資料庫失敗: %s", e)
            raise RuntimeError(f"B流規則資料庫載入失敗: {e}") from e

        self.sequence_states = {}
        self.sequence_timeout = 2.5
        self.sequence_emit_cooldown = 1.0
        # 每個靜態規則各自的連續命中幀數，key 是 gesture_id。
        # 用字典而不是單一變數，避免某個規則（例如另一手的 sequence）贏得某一幀時，
        # 把完全不相關的靜態手勢已經累積的穩定度也一起清空。
        self.static_stability = {}
        self.required_stable_frames = 3
        self.last_match_kind = None
        self.last_match_ids = ()
        self.last_candidate_words = ()
        self.last_candidate_ids = ()
        self.last_suppressed_words = ()
        self.last_suppressed_ids = ()
        self._last_static_log_state = None
        self._last_suppressed_log_state = None
        self._feature_contract_validated = False

    def _validate_rules_dataframe(self, rules_df):
        required_columns = {"ID", "中文", "MediaPipe 關鍵特徵"}
        missing_columns = sorted(required_columns - set(rules_df.columns))
        if missing_columns:
            raise ValueError(f"工作表3 缺少必要欄位: {missing_columns}")

        condition_col = "MediaPipe 關鍵特徵"
        executable = rules_df[
            rules_df[condition_col].notna()
            & rules_df[condition_col].astype(str).str.strip().ne("")
        ].copy()
        if executable["ID"].isna().any():
            raise ValueError("存在有規則條件但沒有 ID 的資料列")
        if executable["中文"].isna().any():
            raise ValueError("存在有規則條件但沒有中文名稱的資料列")

        ids = executable["ID"].astype(str).str.strip()
        duplicates = sorted(ids[ids.duplicated(False)].unique())
        if duplicates:
            raise ValueError(f"存在重複規則 ID: {duplicates}")

        for _, row in executable.iterrows():
            gesture_id = str(row["ID"]).strip()
            condition = self._normalize_condition(row[condition_col])
            steps = self._parse_sequence(condition) if condition.startswith("sequence") else [condition.replace(',', ' and ')]
            if not steps:
                raise ValueError(f"規則 {gesture_id} 的 sequence 格式無法解析")
            for step in steps:
                names = self.rule_evaluator.names_in(step)
                variables = {name: FeatureRegistry.default_for_name(name) for name in names}
                result = self.rule_evaluator.evaluate(step, variables)
                if result.error:
                    raise ValueError(f"規則 {gesture_id} 語法錯誤: {result.error}")

    @staticmethod
    def _normalize_condition(condition):
        return (
            str(condition)
            .replace("\u00a0", " ")
            .strip()
            .replace("TRUE", "True")
            .replace("FALSE", "False")
            .replace("true", "True")
            .replace("false", "False")
        )

    def reset_transient_state(self):
        """Reset partial gestures after hand/pose tracking is interrupted."""
        for state in self.sequence_states.values():
            state['step'] = 0
            state['timestamp'] = 0.0
        self.static_stability = {}
        self.last_match_kind = None
        self.last_match_ids = ()
        self.last_candidate_words = ()
        self.last_candidate_ids = ()
        self.last_suppressed_words = ()
        self.last_suppressed_ids = ()
        self._last_static_log_state = None
        self._last_suppressed_log_state = None

    def _validate_feature_contract(self, variables):
        referenced_names = set()
        for _, row in self.rules_df.iterrows():
            condition = row.get('MediaPipe 關鍵特徵')
            if pd.isna(condition) or not str(condition).strip():
                continue
            condition = self._normalize_condition(condition)
            steps = self._parse_sequence(condition) if condition.startswith("sequence") else [condition.replace(',', ' and ')]
            for step in steps or ():
                referenced_names.update(self.rule_evaluator.names_in(step))
        missing = sorted(referenced_names - set(variables))
        if missing:
            raise RuntimeError(f"Excel 規則引用 A 流未提供的特徵: {missing}")
        self._feature_contract_validated = True

    def _evaluate_rule(self, logic_str, variables):
        if not logic_str or pd.isna(logic_str):
            return self.rule_evaluator.evaluate("", variables)

        result = self.rule_evaluator.evaluate(str(logic_str), variables)
        if result.error:
            logger.debug("[B流] 規則解析錯誤: %s | %s", logic_str, result.error)
        if result.unknown_names:
            logger.debug("[B流] 規則使用未知特徵: %s | %s", sorted(result.unknown_names), logic_str)
        return result

    def _parse_sequence(self, condition_str):
        """
        解析 sequence([Step1], [Step2], [Step3]) 結構。

        Excel 規則可以寫成：
            sequence([is_flat_HAND], [move_downwards_RIGHT_HAND])

        每個中括號代表一個階段。B 流會用 sequence_states 記住目前走到第幾步，
        並用 sequence_timeout 避免使用者停太久後仍誤判為同一個連續動作。
        """
        if not isinstance(condition_str, str) or not condition_str.startswith("sequence"):
            return None
            
        # 萃取括號內的所有步驟，例如: [A == True, B < 1], [C == True]
        match = re.search(r'sequence\((.*)\)', condition_str)
        if not match:
            return None
            
        inner_content = match.group(1)
        # 用正則表達式抓取中括號內的內容
        steps_raw = re.findall(r'\[(.*?)\]', inner_content)
        
        steps = []
        for step in steps_raw:
            # 將 Excel 內的逗號(,)轉為 and 邏輯
            step_logic = step.replace(',', ' and ')
            steps.append(step_logic)
            
        return steps

    def _rule_priority(self, row) -> int:
        for key in ("優先級", "Priority", "排序", "Order"):
            if key in row and pd.notna(row[key]):
                try:
                    return int(row[key])
                except Exception:
                    continue
        return 0

    def _rule_specificity(self, condition: str) -> int:
        return len(self.rule_evaluator.names_in(condition))

    def _pick_better_candidate(self, current, new):
        if current is None:
            return new
        if new["sort_key"] > current["sort_key"]:
            return new
        if new["sort_key"] < current["sort_key"]:
            return current

        # Equal rules are an ambiguity, not a reason to prefer the Excel row above.
        words = list(current["words"])
        for word in new["words"]:
            if word not in words:
                words.append(word)
        gesture_ids = list(current["gesture_ids"])
        for gesture_id in new["gesture_ids"]:
            if gesture_id not in gesture_ids:
                gesture_ids.append(gesture_id)
        current["words"] = words
        current["gesture_ids"] = gesture_ids
        current["word"] = format_candidates(words)
        return current

    @staticmethod
    def _candidate_words(candidates):
        return tuple(candidate["word"] for candidate in candidates)

    @staticmethod
    def _candidate_ids(candidates):
        ids = []
        for candidate in candidates:
            ids.extend(candidate["gesture_ids"])
        return tuple(ids)

    def _update_candidate_visibility(self, candidates, selected):
        self.last_candidate_words = self._candidate_words(candidates)
        self.last_candidate_ids = self._candidate_ids(candidates)

        if not selected:
            self.last_suppressed_words = ()
            self.last_suppressed_ids = ()
            self._last_suppressed_log_state = None
            return

        selected_ids = set(selected["gesture_ids"])
        suppressed = [
            candidate
            for candidate in candidates
            if not selected_ids.intersection(candidate["gesture_ids"])
        ]
        self.last_suppressed_words = self._candidate_words(suppressed)
        self.last_suppressed_ids = self._candidate_ids(suppressed)

        suppressed_state = (selected["word"], self.last_suppressed_words, self.last_suppressed_ids)
        if suppressed and suppressed_state != self._last_suppressed_log_state:
            logger.info(
                "[B流][candidate] 同幀命中但未選用 words=%s ids=%s selected=%s selected_ids=%s",
                self.last_suppressed_words,
                self.last_suppressed_ids,
                selected["word"],
                tuple(selected["gesture_ids"]),
            )
            self._last_suppressed_log_state = suppressed_state

    def _log_static_progress(self, candidate, stable_count, confidence=None):
        ids = tuple(candidate["gesture_ids"])
        if stable_count < self.required_stable_frames:
            state = ("pending", ids, stable_count)
            if state != self._last_static_log_state:
                logger.info(
                    "[B流][static] %s 命中候選 %s/%s ids=%s",
                    candidate["word"],
                    stable_count,
                    self.required_stable_frames,
                    ids,
                )
                self._last_static_log_state = state
            return

        state = ("stable", ids)
        if state != self._last_static_log_state:
            logger.info(
                "[B流][static] %s 穩定 %s/%s -> %s (conf=%.3f) ids=%s",
                candidate["word"],
                stable_count,
                self.required_stable_frames,
                candidate["word"],
                confidence if confidence is not None else candidate["confidence"],
                ids,
            )
            self._last_static_log_state = state

    def evaluate_frame(self, variables: dict):
        """
        舊版相容 API。

        舊主程式只需要手語詞，因此這個方法仍回傳 word 或 None。
        新版 main.py 會改呼叫 evaluate_frame_with_confidence() 取得信心分數。
        """
        result = self.evaluate_frame_with_confidence(variables)
        return result[0] if result else None

    def evaluate_frame_with_confidence(self, variables: dict):
        """
        評估目前影格的所有 Excel 規則，回傳 (word, confidence) 或 None。
        已移除早退的 break 機制，改用民主競標制，挑選當前影格中信心分數最高的最佳結果。
        """
        if not self._feature_contract_validated:
            self._validate_feature_contract(variables)

        best_candidate = None
        frame_candidates = []
        current_time = time.monotonic()
        self.last_match_kind = None
        self.last_match_ids = ()
        self.last_candidate_words = ()
        self.last_candidate_ids = ()
        self.last_suppressed_words = ()
        self.last_suppressed_ids = ()

        for row_index, (gesture_id, row) in enumerate(self.rules_df.iterrows()):
            condition = row['MediaPipe 關鍵特徵']
            if pd.isna(condition) or not str(condition).strip():
                continue
                
            condition = self._normalize_condition(condition)
            priority = self._rule_priority(row)
            specificity = self._rule_specificity(condition)

            # --- 處理 Sequence (連續動作) ---
            if condition.startswith("sequence"):
                steps = self._parse_sequence(condition)
                if not steps:
                    continue
                    
                # 初始化狀態機
                if gesture_id not in self.sequence_states:
                    self.sequence_states[gesture_id] = {
                        'step': 0,
                        'timestamp': current_time,
                        'cooldown_until': 0.0,
                    }
                    
                state = self.sequence_states[gesture_id]

                if current_time < state.get('cooldown_until', 0.0):
                    continue

                current_step_idx = state['step']
                
                # 逾時重置
                if current_step_idx > 0 and (current_time - state['timestamp']) > self.sequence_timeout:
                    state['step'] = 0
                    current_step_idx = 0

                if current_step_idx >= len(steps):
                    state['step'] = 0
                    continue
                
                # 驗證當前步驟
                current_step_logic = steps[current_step_idx]
                evaluation = self._evaluate_rule(current_step_logic, variables)
                if evaluation.matched:
                    display_word = str(row.get('中文', gesture_id)).strip() or gesture_id
                    print(f"[B流][sequence] {display_word} Step {current_step_idx + 1}/{len(steps)} 命中")
                    state['step'] += 1
                    state['timestamp'] = current_time
                    # 判斷是否為最後一步
                    if state['step'] >= len(steps):
                        conf = self.SEQUENCE_CONFIDENCE
                        word = str(row['中文']).strip()
                        candidate = {
                            "word": word,
                            "words": [word],
                            "confidence": conf,
                            "priority": priority,
                            "specificity": specificity,
                            "kind": "sequence",
                            "gesture_id": gesture_id,
                            "gesture_ids": [gesture_id],
                            "sort_key": (1, priority, specificity, conf),
                        }
                        frame_candidates.append(candidate)
                        print(f"[B流][sequence] {display_word} 完成 -> {row['中文']} (conf={conf:.3f})")
                        state['step'] = 0
                        state['cooldown_until'] = current_time + self.sequence_emit_cooldown

            # --- 處理一般靜態/單一動作 ---
            else:
                # 一般動作也把逗號替換成 and，以防 Excel 寫錯
                condition_logic = condition.replace(',', ' and ')
                evaluation = self._evaluate_rule(condition_logic, variables)
                if evaluation.matched:
                    # 這條規則自己這一幀成立，累加它自己的連續命中幀數。
                    # 用 gesture_id 當 key，不會被其他不相關規則（例如另一手完成的 sequence）覆蓋。
                    self.static_stability[gesture_id] = self.static_stability.get(gesture_id, 0) + 1
                    conf = self.STATIC_CONFIDENCE
                    word = str(row['中文']).strip()
                    candidate = {
                        "word": word,
                        "words": [word],
                        "confidence": conf,
                        "priority": priority,
                        "specificity": specificity,
                        "kind": "static",
                        "gesture_id": gesture_id,
                        "gesture_ids": [gesture_id],
                        "sort_key": (0, priority, specificity, conf),
                    }
                    frame_candidates.append(candidate)
                else:
                    # 這一幀條件不成立，代表這條規則自己的連續命中中斷，重置它自己的計數。
                    self.static_stability.pop(gesture_id, None)

        # --- 穩定幀機制 ---
        # 只有 sequence，或是自己已經連續命中滿 required_stable_frames 幀的 static，
        # 才有資格代表這一幀輸出。這樣即使某一幀是 sequence 贏得輸出，
        # 也不會影響其他不相關 static 規則已經累積的穩定度（各自獨立用 gesture_id 追蹤）。
        eligible_candidates = [
            candidate for candidate in frame_candidates
            if candidate["kind"] == "sequence"
            or self.static_stability.get(candidate["gesture_id"], 0) >= self.required_stable_frames
        ]

        best_candidate = None
        for candidate in eligible_candidates:
            best_candidate = self._pick_better_candidate(best_candidate, candidate)

        self._update_candidate_visibility(frame_candidates, best_candidate)

        if best_candidate:
            best_match = best_candidate["word"]
            if best_candidate["kind"] == "sequence":
                self.last_match_kind = "sequence"
                self.last_match_ids = tuple(best_candidate["gesture_ids"])
                return best_match, best_candidate["confidence"]

            stable_count = self.static_stability.get(best_candidate["gesture_id"], self.required_stable_frames)
            stability_score = min(1.0, stable_count / self.required_stable_frames)
            confidence = min(1.0, best_candidate["confidence"] * 0.8 + stability_score * 0.2)
            self._log_static_progress(best_candidate, stable_count, confidence)
            self.last_match_kind = "static"
            self.last_match_ids = tuple(best_candidate["gesture_ids"])
            return best_match, confidence

        # 沒有規則累積到可以輸出的程度時，仍找出目前領先的 static 候選印出進度，
        # 純粹方便除錯用，不影響任何判斷。
        pending_statics = [c for c in frame_candidates if c["kind"] == "static"]
        if pending_statics:
            leading = max(
                pending_statics,
                key=lambda c: (self.static_stability.get(c["gesture_id"], 0), c["sort_key"]),
            )
            self._log_static_progress(leading, self.static_stability.get(leading["gesture_id"], 0))
        else:
            self._last_static_log_state = None

        return None
