import pandas as pd
import re
import time
import logging

from core.safe_rule_engine import SafeRuleEvaluator


logger = logging.getLogger(__name__)

class BStreamGestureMatcher:
    def __init__(self, excel_path="database.xlsx"):
        logger.info("[B流] 系統初始化")
        try:
            # 強制讀取 工作表3
            self.rules_df = pd.read_excel(excel_path, sheet_name="工作表3").set_index('ID')
            logger.info("[B流] 成功載入 %s 條手勢規則", len(self.rules_df))
        except Exception as e:
            logger.exception("[B流] 讀取資料庫失敗: %s", e)
            self.rules_df = pd.DataFrame(columns=[
                'ID', '中文', '主要分類', '是否可強調', '對應狀聲詞', 'MediaPipe 關鍵特徵'
            ]).set_index('ID')

        self.sequence_states = {}
        self.sequence_timeout = 2.5
        self.last_result = None
        self.stable_count = 0
        self.required_stable_frames = 3
        self.rule_evaluator = SafeRuleEvaluator()

    def _safe_eval(self, logic_str, variables):
        """相容舊呼叫點：回傳布林結果，但內部不再使用 eval()."""
        return self._evaluate_rule(logic_str, variables).matched

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
        """解析 sequence([Step1], [Step2], [Step3]) 結構"""
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

    def evaluate_frame(self, variables: dict):
        result = self.evaluate_frame_with_confidence(variables)
        return result[0] if result else None

    def evaluate_frame_with_confidence(self, variables: dict):
        best_match = None
        best_confidence = 0.0
        current_time = time.time()

        for gesture_id, row in self.rules_df.iterrows():
            condition = row['MediaPipe 關鍵特徵']
            if pd.isna(condition) or not str(condition).strip():
                continue
                
            condition = str(condition).strip()

            # --- 處理 Sequence (連續動作) ---
            if condition.startswith("sequence"):
                steps = self._parse_sequence(condition)
                if not steps:
                    continue
                    
                # 初始化狀態機
                if gesture_id not in self.sequence_states:
                    self.sequence_states[gesture_id] = {'step': 0, 'timestamp': current_time}
                    
                state = self.sequence_states[gesture_id]
                current_step_idx = state['step']
                
                # 逾時重置
                if current_step_idx > 0 and (current_time - state['timestamp']) > self.sequence_timeout:
                    state['step'] = 0
                    current_step_idx = 0
                
                # 如果已經完成所有步驟，保持觸發狀態一小段時間
                if current_step_idx >= len(steps):
                    if (current_time - state['timestamp']) < 1.0: # 保持 1 秒
                        best_match = row['中文']
                        best_confidence = 1.0
                        break
                    else:
                        state['step'] = 0 # 重置
                        continue
                
                # 驗證當前步驟
                current_step_logic = steps[current_step_idx]
                evaluation = self._evaluate_rule(current_step_logic, variables)
                if evaluation.matched:
                    state['step'] += 1
                    state['timestamp'] = current_time
                    # 判斷是否為最後一步
                    if state['step'] >= len(steps):
                        best_match = row['中文']
                        best_confidence = min(1.0, 0.65 + 0.35 * evaluation.confidence)
                        break

            # --- 處理一般靜態/單一動作 ---
            else:
                # 一般動作也把逗號替換成 and，以防 Excel 寫錯
                condition_logic = condition.replace(',', ' and ')
                evaluation = self._evaluate_rule(condition_logic, variables)
                if evaluation.matched:
                    best_match = row['中文']
                    best_confidence = max(0.5, evaluation.confidence)
                    break

        # --- 穩定幀機制 ---
        if best_match:
            if best_match == self.last_result:
                self.stable_count += 1
            else:
                self.last_result = best_match
                self.stable_count = 1
                
            if self.stable_count >= self.required_stable_frames:
                stability_score = min(1.0, self.stable_count / self.required_stable_frames)
                confidence = min(1.0, best_confidence * 0.8 + stability_score * 0.2)
                return best_match, confidence
        else:
            self.stable_count = 0
            
        return None

    def _evaluate_condition(self, condition_str: str, variables: dict):

        try:

            safe_condition = condition_str

            # Excel Boolean 修正
            safe_condition = safe_condition.replace("TRUE", "True")
            safe_condition = safe_condition.replace("FALSE", "False")
            safe_condition = safe_condition.replace("true", "True")
            safe_condition = safe_condition.replace("false", "False")

            return self._safe_eval(safe_condition, variables)

        except NameError:

            return False

        except Exception as e:

            logger.exception("規則解析錯誤 (%s): %s", condition_str, e)

            return False

    def _check_sequence(self, gesture_name, condition_str, variables):
        match = re.search(
            r'sequence\(\s*\[(.*?)\]\s*,\s*\[(.*?)\]\s*\)',
            condition_str
        )
        if not match:
            return False

        step1_cond, step2_cond = match.groups()

        if gesture_name not in self.sequence_states:
            self.sequence_states[gesture_name] = {
                'step': 0,
                'timestamp': 0
            }

        current_time = time.time()
        state = self.sequence_states[gesture_name]

        # 逾時重置 (只有在 Step 1 等待 Step 2 時才算逾時)
        if state['step'] == 1 and (current_time - state['timestamp']) > self.sequence_timeout:
            state['step'] = 0

        # ==========================================
        # 🌟 Step 1: 檢查第一步
        # ==========================================
        if state['step'] == 0:
            step1_logic = step1_cond.replace(',', ' and ')
            if self._evaluate_condition(step1_logic, variables):
                state['step'] = 1
                state['timestamp'] = current_time
                logger.debug("【%s】 Step1", gesture_name)

        # ==========================================
        # 🌟 Step 2: 檢查第二步 (若成功，進入保持狀態)
        # ==========================================
        elif state['step'] == 1:
            step2_logic = step2_cond.replace(',', ' and ')
            if self._evaluate_condition(step2_logic, variables):
                state['step'] = 2  # 🟢 改成進入 Step 2 (完成保持狀態)
                state['timestamp'] = current_time
                return True

        # ==========================================
        # 🌟 Step 3: VIP 完成保持狀態 (騙過 3 幀穩定器)
        # ==========================================
        elif state['step'] == 2:
            # 讓這個成功狀態存活 0.5 秒，確保能印到畫面上
            if (current_time - state['timestamp']) < 0.5:
                return True
            else:
                state['step'] = 0  # 0.5 秒後，真正歸零準備抓下一次
                return False

        return False
