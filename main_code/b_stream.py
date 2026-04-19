import pandas as pd
import re
import time

class BStreamGestureMatcher:

    def __init__(self, excel_path="database.xlsx"):

        print(">> [B流] 系統初始化...")

        try:
            # 強制讀取 工作表3
            self.rules_df = pd.read_excel(
                excel_path,
                sheet_name="工作表3"
            ).set_index('ID')

            print(f">> [B流] ✅ 成功載入 {len(self.rules_df)} 條手勢規則\n")

        except Exception as e:

            print(f">> [B流] ❌ 讀取資料庫失敗: {e}")

            self.rules_df = pd.DataFrame(columns=[
                'ID',
                '中文',
                '主要分類(數字)',
                '是否可強調(TF)',
                '對應壯聲詞(中文)',
                'MediaPipe 關鍵特徵'
            ]).set_index('ID')

        # sequence 狀態機
        self.sequence_states = {}
        self.sequence_timeout = 2.5

        # 穩定 frame 機制
        self.last_result = None
        self.stable_count = 0
        self.required_stable_frames = 3


    def evaluate_frame(self, variables: dict):

        best_match = None
        best_score = 0

        for index, row in self.rules_df.iterrows():

            gesture_name = row['中文']
            condition_str = str(row['MediaPipe 關鍵特徵'])

            # ==========================================
            # 🛡️ 加入防呆 1：如果 Excel 沒填特徵 (空字串或 nan)，直接跳過不處理
            # ==========================================
            if condition_str.lower() == 'nan' or not condition_str.strip():
                continue

            # sequence 手勢
            if "sequence" in condition_str:

                if self._check_sequence(gesture_name, condition_str, variables):
                    return gesture_name

            else:

                # 將條件拆開計算符合比例
                conditions = condition_str.split("and")

                total_conditions = len(conditions)
                true_count = 0

                for cond in conditions:

                    cond = cond.strip()

                    # ==========================================
                    # 🛡️ 加入防呆 2：如果切開後是空字串 (例如結尾多打 and)，直接跳過
                    # ==========================================
                    if not cond:
                        total_conditions -= 1  # 扣除無效條件數量
                        continue

                    if self._evaluate_condition(cond, variables):
                        true_count += 1

                # 🛡️ 防呆 3：避免 total_conditions 變成 0 導致除以零崩潰
                if total_conditions > 0:
                    score = true_count / total_conditions
                else:
                    score = 0

                # 只接受高匹配
                if score > best_score and score > 0.8:

                    best_score = score
                    best_match = gesture_name


        # =========================
        # 穩定幀數機制
        # =========================

        if best_match == self.last_result:

            self.stable_count += 1

        else:

            self.last_result = best_match
            self.stable_count = 1


        if self.stable_count >= self.required_stable_frames:

            return best_match

        else:

            return None



    def _evaluate_condition(self, condition_str: str, variables: dict):

        try:

            safe_condition = condition_str

            # Excel Boolean 修正
            safe_condition = safe_condition.replace("TRUE", "True")
            safe_condition = safe_condition.replace("FALSE", "False")
            safe_condition = safe_condition.replace("true", "True")
            safe_condition = safe_condition.replace("false", "False")

            result = eval(
                safe_condition,
                {"__builtins__": {}},
                variables
            )

            return bool(result)

        except NameError:

            return False

        except Exception as e:

            print(f"❌ 規則解析錯誤 ({condition_str}): {e}")

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
                print(f"⏳ 【{gesture_name}】 Step1")

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
