print("[追蹤雷達] 1. 成功讀取檔案，開始執行...")

import cv2
import numpy as np
import time
import traceback
import pandas as pd
from PIL import Image, ImageDraw, ImageFont 
import threading  
import requests   

print("[追蹤雷達] 2. 成功載入基礎套件...")
import mediapipe as mp

# 載入獨立出來的 A 流與 B 流
from a_stream import AStreamFeatureExtractor
try:
    from b_stream import BStreamGestureMatcher
    print("[追蹤雷達] 4. 成功找到 a_stream 與 b_stream 模組...")
except ImportError:
    print("⚠️ 找不到 b_stream_matcher.py，程式可能無法完整運作。")

# 中文字體繪製器
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
# 🌟 LLM 翻譯模組 (2.5 專屬跳島機制)
# ==========================================
class TranslationWorker:
    def __init__(self, api_key, use_real_api=False):
        self.use_real_api = use_real_api
        self.api_key = api_key
        self.word_buffer = []      
        self.final_sentence = ""   
        self.is_translating = False
        self.last_word_time = time.time()
        self.last_added_word = ""  
        self.last_hands_seen_time = time.time() # 🌟 新增：記錄最後一次看到手的時間

        # ⏱️ 時間設定區
        self.debounce_time = 2.0     
        self.translate_delay = 2.0   # 🌟 修改：畫面沒手超過 2 秒就翻譯 (原為 2.5 秒)
        self.display_duration = 5.0  
        self.translation_done_time = 0 

        # 💡 提示詞工程
        self.system_instruction = """
        你是一個專業的台灣手語(TSL)翻譯員。
        使用者會輸入一連串的手語單字（Glosses），請你根據台灣手語的文法習慣，
        將它們重組、潤飾成一句自然、通順的繁體中文日常用語。

        【翻譯規則與範例】
        1. 手語常將時間副詞放句首，請調整至自然位置。
           - 輸入：["昨天", "我", "學校", "去"] -> 輸出：我昨天去學校。
           - 輸入：["明天", "雨", "下"] -> 輸出：明天會下雨。
        2. 常見問候語請翻譯成最道地的口語。
           - 輸入：["早上", "平安", "好"] -> 輸出：早安，你好。
           - 輸入：["你", "吃飯", "已經"] -> 輸出：你吃過飯了嗎？
        3. 強調語氣（如：非常、很）
           - 輸入：["爸爸", "生氣", "很"] -> 輸出：爸爸很生氣。
        4. 人稱與狀態
           - 輸入：["我們", "學生"] -> 輸出：我們是學生。
        5. 特殊專有名詞組合
           - 輸入：["學生", "證明"] -> 輸出：學生證。
           - 輸入：["我", "不是"] -> 輸出：不是我。
           - 輸入：["爸爸", "弟弟"] -> 輸出：叔叔。
           - 輸入：["爸爸", "弟弟", "太太"] -> 輸出：嬸嬸。
           - 輸入：["結婚", "女生"] -> 輸出：太太。

        請直接輸出翻譯後的句子，絕對不要輸出任何解釋、引言或標點符號之外的廢話。
        """

    def add_word(self, word):
        clean_word = word.split('_')[0] if '_' in word else word
        if clean_word != self.last_added_word or (time.time() - self.last_word_time > self.debounce_time):
            self.word_buffer.append(clean_word)
            self.last_added_word = clean_word
            self.last_word_time = time.time()
            self.final_sentence = "" 
            print(f"[翻譯雷達] 目前收集單字: {self.word_buffer}")

    def _call_llm(self, words_to_translate):
        self.is_translating = True
        try:
            if not self.use_real_api:
                # 🛠️ 開發測試模式：假裝運算 1 秒，不扣 API 額度也不用連網
                time.sleep(1) 
                self.final_sentence = f"[測試翻譯] {' '.join(words_to_translate)}"
            else:
                print(f"[LLM] 發送雲端請求: {words_to_translate}...")
                clean_api_key = self.api_key.strip()
                
                # 🚀 終極解決方案：鎖定您帳號確定可用的 2.5 系列與最新模型
                fallback_models = [
                    "gemini-2.5-flash",         # 驗證可用的最新標準版
                    "gemini-2.5-flash-lite",    # 輕量版 (若有)
                    "gemini-flash-latest"       # 萬用備援別名
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
                        "temperature": 0.2
                    }
                }
                
                max_retries = len(fallback_models)
                for attempt in range(max_retries):
                    current_model = fallback_models[attempt]
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={clean_api_key}"
                    
                    try:
                        response = requests.post(url, headers=headers, json=payload, timeout=8)
                        
                        # 攔截被拒絕的連線
                        if response.status_code in [404, 429, 503]:
                            if response.status_code == 404:
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
                        
                        # 成功拿到資料！
                        data = response.json()
                        self.final_sentence = data['candidates'][0]['content']['parts'][0]['text'].strip()
                        print(f"[LLM] 翻譯成功 🎉 (使用的模型: {current_model}): {self.final_sentence}")
                        return 
                        
                    except requests.exceptions.Timeout:
                        print(f"⚠️ [{current_model}] 請求超時 (Timeout)，切換備用路線...")
                        continue 
                    except requests.exceptions.ConnectionError:
                        print("\n🚨 [LLM 網路錯誤] 無法連線！請檢查學校防火牆或切換手機熱點。")
                        self.final_sentence = "網路連線失敗"
                        return

                print("\n🚨 [LLM 錯誤] 所有的 2.5 備用模型都忙碌中或無可用額度。")
                self.final_sentence = "伺服器忙碌中，請稍後再試"
                
        except Exception as e:
            print(f"\n🚨 [LLM 未知錯誤] 翻譯失敗: {e}")
            self.final_sentence = "發生未知錯誤(看終端機)"
        finally:
            self.is_translating = False
            self.last_added_word = "" 
            self.translation_done_time = time.time() 

    def check_and_translate(self, hands_present=False):
        # 🌟 檢查翻譯結果是否顯示夠久了，超過 display_duration 就自動清空
        if self.final_sentence and not self.is_translating:
            if time.time() - self.translation_done_time > self.display_duration:
                self.final_sentence = "" # 清空句子後，UI 就會自動變回「收集單字」狀態

        # 🌟 記錄手部最後出現的時間
        if hands_present:
            self.last_hands_seen_time = time.time()

        # 🌟 觸發翻譯邏輯大升級！
        if len(self.word_buffer) > 0 and not self.is_translating:
            # 條件 1: 畫面中「沒有手」且持續超過 2 秒 (您提議的完美解法)
            no_hands_trigger = (not hands_present) and (time.time() - self.last_hands_seen_time > self.translate_delay)
            # 條件 2: 雖然有手，但已經超過 5 秒沒有打出新單字 (防呆機制，避免手卡在畫面中系統死等)
            timeout_trigger = (time.time() - self.last_word_time > 5.0)

            if no_hands_trigger or timeout_trigger: 
                words_to_translate = list(self.word_buffer) 
                self.word_buffer.clear()
                threading.Thread(target=self._call_llm, args=(words_to_translate,), daemon=True).start()

def main():
    print("[追蹤雷達] 5. 進入 main() 主程式區塊...")
    
    a_stream_extractor = AStreamFeatureExtractor()
    b_stream_matcher = BStreamGestureMatcher(excel_path="database.xlsx")

    try:
        df = pd.read_excel("database.xlsx")
        print("\n" + "="*50)
        print("🔍 [Excel X光機] 檢查 B 流讀取到的規則：")
        for idx, row in df.iterrows():
            word = str(row.get('中文', '')).strip()
            if word in ['你', '我', '平安', '謝謝']:
                condition = row.get('MediaPipe 關鍵特徵', '【找不到這個欄位】')
                print(f"👉 單字: {word}")
                print(f"   真實條件: [{condition}]")
        print("="*50 + "\n")
    except Exception as e:
        print(f"⚠️ 無法啟動 Excel X光機: {e}")

    print("[追蹤雷達] 6. 準備初始化 MediaPipe 視覺引擎...")
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hands = mp_hands.Hands(max_num_hands=2, 
                           min_detection_confidence=0.5,   
                           min_tracking_confidence=0.5)     
    # 🌟 啟動 LLM 翻譯大腦 (已設定真實 API Key 且開啟連線)
    translator = TranslationWorker(api_key="AIzaSyALflEImMyOR-ZZvzlIDS_q5Wd1NtEUt8A", use_real_api=True)

    # 🌟 啟動時驗證 API Key 與可用模型（背景執行，不阻塞主程式）
    def _verify_api():
        import requests as _req
        key = "AIzaSyALflEImMyOR-ZZvzlIDS_q5Wd1NtEUt8A".strip()
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
    threading.Thread(target=_verify_api, daemon=True).start()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not cap.isOpened(): cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow('Gesture Recognition Core', cv2.WINDOW_NORMAL)

    print("🟢 系統完全啟動成功！請對準鏡頭並比劃手勢... (按 'q' 退出)")

    frame_count = 0
    while True: 
        try:
            ret, frame = cap.read()
            if not ret: continue
                
            frame_count += 1
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            pose_results = pose.process(rgb_frame)
            hand_results = hands.process(rgb_frame)

            # 畫骨架
            hands_present = False # 🌟 新增：預設畫面中沒有手
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if hand_results.multi_hand_landmarks:
                hands_present = True # 🌟 新增：偵測到手了！
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            current_features = a_stream_extractor.extract_features(pose_results, hand_results)

            if frame_count % 30 == 0: 
                active_features = {k: v for k, v in current_features.items() if (isinstance(v, bool)) or (isinstance(v, float) and v < 90)}
                print("\n📦 [給B流的特徵]:")
                for key, value in active_features.items():
                    if isinstance(value, float) or type(value).__name__ == 'float64':
                        print(f"  '{key}': {value:.3f}")
                    else:
                        print(f"  '{key}': {value}")
                print("-" * 40) 

            gesture_name = b_stream_matcher.evaluate_frame(current_features)
            
            if gesture_name:
                translator.add_word(gesture_name)

            # 🌟 修改：將「畫面中有沒有手」的情報傳給翻譯機
            translator.check_and_translate(hands_present=hands_present)
            
            hand_status = "Unknown"
            if current_features.get('is_open_HAND'): hand_status = "OPEN (Flat)"
            elif current_features.get('is_index_pointing_HAND'): hand_status = "POINTING"
            elif current_features.get('is_L_shape_HAND'): hand_status = "L-SHAPE"
            elif current_features.get('is_fist_HAND'): hand_status = "FIST"

            cv2.putText(frame, f"State: {hand_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Z-Align: {current_features.get('vector_align_HAND_8_CAMERA_AXIS', 0):.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"NOSE dist: {current_features.get('dist_HAND_8_FACE_1', 0):.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if current_features.get('move_apart_horizontally_RIGHT_HAND_LEFT_HAND'):
                cv2.putText(frame, "HANDS MOVING APART!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

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
            with open("error_log.txt", "w", encoding="utf-8") as f: f.write(error_details)
            print("🚨 發生錯誤！請查看 error_log.txt")
            time.sleep(5)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()