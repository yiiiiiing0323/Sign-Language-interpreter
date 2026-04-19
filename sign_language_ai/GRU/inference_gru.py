import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
import requests
from PIL import Image, ImageDraw, ImageFont
from collections import deque

# =============================
# 1. 模型與工具函式定義
# =============================
class SignGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def angle_between_points(a, b, c):
    """計算關節夾角"""
    ba, bc = a - b, c - b
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))

def hand_angles(hand_lms):
    """計算 15 個手指角度"""
    fingers = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
    angles = []
    for f in fingers:
        pts = np.array([[hand_lms[i].x, hand_lms[i].y, hand_lms[i].z] for i in f])
        for j in range(1, 4):
            angles.append(angle_between_points(pts[j-1], pts[j], pts[j+1]))
    return angles

def draw_text(img, text, pos=(20, 40), color=(0, 255, 0)):
    """支援中文顯示"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 32)
    except: font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =============================
# 2. LLM 翻譯函式 (對齊 gemma3:4b)
# =============================
def translate_sign_sequence(word_list):
    if not word_list: return ""
    # 先去重，避免重複的單字干擾 LLM（例如：謝謝、謝謝 -> 謝謝）
    unique_words = []
    for w in word_list:
        if not unique_words or w != unique_words[-1]:
            unique_words.append(w)
            
    combined_words = "、".join(unique_words)
    print(f"\n[DEBUG] 準備發送給 LLM: {combined_words}")
    
    # 改回最基礎的 generate 接口
    url = "http://localhost:11434/api/generate"
    
    # 針對 Gemma 調整 Prompt，直接下達指令
    prompt = f"將以下手語詞彙轉換成一句自然流暢的中文句子：{combined_words}。直接輸出句子，不要任何解釋。"
    
    payload = {
        "model": "gemma3:4b", 
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # 使用 verify=False 排除某些 SSL 或環境代理問題
        response = requests.post(url, json=payload, timeout=15)
        
        # 如果還是 404，印出 Ollama 回傳的錯誤內容方便診斷
        if response.status_code == 404:
            print(f"[DEBUG] 伺服器回傳 404。請檢查 Ollama 是否支援此路徑。")
            return "LLM 路徑錯誤"
            
        response.raise_for_status()
        data = response.json()
        
        # 取得翻譯結果
        result = data.get('response', '').strip()
        
        print(f"[DEBUG] LLM 成功回傳: {result}")
        return result
    except Exception as e:
        print(f"[DEBUG] LLM 請求失敗: {e}")
        return "翻譯失敗，請檢查網路或伺服器"

# =============================
# 3. 初始化載入
# =============================
try:
    with open("label_map.json", "r", encoding="utf-8") as f:
        idx2label = {int(k): v for k, v in json.load(f).items()}
except:
    print("找不到 label_map.json，請檢查路徑")
    idx2label = {}

INPUT_DIM = 206 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SignGRU(INPUT_DIM, 128, 2, len(idx2label)).to(device)
try:
    model.load_state_dict(torch.load("sign_gru.pth", map_location=device))
    print("成功載入 206維 GRU 模型")
except:
    print("載入權重失敗，請確認 sign_gru.pth 是否為 206 維")

model.eval()

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=2)
face = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose()

# --- 參數與緩衝區 ---
SEQ_LEN = 30
buffer = []
sentence_words = []    
last_word = ""         
cooldown_counter = 0   
llm_result = ""
display_text = "等待手勢..."

cap = cv2.VideoCapture(0)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- A. Pose (正規化基準) ---
    p_res = pose.process(rgb)
    p_feat, e_ang, norm_dist = [], [0, 0], 1.0
    if p_res.pose_landmarks:
        lms = p_res.pose_landmarks.landmark
        p11, p12 = np.array([lms[11].x, lms[11].y, lms[11].z]), np.array([lms[12].x, lms[12].y, lms[12].z])
        norm_dist = np.linalg.norm(p11 - p12) + 1e-6
        for i in [0, 11, 12, 13, 14, 23, 15, 16]:
            p_feat += [lms[i].x / norm_dist, lms[i].y / norm_dist, lms[i].z / norm_dist]
        e_ang[0] = angle_between_points(p11, np.array([lms[13].x, lms[13].y, lms[13].z]), np.array([lms[15].x, lms[15].y, lms[15].z]))
        e_ang[1] = angle_between_points(p12, np.array([lms[14].x, lms[14].y, lms[14].z]), np.array([lms[16].x, lms[16].y, lms[16].z]))
    else: p_feat = [0]*24

    # --- B. Face ---
    f_res = face.process(rgb)
    f_feat = []
    if f_res.multi_face_landmarks:
        for i in [33, 133, 362, 263, 1, 61, 291, 199]:
            lm = f_res.multi_face_landmarks[0].landmark[i]
            f_feat += [lm.x / norm_dist, lm.y / norm_dist, lm.z / norm_dist]
    else: f_feat = [0]*24

    # --- C. Hand ---
    h_res = hands.process(rgb)
    h_feat, f_ang = [], []
    hand_exist = False
    if h_res.multi_hand_landmarks:
        hand_exist = True
        for h in h_res.multi_hand_landmarks[:2]:
            wrist = np.array([h.landmark[0].x, h.landmark[0].y, h.landmark[0].z])
            for lm in h.landmark:
                h_feat += ((np.array([lm.x, lm.y, lm.z]) - wrist) / norm_dist).tolist()
            f_ang += hand_angles(h.landmark)
    while len(h_feat) < 126: h_feat += [0,0,0]
    while len(f_ang) < 30: f_ang += [0]*15

    final = f_feat + h_feat + f_ang + p_feat + e_ang

    # --- D. 預測與 LLM 觸發邏輯 ---
    energy = np.sum(np.abs(h_feat))
    
    # 判斷手是否存在且有在動
    if hand_exist and energy > 0.05: # 門檻調高至 0.05
        cooldown_counter = 0  
        buffer.append(final)
        if len(buffer) > SEQ_LEN: buffer.pop(0)
        if len(buffer) == SEQ_LEN:
            x_in = torch.tensor(buffer, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x_in)
                conf, idx = torch.softmax(out, dim=1).max(dim=1)
            
            if conf.item() > 0.85:
                current_word = idx2label.get(idx.item(), "??")
                if current_word != last_word:
                    sentence_words.append(current_word)
                    last_word = current_word
                display_text = f"正在辨識: {current_word}"
            else:
                display_text = "..."
    else:
        # 手放下或移開鏡頭，開始冷卻計時
        buffer = []
        cooldown_counter += 1
        
        if 0 < cooldown_counter < 60:
            display_text = f"完成動作請靜止... ({cooldown_counter}/60)"
        
        # 剛好到 60 幀觸發 LLM
        if cooldown_counter == 60:
            if sentence_words:
                print("\n[SYSTEM] 開始發送翻譯請求...")
                llm_result = translate_sign_sequence(sentence_words)
                sentence_words = [] 
                last_word = ""
            display_text = "翻譯完成"

    # --- E. 畫面顯示 ---
    frame = draw_text(frame, f"序列: {' '.join(sentence_words)}", pos=(20, 40))
    frame = draw_text(frame, f"LLM: {llm_result}", pos=(20, 90), color=(255, 0, 0))
    frame = draw_text(frame, display_text, pos=(20, 140), color=(255, 255, 0))
    
    cv2.imshow("Sign-to-LLM System", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()