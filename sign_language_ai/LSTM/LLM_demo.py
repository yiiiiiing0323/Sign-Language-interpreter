import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
import requests
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# 1. 模型定義 (206 維度)
# -----------------------------
class SignRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# -----------------------------
# 2. LLM 翻譯函式 (Ollama)
# -----------------------------
def translate_sign_sequence(word_list):
    if not word_list: return ""
    
    # 移除連續重複的單字 (Debounce)
    unique_words = []
    for w in word_list:
        if not unique_words or w != unique_words[-1]:
            unique_words.append(w)
            
    combined_words = "、".join(unique_words)
    print(f"\n[SYSTEM] 準備翻譯序列: {combined_words}")
    
    url = "http://127.0.0.1:11434/api/generate"
    prompt = f"你是一個手語翻譯專家。請將以下手語單字序列轉換成一句流暢的中文句子：{combined_words}。直接輸出句子，不要任何解釋。"
    
    payload = {
        "model": "gemma3:4b", 
        "prompt": prompt, 
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        result = response.json().get('response', '').strip()
        print(f"[LLM 回傳]: {result}")
        return result
    except Exception as e:
        print(f"[LLM 錯誤]: {e}")
        return "翻譯暫時不可用"

# -----------------------------
# 3. 初始化與載入
# -----------------------------
with open("label_map.json", "r", encoding="utf-8") as f:
    idx2label = {int(k): v for k, v in json.load(f).items()}

INPUT_DIM = 206 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SignRNN(INPUT_DIM, 128, 2, len(idx2label)).to(device)
model.load_state_dict(torch.load("sign_lstm.pth", map_location=device))
model.eval()

mp_hands, mp_face, mp_pose = mp.solutions.hands, mp.solutions.face_mesh, mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=2)
face = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose()

FACE_IDS = [33, 133, 362, 263, 1, 61, 291, 199]
POSE_IDS = [0, 11, 12, 13, 14, 23, 15, 16]

# -----------------------------
# 4. 工具函式
# -----------------------------
def angle_between_points(a, b, c):
    ba, bc = a - b, c - b
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))

def hand_angles(hand_lms):
    fingers = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
    angles = []
    for f in fingers:
        pts = np.array([[hand_lms[i].x, hand_lms[i].y, hand_lms[i].z] for i in f])
        for j in range(1, 4):
            angles.append(angle_between_points(pts[j-1], pts[j], pts[j+1]))
    return angles

def draw_styled_text(img, text, pos=(20, 40), color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 32)
    except: font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# -----------------------------
# 5. 即時推論迴圈
# -----------------------------
cap = cv2.VideoCapture(0)
SEQ_LEN = 30
buffer = []
sentence_words = []
last_word = ""
cooldown_counter = 0
llm_result = ""
display_text = "等待手勢..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- A. Pose (正規化基準) ---
    pres = pose.process(rgb)
    pose_feat, elbow_ang, norm_dist = [], [0, 0], 1.0
    if pres.pose_landmarks:
        lms = pres.pose_landmarks.landmark
        p11, p12 = np.array([lms[11].x, lms[11].y, lms[11].z]), np.array([lms[12].x, lms[12].y, lms[12].z])
        norm_dist = np.linalg.norm(p11 - p12) + 1e-6
        for i in POSE_IDS:
            pose_feat += [lms[i].x / norm_dist, lms[i].y / norm_dist, lms[i].z / norm_dist]
        elbow_ang[0] = angle_between_points(p11, np.array([lms[13].x, lms[13].y, lms[13].z]), np.array([lms[15].x, lms[15].y, lms[15].z]))
        elbow_ang[1] = angle_between_points(p12, np.array([lms[14].x, lms[14].y, lms[14].z]), np.array([lms[16].x, lms[16].y, lms[16].z]))
    else: pose_feat = [0]*24

    # --- B. Face ---
    fres = face.process(rgb)
    face_feat = []
    if fres.multi_face_landmarks:
        for i in FACE_IDS:
            lm = fres.multi_face_landmarks[0].landmark[i]
            face_feat += [lm.x / norm_dist, lm.y / norm_dist, lm.z / norm_dist]
    else: face_feat = [0]*24

    # --- C. Hands ---
    hres = hands.process(rgb)
    hand_feat, finger_ang = [], []
    hand_exist = False
    if hres.multi_hand_landmarks:
        hand_exist = True
        for hand in hres.multi_hand_landmarks[:2]:
            wrist = np.array([hand.landmark[0].x, hand.landmark[0].y, hand.landmark[0].z])
            for lm in hand.landmark:
                hand_feat += ((np.array([lm.x, lm.y, lm.z]) - wrist) / norm_dist).tolist()
            finger_ang += hand_angles(hand.landmark)
    
    while len(hand_feat) < 126: hand_feat += [0,0,0]
    while len(finger_ang) < 30: finger_ang += [0]*15

    # --- D. 拼接與預測 ---
    final_features = face_feat + hand_feat + finger_ang + pose_feat + elbow_ang
    energy = np.sum(np.abs(hand_feat))

    if hand_exist and energy > 0.05:
        cooldown_counter = 0  
        buffer.append(final_features)
        if len(buffer) > SEQ_LEN: buffer.pop(0)
        if len(buffer) == SEQ_LEN:
            x_in = torch.tensor(buffer, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x_in)
                conf, idx = torch.softmax(out, dim=1).max(dim=1)
            
            if conf.item() > 0.8: # 信心門檻
                current_word = idx2label[idx.item()]
                if current_word != last_word:
                    sentence_words.append(current_word)
                    last_word = current_word
                display_text = f"辨識: {current_word} ({conf.item():.2f})"
            else:
                display_text = "..."
    else:
        buffer = []
        cooldown_counter += 1
        if 0 < cooldown_counter < 60:
            display_text = f"完成動作請靜止... ({cooldown_counter}/60)"
        
        if cooldown_counter == 60:
            if sentence_words:
                llm_result = translate_sign_sequence(sentence_words)
                sentence_words = [] 
                last_word = ""
            display_text = "翻譯完成"

    # --- E. 畫面顯示 ---
    frame = draw_styled_text(frame, f"序列: {' '.join(sentence_words)}", pos=(20, 40))
    frame = draw_styled_text(frame, f"LLM: {llm_result}", pos=(20, 90), color=(255, 0, 0))
    frame = draw_styled_text(frame, display_text, pos=(20, 140), color=(255, 255, 0))
    
    cv2.imshow("Sign LSTM 206-Dim LLM", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()