import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# 1. Model 定義 (必須與 218 維訓練端一致)
# -----------------------------
class SignTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.d_model = 256
        self.embedding = nn.Linear(input_dim, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# -----------------------------
# 2. 初始化與載入
# -----------------------------
INPUT_DIM = 218  # ⭐ 修正為 218 維
SEQ_LEN = 30

with open("label_map.json", "r", encoding="utf-8") as f:
    idx2label = {int(k): v for k, v in json.load(f).items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SignTransformer(INPUT_DIM, len(idx2label)).to(device)
model.load_state_dict(torch.load("sign_transformer.pth", map_location=device))
model.eval()

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose(min_detection_confidence=0.7)

# 與特徵擷取端嚴格對齊
FACE_IDS = [33, 133, 362, 263, 1, 61, 291, 199]
POSE_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23] # ⭐ 修正順序

# -----------------------------
# 3. 工具函式
# -----------------------------
def angle_between_points(a, b, c):
    ba, bc = a - b, c - b
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))

# ✅ 修正：加入 img_w, img_h 參數，轉換為像素空間計算物理角度
def hand_angles(hand_lms, img_w, img_h):
    fingers = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
    angles = []
    for f in fingers:
        pts = np.array([
            [hand_lms[i].x * img_w, hand_lms[i].y * img_h, hand_lms[i].z * img_w] 
            for i in f
        ])
        for j in range(1, 4):
            angles.append(angle_between_points(pts[j-1], pts[j], pts[j+1]))
    return angles

def draw_chinese_text(img, text, pos=(20, 40)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 36)
    except: font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# -----------------------------
# 4. 即時推論迴圈
# -----------------------------
cap = cv2.VideoCapture(0)
buffer = []
prev_l_wrist_y = None
prev_r_wrist_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ✅ 新增：取得 Webcam 畫面的真實寬高
    img_h, img_w = frame.shape[:2]
    
    # ✅ 新增：定義像素轉換工具
    def get_px_coord(lm):
        return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w])
    
    # 特徵初始化 (歸零)
    pose_feat, face_feat = [0.0]*30, [0.0]*24
    l_hand_f, r_hand_f = [0.0]*63, [0.0]*63
    l_hand_a, r_hand_a = [0.0]*15, [0.0]*15
    elbow_ang = [0.0, 0.0]
    extra_feats = [0.0] * 6
    
    norm_dist = 1.0
    ref_pt = np.zeros(3)
    nose_x, l_ear_x, r_ear_x = 0.5 * img_w, 0.5 * img_w, 0.5 * img_w # 改為像素預設值

    # --- A. Pose ---
    p_res = pose.process(rgb)
    if p_res.pose_landmarks:
        lms = p_res.pose_landmarks.landmark
        ref_pt = get_px_coord(lms[0]) # 鼻子基準
        p11, p12 = get_px_coord(lms[11]), get_px_coord(lms[12])
        norm_dist = np.linalg.norm(p11 - p12) + 1e-6 # 肩寬正規化
        
        nose_x = get_px_coord(lms[0])[0]
        l_ear_x = get_px_coord(lms[7])[0] # 加入左耳
        r_ear_x = get_px_coord(lms[8])[0]
        
        pose_feat = []
        for i in POSE_IDS:
            c = (get_px_coord(lms[i]) - ref_pt) / norm_dist
            pose_feat += c.tolist()
        
        # 手肘角
        p13, p14 = get_px_coord(lms[13]), get_px_coord(lms[14])
        p15, p16 = get_px_coord(lms[15]), get_px_coord(lms[16])
        elbow_ang = [angle_between_points(p11, p13, p15)/np.pi, angle_between_points(p12, p14, p16)/np.pi]

    # --- B. Face ---
    f_res = face.process(rgb)
    if f_res.multi_face_landmarks:
        face_feat = []
        for i in FACE_IDS:
            lm = f_res.multi_face_landmarks[0].landmark[i]
            face_feat += ((get_px_coord(lm) - ref_pt) / norm_dist).tolist()

    # --- C. Hands ---
    h_res = hands.process(rgb)
    hand_exist = False
    if h_res.multi_hand_landmarks and h_res.multi_handedness:
        hand_exist = True
        for idx, h_lms in enumerate(h_res.multi_hand_landmarks):
            label = h_res.multi_handedness[idx].classification[0].label
            wrist = h_lms.landmark[0]
            wrist_pt = get_px_coord(wrist)
            
            feat = []
            for lm in h_lms.landmark:
                c = (get_px_coord(lm) - wrist_pt) / norm_dist
                feat += c.tolist()
            
            # ✅ 傳入 img_w, img_h
            angles = hand_angles(h_lms.landmark, img_w, img_h)
            
            if label == 'Left':
                l_hand_f, l_hand_a = feat, angles
                if prev_l_wrist_y is not None:
                    extra_feats[3] = (wrist_pt[1] - prev_l_wrist_y) / norm_dist 
                prev_l_wrist_y = wrist_pt[1]
                extra_feats[4] = (wrist_pt[0] - nose_x) / norm_dist 
                extra_feats[5] = (wrist_pt[0] - l_ear_x) / norm_dist # ✅ 修正為對齊左耳
            else:
                r_hand_f, r_hand_a = feat, angles
                if prev_r_wrist_y is not None:
                    extra_feats[0] = (wrist_pt[1] - prev_r_wrist_y) / norm_dist 
                prev_r_wrist_y = wrist_pt[1]
                extra_feats[1] = (wrist_pt[0] - nose_x) / norm_dist 
                extra_feats[2] = (wrist_pt[0] - r_ear_x) / norm_dist 

    if not hand_exist: prev_l_wrist_y = prev_r_wrist_y = None

    # --- D. 拼接 (218維) ---
    combined = face_feat + l_hand_f + r_hand_f + l_hand_a + r_hand_a + pose_feat + elbow_ang + extra_feats
    
    # 數值限幅增加穩定性
    final_features = np.clip(combined, -5.0, 5.0).tolist()
    assert len(final_features) == INPUT_DIM, f"維度錯誤: {len(final_features)}"

    # --- E. 預測邏輯 ---
    display_text = "請做手勢"
    if hand_exist:
        buffer.append(final_features)
        if len(buffer) > SEQ_LEN: buffer.pop(0)
        
        if len(buffer) == SEQ_LEN:
            x_in = torch.tensor([buffer], dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(x_in)
                conf, idx = torch.softmax(out, dim=1).max(dim=1)
            
            if conf.item() > 0.7:
                display_text = f"{idx2label[idx.item()]} ({conf.item():.2f})"
            else:
                display_text = "辨識中..."
    else:
        buffer, display_text = [], "請做手勢"

    frame = draw_chinese_text(frame, display_text)
    cv2.imshow("Sign Transformer 218-Dim (Precision Alignment)", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()