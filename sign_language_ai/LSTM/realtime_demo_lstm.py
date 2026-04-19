import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# 1. 模型定義
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
# 2. 初始化與載入
# -----------------------------
TOTAL_DIM = 218  
SEQ_LEN = 30     

with open("label_map.json", "r", encoding="utf-8") as f:
    idx2label = {int(k): v for k, v in json.load(f).items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SignRNN(TOTAL_DIM, 128, 2, len(idx2label)).to(device)
model.load_state_dict(torch.load("sign_lstm.pth", map_location=device))
model.eval()

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose(min_detection_confidence=0.7)

FACE_IDS = [33, 133, 362, 263, 1, 61, 291, 199]
POSE_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23]

# -----------------------------
# 3. 工具函式
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
prev_left_wrist_y = None
prev_right_wrist_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pose_feat, face_feat = [0.0]*30, [0.0]*24
    l_hand_feat, r_hand_feat = [0.0]*63, [0.0]*63
    l_finger_ang, r_finger_ang = [0.0]*15, [0.0]*15
    elbow_ang = [0.0, 0.0]
    extra_feats = [0.0] * 6
    norm_dist = 1.0
    ref_pt = np.zeros(3)
    nose_x, r_ear_x = 0.5, 0.5

    # --- A. Pose ---
    pres = pose.process(rgb)
    if pres.pose_landmarks:
        lms = pres.pose_landmarks.landmark
        ref_pt = np.array([lms[0].x, lms[0].y, lms[0].z])
        p11, p12 = np.array([lms[11].x, lms[11].y, lms[11].z]), np.array([lms[12].x, lms[12].y, lms[12].z])
        norm_dist = np.linalg.norm(p11 - p12) + 1e-6
        nose_x, r_ear_x = lms[0].x, lms[8].x
        pose_feat = []
        for i in POSE_IDS:
            pose_feat += ((np.array([lms[i].x, lms[i].y, lms[i].z]) - ref_pt) / norm_dist).tolist()
        p13, p14, p15, p16 = np.array([lms[13].x, lms[13].y, lms[13].z]), np.array([lms[14].x, lms[14].y, lms[14].z]), \
                             np.array([lms[15].x, lms[15].y, lms[15].z]), np.array([lms[16].x, lms[16].y, lms[16].z])
        elbow_ang = [angle_between_points(p11, p13, p15)/np.pi, angle_between_points(p12, p14, p16)/np.pi]

    # --- B. Face ---
    fres = face.process(rgb)
    if fres.multi_face_landmarks:
        face_feat = []
        for i in FACE_IDS:
            lm = fres.multi_face_landmarks[0].landmark[i]
            face_feat += ((np.array([lm.x, lm.y, lm.z]) - ref_pt) / norm_dist).tolist()

    # --- C. Hands ---
    hres = hands.process(rgb)
    hand_exist = False
    if hres.multi_hand_landmarks and hres.multi_handedness:
        hand_exist = True
        for idx, hand_lms in enumerate(hres.multi_hand_landmarks):
            label = hres.multi_handedness[idx].classification[0].label
            wrist = hand_lms.landmark[0]
            feat = []
            for lm in hand_lms.landmark:
                feat += [(lm.x - wrist.x)/norm_dist, (lm.y - wrist.y)/norm_dist, (lm.z - wrist.z)/norm_dist]
            angles = hand_angles(hand_lms.landmark)
            if label == 'Left':
                l_hand_feat, l_finger_ang = feat, angles
                if prev_left_wrist_y is not None: extra_feats[3] = (wrist.y - prev_left_wrist_y) / norm_dist
                prev_left_wrist_y = wrist.y
                extra_feats[4], extra_feats[5] = (wrist.x - nose_x)/norm_dist, (wrist.x - r_ear_x)/norm_dist
            else:
                r_hand_feat, r_finger_ang = feat, angles
                if prev_right_wrist_y is not None: extra_feats[0] = (wrist.y - prev_right_wrist_y) / norm_dist
                prev_right_wrist_y = wrist.y
                extra_feats[1], extra_feats[2] = (wrist.x - nose_x)/norm_dist, (wrist.x - r_ear_x)/norm_dist

    if not hand_exist: prev_left_wrist_y = prev_right_wrist_y = None

    # --- D. 預測 ---
    combined = face_feat + l_hand_feat + r_hand_feat + l_finger_ang + r_finger_ang + pose_feat + elbow_ang + extra_feats
    buffer.append(combined)
    if len(buffer) > SEQ_LEN: buffer.pop(0)

    display_text = "請做手勢"
    if hand_exist and len(buffer) == SEQ_LEN:
        x_in = torch.tensor([buffer], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x_in)
            conf, idx = torch.softmax(out, dim=1).max(dim=1)
        
        if conf.item() > 0.7:
            display_text = f"{idx2label[idx.item()]} ({conf.item():.2f})"
        else:
            display_text = "辨識中..."
    elif not hand_exist:
        buffer = []
        display_text = "請做手勢"

    frame = draw_chinese_text(frame, display_text)
    cv2.imshow("Sign Recognition Real-time", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()