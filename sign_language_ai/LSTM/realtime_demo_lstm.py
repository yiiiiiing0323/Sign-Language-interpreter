import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# 1. 模型定義 (Model Definition)
# -----------------------------
class SignRNN(nn.Module):
    """
    自定義的手語辨識循環神經網路模型（使用 LSTM 架構）
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        # 定義多層 LSTM 網路，batch_first=True 代表輸入資料格式為 (batch, seq_len, feature)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全連接層（Linear），負責將 LSTM 輸出的隱藏狀態轉換為各手語類別的分數（Logits）
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # rnn 輸出格式為 out: (batch, seq_len, hidden_dim)
        out, _ = self.rnn(x)
        # 取最後一個時序（out[:, -1, :]) 的特徵送入全連接層進行分類預測
        return self.fc(out[:, -1, :])

# -----------------------------
# 2. 初始化與載入 (Initialization & Loading)
# -----------------------------
TOTAL_DIM = 218  # 每幀提取的合併特徵總維度
SEQ_LEN = 30     # 預測所需的時序長度（即模型一次看連續 30 幀的動作）

# 載入標籤映射表，將數字索引轉換成對應的中文手語詞彙
with open("label_map.json", "r", encoding="utf-8") as f:
    idx2label = {int(k): v for k, v in json.load(f).items()}

# 檢查是否有可用 GPU (CUDA)，若無則使用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 實例化模型並移至指定運算裝置
model = SignRNN(TOTAL_DIM, 128, 2, len(idx2label)).to(device)
# 載入已訓練完成的 LSTM 模型權重檔案
model.load_state_dict(torch.load("sign_lstm.pth", map_location=device))
# 將模型設定為評估/推論模式（Eval Mode）
model.eval()

# 初始化 MediaPipe 解決方案（雙手、人臉網格、身體骨架）
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# 初始化 MediaPipe 實例，設定最大追蹤手數量與核心信心度閾值
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose(min_detection_confidence=0.7)

# 篩選特定關鍵點，精簡特徵維度（人臉特定 8 個點，身體特定 10 個點）
FACE_IDS = [33, 133, 362, 263, 1, 61, 291, 199]
POSE_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23]

# -----------------------------
# 3. 工具函式 (Helper Functions)
# -----------------------------
def angle_between_points(a, b, c):
    """
    計算三點之間所夾的角度（利用向量夾角公式，回傳弧度值值）
    """
    ba, bc = a - b, c - b
    # 計算分母，加入 1e-6 避免分母為零
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    # 點積除以長度乘積，並使用 clip 將數值限制在 [-1, 1] 範圍內防浮點數誤差出錯
    return np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))

def hand_angles(hand_lms, img_w, img_h):
    """
    計算單手五根手指的所有關節夾角（每手共 15 個角度特徵）
    """
    # 定義五根手指對應的 MediaPipe 節點編號
    fingers = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
    angles = []
    for f in fingers:
        # ✅ 還原成真實像素坐標，將標準化坐標乘上影像寬高 (Z 軸深度一般與 X 軸同等縮放比例處理)
        pts = np.array([
            [hand_lms[i].x * img_w, hand_lms[i].y * img_h, hand_lms[i].z * img_w] 
            for i in f
        ])
        # 針對每根手指計算其中三個鄰近節點組成的關節角度（每根手指 3 個角，共 15 個角）
        for j in range(1, 4):
            angles.append(angle_between_points(pts[j-1], pts[j], pts[j+1]))
    return angles

def draw_chinese_text(img, text, pos=(20, 40)):
    """
    使用 PIL 套件在 OpenCV 的影像畫面上正確繪製中文字串（避免 OpenCV 原生不支援中文的問題）
    """
    # 將 OpenCV 格式（BGR）轉換為 PIL 格式（RGB）
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 嘗試載入系統預設的微軟正黑體，若找不到則退回預設低解析度字體
    try: font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 36)
    except: font = ImageFont.load_default()
    # 於畫面上繪製綠色文字
    draw.text(pos, text, font=font, fill=(0, 255, 0))
    # 將 PIL 格式轉換回 OpenCV 支援的 BGR 陣列
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# -----------------------------
# 4. 即時推論迴圈 (Real-time Inference Loop)
# -----------------------------
cap = cv2.VideoCapture(0) # 開啟編號 0 的內建或預設 USB 攝影機
buffer = []               # 核心時序緩衝區，用來儲存最新連續 30 幀的特徵向量
prev_left_wrist_y = None  # 儲存前一幀左手腕的 Y 座標，用來計算垂直移動速度
prev_right_wrist_y = None # 儲存前一幀右手腕的 Y 座標，用來計算垂直移動速度

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # 水平鏡像翻轉畫面，使其符合人直覺反射動作
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 將影像轉為 RGB 供 MediaPipe 分析
    
    # ✅ 取得目前 Webcam 串流的真實解析度寬高
    img_h, img_w = frame.shape[:2]
    
    # ✅ 定義坐標像素還原轉換函式（確保與訓練時的特徵提取尺度完全對齊）
    def get_px_coord(lm):
        return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w])

    # 初始化預設特徵陣列（當沒偵測到特定部位時，用 0 填充以防模型輸入維度不合）
    pose_feat, face_feat = [0.0]*30, [0.0]*24
    l_hand_feat, r_hand_feat = [0.0]*63, [0.0]*63
    l_finger_ang, r_finger_ang = [0.0]*15, [0.0]*15
    elbow_ang = [0.0, 0.0]
    extra_feats = [0.0] * 6 # 額外速度與位置相對特徵 [右速Y, 右對鼻X, 右對耳X, 左速Y, 左對鼻X, 左對耳X]
    norm_dist = 1.0         # 歸一化基準距離（肩膀寬度），防人站的前後距離遠近影響特徵
    ref_pt = np.zeros(3)    # 基準點（以鼻子為原點）
    
    # 五官 X 軸像素坐標預設值（當偵測不到骨架時預設在畫面正中間）
    nose_x, l_ear_x, r_ear_x = 0.5 * img_w, 0.5 * img_w, 0.5 * img_w 

    # --- A. 骨架偵測 (Pose) ---
    pres = pose.process(rgb)
    if pres.pose_landmarks:
        lms = pres.pose_landmarks.landmark
        ref_pt = get_px_coord(lms[0]) # 以鼻子（INDEX 0）作為全身坐標系的空間原點
        p11, p12 = get_px_coord(lms[11]), get_px_coord(lms[12]) # 取得左右肩坐標
        norm_dist = np.linalg.norm(p11 - p12) + 1e-6            # 計算雙肩寬度作為歸一化比例尺
        
        nose_x = get_px_coord(lms[0])[0]  # 取得精確鼻子像素 X 坐標
        l_ear_x = get_px_coord(lms[7])[0]  # ✅ 取得精確左耳像素 X 坐標
        r_ear_x = get_px_coord(lms[8])[0]  # 取得精確右耳像素 X 坐標
        
        # 收集指定的 10 個核心身體骨架特徵點，並進行相對於鼻子的相對位移與肩寬歸一化
        pose_feat = []
        for i in POSE_IDS:
            pose_feat += ((get_px_coord(lms[i]) - ref_pt) / norm_dist).tolist()
            
        # 額外計算：左右手肘夾角特徵（點 11-13-15 與 12-14-16 的夾角）
        p13, p14 = get_px_coord(lms[13]), get_px_coord(lms[14])
        p15, p16 = get_px_coord(lms[15]), get_px_coord(lms[16])
        elbow_ang = [angle_between_points(p11, p13, p15)/np.pi, angle_between_points(p12, p14, p16)/np.pi]

    # --- B. 人臉網格偵測 (Face) ---
    fres = face.process(rgb)
    if fres.multi_face_landmarks:
        face_feat = []
        # 收集 8 個特定人臉關鍵點，同樣以鼻子為原點進行位移與肩寬歸一化
        for i in FACE_IDS:
            lm = fres.multi_face_landmarks[0].landmark[i]
            face_feat += ((get_px_coord(lm) - ref_pt) / norm_dist).tolist()

    # --- C. 雙手偵測 (Hands) ---
    hres = hands.process(rgb)
    hand_exist = False # 標記目前畫面上是否有出現任何一隻手
    
    if hres.multi_hand_landmarks and hres.multi_handedness:
        hand_exist = True
        for idx, hand_lms in enumerate(hres.multi_hand_landmarks):
            # 取得該手是「左手」還是「右手」
            label = hres.multi_handedness[idx].classification[0].label
            wrist = hand_lms.landmark[0]   # 取得手腕節點
            wrist_pt = get_px_coord(wrist) # 還原成像素坐標
            
            # 手部關節特徵歸一化：所有點減去自己手腕的坐標，轉為相對位移特徵
            feat = []
            for lm in hand_lms.landmark:
                feat += ((get_px_coord(lm) - wrist_pt) / norm_dist).tolist()
                
            # ✅ 計算這隻手本身的 15 個指骨夾角特徵
            angles = hand_angles(hand_lms.landmark, img_w, img_h)
            
            # 根據左右手標籤，將特徵填入各自專屬的空間，並計算垂直運動速度、五官相對距離
            if label == 'Left':
                l_hand_feat, l_finger_ang = feat, angles
                # 計算左手腕相較於上一幀的 Y 軸位移速度
                if prev_left_wrist_y is not None: 
                    extra_feats[3] = (wrist_pt[1] - prev_left_wrist_y) / norm_dist
                prev_left_wrist_y = wrist_pt[1]
                # 計算左手腕與鼻子、左耳的橫向相對距離
                extra_feats[4] = (wrist_pt[0] - nose_x) / norm_dist
                extra_feats[5] = (wrist_pt[0] - l_ear_x) / norm_dist # ✅ 改對齊左耳 l_ear_x
            else:
                r_hand_feat, r_finger_ang = feat, angles
                # 計算右手腕相較於上一幀的 Y 軸位移速度
                if prev_right_wrist_y is not None: 
                    extra_feats[0] = (wrist_pt[1] - prev_right_wrist_y) / norm_dist
                prev_right_wrist_y = wrist_pt[1]
                # 計算右手腕與鼻子、右耳的橫向相對距離
                extra_feats[1] = (wrist_pt[0] - nose_x) / norm_dist
                extra_feats[2] = (wrist_pt[0] - r_ear_x) / norm_dist

    # 安全機制：如果鏡頭內完全沒有任何手，重設歷史手腕位置為 None，避免隔空連線產生錯誤速度速度特徵
    if not hand_exist: prev_left_wrist_y = prev_right_wrist_y = None

    # --- D. 串接特徵與即時模型預測 (Inference) ---
    # 將所有部位抽出來的特徵，依固定順序暴力拼接，長度正好必須為 218 維
    combined = face_feat + l_hand_feat + r_hand_feat + l_finger_ang + r_finger_ang + pose_feat + elbow_ang + extra_feats
    buffer.append(combined) # 將這一幀的超大特徵串加進時序暫存佇列
    
    # ✅ 滑動視窗控制：若長度大於 30 幀，則吐出最舊的一幀（永遠保持最新連續 30 幀）
    if len(buffer) > SEQ_LEN: buffer.pop(0)

    display_text = "請做手勢"
    # 當畫面上至少有一隻手，且收集足夠的 30 幀時序，啟動深度學習預測
    if hand_exist and len(buffer) == SEQ_LEN:
        # 將資料打包成張量 (Tensor)，並擴充 Batch 維度為 1 且送入 GPU/CPU 裝置
        x_in = torch.tensor([buffer], dtype=torch.float32).to(device)
        with torch.no_grad(): # 關閉梯度計算，節省顯存並加速推論速度
            out = model(x_in)
            # 使用 Softmax 將輸出轉為各機率值，並找出機率最高（Max）的類別與其信心度
            conf, idx = torch.softmax(out, dim=1).max(dim=1)
        
        # 信心度閾值設定：大於 70% 才予以承認，並顯示該詞彙與信心度
        if conf.item() > 0.7:
            display_text = f"{idx2label[idx.item()]} ({conf.item():.2f})"
        else:
            display_text = "辨識中..."
    elif not hand_exist:
        buffer = [] # 只要手一離開鏡頭範圍，即刻清空 Buffer，防止殘留特徵殘留導致誤判
        display_text = "請做手勢"

    # 在畫面上印出美化後的中文字幕
    frame = draw_chinese_text(frame, display_text)
    # 開啟名為 "Sign Recognition Real-time" 的 OpenCV 顯示視窗
    cv2.imshow("Sign Recognition Real-time", frame)
    # 偵測鍵盤事件：若使用者按下鍵盤上的 ESC 鍵（ASCII 碼 27），跳出無窮迴圈
    if cv2.waitKey(1) & 0xFF == 27: break

# 釋放攝影機硬體資源、關閉所有即時跳出的 OpenCV 視窗
cap.release()
cv2.destroyAllWindows()
