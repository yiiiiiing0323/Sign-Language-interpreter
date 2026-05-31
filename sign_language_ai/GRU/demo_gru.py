import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# 1. 模型定義 (218維)
# -----------------------------
class SignGRU(nn.Module):
    """
    自定義的手語辨識循環神經網路模型（使用 GRU 架構）
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        # 定義多層 GRU 網路，batch_first=True 代表輸入資料格式為 (batch, seq_len, feature)
        # GRU 結構較 LSTM 精簡，運算速度通常較快
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全連接線性輸出層，將 GRU 的隱藏狀態映射到手語詞彙的分類分數
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x 的輸入形狀為 (Batch_Size, 30, 218)
        # out 的輸出形狀為 (Batch_Size, 30, hidden_dim)
        out, _ = self.gru(x)
        # 提取時間步最末尾（第 30 幀）的隱藏狀態特徵送入全連接層進行分類
        return self.fc(out[:, -1, :])

# -----------------------------
# 2. 初始化
# -----------------------------
TOTAL_DIM = 218  # 每幀提取的合併特徵總維度
SEQ_LEN = 30     # 預測所需的時序時間長度（固定為 30 幀）

# 載入標籤映射表，將數字索引還原成中文手語字串
with open("label_map.json", "r", encoding="utf-8") as f:
    idx2label = {int(k): v for k, v in json.load(f).items()}

# 硬體加速偵測，優先使用 GPU (CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 實例化 GRU 模型並移至指定硬體裝置
model = SignGRU(TOTAL_DIM, 128, 2, len(idx2label)).to(device)
# 載入預先訓練完成的 GRU 模型權重檔案
model.load_state_dict(torch.load("sign_gru.pth", map_location=device))
# 切換模型至評估推論模式
model.eval()

# 初始化 MediaPipe 解決方案組件
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# 實例化偵測器
hands = mp_hands.Hands(max_num_hands=2)
face = mp_face.FaceMesh()
pose = mp_pose.Pose()

# 定義人臉與骨架特徵篩選的關鍵點 ID（保持與 218 維訓練規格完全一致）
FACE_IDS = [33, 133, 362, 263, 1, 61, 291, 199]
POSE_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23]

# -----------------------------
# 3. 工具函式
# -----------------------------
def angle_between_points(a, b, c):
    """
    計算三點在空間中產生的夾角（弧度值）
    """
    ba, bc = a - b, c - b
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6) # 防除以零
    return np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))

def hand_angles(hand_lms, img_w, img_h):
    """
    ✅ 修正：加入 img_w, img_h，確保角度在像素物理空間中計算
    避免標準化比例坐標直接計算時，因影片長寬比非 1:1 導致角度扭曲變形
    """
    fingers = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
    angles = []
    for f in fingers:
        # 將節點轉換乘影像實際寬高，還原成真實像素空間坐標
        pts = np.array([
            [hand_lms[i].x * img_w, hand_lms[i].y * img_h, hand_lms[i].z * img_w] 
            for i in f
        ])
        # 每根手指計算 3 個相鄰關節的夾角
        for j in range(1, 4):
            angles.append(angle_between_points(pts[j-1], pts[j], pts[j+1]))
    return angles

def draw_chinese_text(img, text, pos=(20, 40)):
    """
    轉換為 PIL 格式在影像上繪製流暢的中文字幕，再轉回 OpenCV 格式格式
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 36) # 載入微軟正黑體
    except: font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(0, 255, 0)) # 繪製綠色文字
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# -----------------------------
# 4. 主迴圈
# -----------------------------
cap = cv2.VideoCapture(0) # 開啟預設攝影機
buffer = []               # 時序特徵滑動視窗緩衝佇列
prev_left_wrist_y = None  # 上一影格左手腕 Y 軸坐標
prev_right_wrist_y = None # 上一影格右手腕 Y 軸坐標

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # 畫面水平鏡像翻轉
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 轉為 RGB 供偵測器處理
    
    # ✅ 取得畫面真實解析度高與寬
    img_h, img_w = frame.shape[:2]
    
    # ✅ 轉換為真實像素座標的內部封裝小工具
    def get_px_coord(lm):
        return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w])

    # 特徵容器初始化 (當某部位消失於畫面時自動補零)
    pose_feat, face_feat = [0.0]*30, [0.0]*24
    l_hand_feat, r_hand_feat = [0.0]*63, [0.0]*63
    l_finger_ang, r_finger_ang = [0.0]*15, [0.0]*15
    elbow_ang = [0.0, 0.0]
    extra_feats = [0.0] * 6 # 包含手腕速度、與鼻子及左右耳水平像素距離
    
    norm_dist = 1.0
    ref_pt = np.zeros(3)
    # ✅ 改用像素概念的預設初始值，並加入左耳與右耳的空間坐標
    nose_x, l_ear_x, r_ear_x = 0.5 * img_w, 0.5 * img_w, 0.5 * img_w

    # --- A. Pose 身體骨架特徵分析 ---
    pres = pose.process(rgb)
    if pres.pose_landmarks:
        lms = pres.pose_landmarks.landmark
        ref_pt = get_px_coord(lms[0]) # 設定鼻子為全身空間原點
        p11, p12 = get_px_coord(lms[11]), get_px_coord(lms[12]) # 左右肩
        norm_dist = np.linalg.norm(p11 - p12) + 1e-6            # 雙肩寬度作為歸一化比例尺
        
        nose_x = get_px_coord(lms[0])[0]
        l_ear_x = get_px_coord(lms[7])[0] # 左耳像素 X 軸位置
        r_ear_x = get_px_coord(lms[8])[0] # 右耳像素 X 軸位置
        
        pose_feat = []
        for i in POSE_IDS:
            # 相對位移計算並排除站姿前後距離影響（除以肩寬）
            pose_feat += ((get_px_coord(lms[i]) - ref_pt) / norm_dist).tolist()
            
        # 計算手肘關節角度特徵
        p13, p14 = get_px_coord(lms[13]), get_px_coord(lms[14])
        p15, p16 = get_px_coord(lms[15]), get_px_coord(lms[16])
        elbow_ang = [angle_between_points(p11, p13, p15)/np.pi, angle_between_points(p12, p14, p16)/np.pi]

    # --- B. Face 人臉網格特徵分析 ---
    fres = face.process(rgb)
    if fres.multi_face_landmarks:
        face_feat = []
        for i in FACE_IDS:
            lm = fres.multi_face_landmarks[0].landmark[i]
            # 相對於鼻子原點並除以肩寬進行歸一化
            face_feat += ((get_px_coord(lm) - ref_pt) / norm_dist).tolist()

    # --- C. Hands 雙手特徵分析與相對度量 ---
    hres = hands.process(rgb)
    hand_exist = False # 標記當前畫面是否有偵測到手部
    if hres.multi_hand_landmarks and hres.multi_handedness:
        hand_exist = True
        for idx, hand_lms in enumerate(hres.multi_hand_landmarks):
            label = hres.multi_handedness[idx].classification[0].label # 判斷左或右手
            wrist = hand_lms.landmark[0]
            wrist_pt = get_px_coord(wrist)
            
            # 手部自身特徵：所有點減去手腕像素坐標，轉為手內部的相對空間特徵特徵
            feat = []
            for lm in hand_lms.landmark:
                feat += ((get_px_coord(lm) - wrist_pt) / norm_dist).tolist()
                
            # 計算該手 15 維的手指夾角
            angles = hand_angles(hand_lms.landmark, img_w, img_h)
            
            if label == 'Left':
                l_hand_feat, l_finger_ang = feat, angles
                # 計算左手垂直瞬時速度特徵
                if prev_left_wrist_y is not None: 
                    extra_feats[3] = (wrist_pt[1] - prev_left_wrist_y) / norm_dist
                prev_left_wrist_y = wrist_pt[1]
                # 計算左手與鼻子、左耳的橫向相對像素位置差距
                extra_feats[4] = (wrist_pt[0] - nose_x) / norm_dist
                extra_feats[5] = (wrist_pt[0] - l_ear_x) / norm_dist # ✅ 對齊左耳
            else:
                r_hand_feat, r_finger_ang = feat, angles
                # 計算右手垂直瞬時速度特徵
                if prev_right_wrist_y is not None: 
                    extra_feats[0] = (wrist_pt[1] - prev_right_wrist_y) / norm_dist
                prev_right_wrist_y = wrist_pt[1]
                # 計算右手與鼻子、右耳的橫向相對像素位置差距
                extra_feats[1] = (wrist_pt[0] - nose_x) / norm_dist
                extra_feats[2] = (wrist_pt[0] - r_ear_x) / norm_dist # ✅ 對齊右耳

    # 若手不見了，即刻重設速度基準點，避免下一秒手突然出現時產生極大速度爆發現象
    if not hand_exist: prev_left_wrist_y = prev_right_wrist_y = None

    # --- D. 拼接 (嚴格按照 218 維順序) ---
    combined = face_feat + l_hand_feat + r_hand_feat + l_finger_ang + r_finger_ang + pose_feat + elbow_ang + extra_feats
    
    # ✅ 數值防呆限幅：將合併後的所有數值強行截斷在 [-5.0, 5.0] 區間之內
    # 目的在於消除 MediaPipe 偶爾產生的極端噪點或節點大範圍抖動，避免模型預測崩潰
    final_features = np.clip(combined, -5.0, 5.0).tolist()
    
    # 送入滑動視窗緩衝區
    buffer.append(final_features)
    # 確保緩衝區永遠只留存最新的連續 30 幀特徵
    if len(buffer) > SEQ_LEN: buffer.pop(0)

    display_text = "請做手勢"
    # 當手存在且緩衝區集滿 30 幀，啟動深度學習 GRU 模型預測
    if hand_exist and len(buffer) == SEQ_LEN:
        # 包裝成 PyTorch Tensor 並擴充 Batch 維度且移動到運算晶片 (GPU/CPU)
        x_in = torch.tensor([buffer], dtype=torch.float32).to(device)
        with torch.no_grad(): # 關閉梯度，加速推論
            out = model(x_in)
            # 使用 Softmax 函數計算類別機率，max(dim=1) 取出最高信心度與對應類別編號
            conf, idx = torch.softmax(out, dim=1).max(dim=1)
        
        # 信心度大於 70% 則印出辨識成功的中文詞彙，否則顯示等待辨識狀態
        if conf.item() > 0.7:
            display_text = f"{idx2label[idx.item()]} ({conf.item():.2f})"
        else:
            display_text = "..."
    elif not hand_exist:
        buffer = [] # 手若完全離開畫面，立刻清空緩衝區，杜絕前一動作特徵殘留造成的誤判
        display_text = "請做手勢"

    # 在影像上繪製中文字幕
    frame = draw_chinese_text(frame, display_text)
    # 顯示即時視窗
    cv2.imshow("Sign Language GRU 218-Dim", frame)
    # 偵測 ESC 鍵 (ASCII 27)，按下則退出無窮迴圈
    if cv2.waitKey(1) & 0xFF == 27: break

# 釋放攝影機、銷毀所有視窗資源
cap.release()
cv2.destroyAllWindows()
