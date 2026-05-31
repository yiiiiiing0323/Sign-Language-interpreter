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
    """
    自定義的手語辨識 Transformer 模型，利用自注意力機制擷取時序動作特徵
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.d_model = 256  # Transformer 內部的特徵維度（隱藏層寬度）
        
        # 1. 嵌入層（Embedding）：將 218 維的原始輸入特徵，線性映射升維到 256 維
        self.embedding = nn.Linear(input_dim, self.d_model)
        
        # 2. 位置編碼（Positional Embedding）：Transformer 本身不具備時序觀念，
        # 必須加入一個可學習的參數矩陣 (1, 30幀, 256維) 來賦予各個時間步的位置資訊
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, self.d_model))
        
        # 3. 建立 Transformer 編碼器層：設定多頭注意力機制頭數（nhead）為 8，並設定 batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        # 堆疊 3 層 Transformer 編碼器
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 4. 分類層：將編碼後的 256 維隱藏特徵映射到手語總類別數
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x 的輸入形狀為 (Batch_Size, 30, 218)
        x = self.embedding(x)          # 升維至 (Batch_Size, 30, 256)
        x = x + self.pos_embedding     # 加上時序位置編碼資訊
        x = self.encoder(x)            # 送入自注意力編碼器計算，形狀維持 (Batch_Size, 30, 256)
        x = x.mean(dim=1)              # 時間步平均池化（Mean Pooling）：將 30 幀的維度壓縮平均成 1 幀特徵
        return self.classifier(x)      # 送入全連接分類層計算類別 Logits 分數

# -----------------------------
# 2. 初始化與載入
# -----------------------------
INPUT_DIM = 218  # ⭐ 修正為與 A 流資料處理流程完全對齊的 218 維
SEQ_LEN = 30     # 預測所需的時序滑動視窗長度（30 幀）

# 載入標籤映射表，將數字索引還原成手語中文詞彙
with open("label_map.json", "r", encoding="utf-8") as f:
    idx2label = {int(k): v for k, v in json.load(f).items()}

# 自動偵測 CUDA 顯示卡加速硬體
device = "cuda" if torch.cuda.is_available() else "cpu"

# 實例化 Transformer 模型並部署至對應運算裝置
model = SignTransformer(INPUT_DIM, len(idx2label)).to(device)
# 載入訓練完成的 Transformer 狀態權重字典
model.load_state_dict(torch.load("sign_transformer.pth", map_location=device))
# 切換模型至評估推論模式（Eval Mode）
model.eval()

# 初始化 MediaPipe 各視覺偵測解決方案
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# 建立具備高度核心信心度閾值的偵測器實例
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose(min_detection_confidence=0.7)

# 與特徵擷取端、資料增強腳本嚴格對齊的關鍵點選取 ID 清單
FACE_IDS = [33, 133, 362, 263, 1, 61, 291, 199]
POSE_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23] # ⭐ 修正並確保身體節點提取順序

# -----------------------------
# 3. 工具函式
# -----------------------------
def angle_between_points(a, b, c):
    """
    計算三點在空間中夾角（弧度值）
    """
    ba, bc = a - b, c - b
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6) # 加微小值防除以零
    return np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))

def hand_angles(hand_lms, img_w, img_h):
    """
    ✅ 修正：加入 img_w, img_h 參數，轉換為像素空間計算物理角度
    藉此還原被標準化等比例縮放的深度資訊，計算出更準確的 15 維手指夾角特徵
    """
    fingers = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
    angles = []
    for f in fingers:
        # 將標準化浮點比例尺度轉換為真實像素坐標空間
        pts = np.array([
            [hand_lms[i].x * img_w, hand_lms[i].y * img_h, hand_lms[i].z * img_w] 
            for i in f
        ])
        for j in range(1, 4):
            angles.append(angle_between_points(pts[j-1], pts[j], pts[j+1]))
    return angles

def draw_chinese_text(img, text, pos=(20, 40)):
    """
    利用 PIL 橋接機制在 OpenCV BGR 畫面上正確繪製抗鋸齒的中文手語預測字幕
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 36) # 微軟正黑體
    except: font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(0, 255, 0)) # 渲染綠色文字
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# -----------------------------
# 4. 即時推論迴圈
# -----------------------------
cap = cv2.VideoCapture(0) # 驅動本機端 Webcam 攝影機
buffer = []               # 時序特徵滑動視窗緩衝器
prev_l_wrist_y = None     # 儲存上一幀左手腕位置用以計算速度
prev_r_wrist_y = None     # 儲存上一幀右手腕位置用以計算速度

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # 畫面鏡像翻轉，對齊使用者左右手鏡像感
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 將影像轉為 RGB 供 MediaPipe 分析
    
    # ✅ 新增：取得 Webcam 畫面的真實高、寬解析度
    img_h, img_w = frame.shape[:2]
    
    # ✅ 新增：定義將標準化節點轉換為真實像素座標的封裝工具
    def get_px_coord(lm):
        return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w])
    
    # 每影格特徵容器初始化 (若偵測不到該部位時自動填充 0.0)
    pose_feat, face_feat = [0.0]*30, [0.0]*24
    l_hand_f, r_hand_f = [0.0]*63, [0.0]*63
    l_hand_a, r_hand_a = [0.0]*15, [0.0]*15
    elbow_ang = [0.0, 0.0]
    extra_feats = [0.0] * 6 # 移動速度與相對位置距離特徵
    
    norm_dist = 1.0
    ref_pt = np.zeros(3)
    # 預設橫向中心參考點（改為像素預設值，並加入左耳、右耳空間坐標）
    nose_x, l_ear_x, r_ear_x = 0.5 * img_w, 0.5 * img_w, 0.5 * img_w 

    # --- A. Pose 身體骨架偵測 ---
    p_res = pose.process(rgb)
    if p_res.pose_landmarks:
        lms = p_res.pose_landmarks.landmark
        ref_pt = get_px_coord(lms[0]) # 以鼻子為坐標系原點
        p11, p12 = get_px_coord(lms[11]), get_px_coord(lms[12]) # 左右肩膀點
        norm_dist = np.linalg.norm(p11 - p12) + 1e-6            # 肩寬比例尺（排除前後距離干擾）
        
        nose_x = get_px_coord(lms[0])[0]
        l_ear_x = get_px_coord(lms[7])[0] # ✅ 擷取左耳像素坐標
        r_ear_x = get_px_coord(lms[8])[0] # 擷取右耳像素坐標
        
        pose_feat = []
        for i in POSE_IDS:
            # 相對於鼻子並除以肩寬進行尺度歸一化
            c = (get_px_coord(lms[i]) - ref_pt) / norm_dist
            pose_feat += c.tolist()
        
        # 計算手肘關節角度特徵
        p13, p14 = get_px_coord(lms[13]), get_px_coord(lms[14])
        p15, p16 = get_px_coord(lms[15]), get_px_coord(lms[16])
        elbow_ang = [angle_between_points(p11, p13, p15)/np.pi, angle_between_points(p12, p14, p16)/np.pi]

    # --- B. Face 人臉特徵點偵測 ---
    f_res = face.process(rgb)
    if f_res.multi_face_landmarks:
        face_feat = []
        for i in FACE_IDS:
            lm = f_res.multi_face_landmarks[0].landmark[i]
            # 相對於鼻子原點進行位移與歸一化
            face_feat += ((get_px_coord(lm) - ref_pt) / norm_dist).tolist()

    # --- C. Hands 雙手特徵與時序相對量度 ---
    h_res = hands.process(rgb)
    hand_exist = False # 標記畫面上此影格是否存在手部動作
    if h_res.multi_hand_landmarks and h_res.multi_handedness:
        hand_exist = True
        for idx, h_lms in enumerate(h_res.multi_hand_landmarks):
            label = h_res.multi_handedness[idx].classification[0].label # 取得左/右手標籤
            wrist = h_lms.landmark[0]
            wrist_pt = get_px_coord(wrist)
            
            # 手部自身特徵：所有點減去自己手腕，轉為手部的相對局部空間特徵
            feat = []
            for lm in h_lms.landmark:
                c = (get_px_coord(lm) - wrist_pt) / norm_dist
                feat += c.tolist()
            
            # ✅ 傳入即時 Webcam 影像寬高計算出精準的物理關節角度
            angles = hand_angles(h_lms.landmark, img_w, img_h)
            
            if label == 'Left':
                l_hand_f, l_hand_a = feat, angles
                # 計算左手腕相較於上一影格的垂直速度位移
                if prev_l_wrist_y is not None:
                    extra_feats[3] = (wrist_pt[1] - prev_l_wrist_y) / norm_dist 
                prev_l_wrist_y = wrist_pt[1]
                # 計算左手腕與鼻子、左耳的橫向相對像素位移
                extra_feats[4] = (wrist_pt[0] - nose_x) / norm_dist 
                extra_feats[5] = (wrist_pt[0] - l_ear_x) / norm_dist # ✅ 修正為精確對齊左耳
            else:
                r_hand_f, r_hand_a = feat, angles
                # 計算右手腕相較於上一影格的垂直速度位移
                if prev_r_wrist_y is not None:
                    extra_feats[0] = (wrist_pt[1] - prev_r_wrist_y) / norm_dist 
                prev_r_wrist_y = wrist_pt[1]
                # 計算右手腕與鼻子、右耳的橫向相對像素位移
                extra_feats[1] = (wrist_pt[0] - nose_x) / norm_dist 
                extra_feats[2] = (wrist_pt[0] - r_ear_x) / norm_dist 

    # 斷手安全重置機制：若手部在鏡頭中消失，重置速度追蹤點，避免大範圍空降連線誤差
    if not hand_exist: prev_l_wrist_y = prev_r_wrist_y = None

    # --- D. 拼接 (218維) ---
    # 嚴格遵循訓練時的特徵拼接順序
    combined = face_feat + l_hand_f + r_hand_f + l_hand_a + r_hand_a + pose_feat + elbow_ang + extra_feats
    
    # 數值限幅增加穩定性：利用 np.clip 將特徵限制在 [-5.0, 5.0] 區間
    # 目的在於消除極端抖動噪點，避免注意力機制產生不穩定的突變權重值
    final_features = np.clip(combined, -5.0, 5.0).tolist()
    # 嚴格查核特徵總維度是否正確
    assert len(final_features) == INPUT_DIM, f"維度錯誤: {len(final_features)}"

    # --- E. 預測邏輯 ---
    display_text = "請做手勢"
    if hand_exist:
        # 將特徵加入視窗佇列
        buffer.append(final_features)
        # 滾動視窗：永遠拋棄舊影格，保持最新 30 幀特徵
        if len(buffer) > SEQ_LEN: buffer.pop(0)
        
        # 當滑動視窗集滿 30 幀特徵，發動 Transformer 模型前向推論
        if len(buffer) == SEQ_LEN:
            # 打包張量並添加 Batch 維度送入指定加速裝置
            x_in = torch.tensor([buffer], dtype=torch.float32).to(device)
            with torch.no_grad(): # 關閉梯度，加速推論
                out = model(x_in)
                # 使用 Softmax 轉換為機率，並取出信心度最高的類別編號
                conf, idx = torch.softmax(out, dim=1).max(dim=1)
            
            # 設定信心度高於 70% 門檻才判定成功並印出字串
            if conf.item() > 0.7:
                display_text = f"{idx2label[idx.item()]} ({conf.item():.2f})"
            else:
                display_text = "辨識中..."
    else:
        # 手一旦完全完全離開視窗，即刻清空時序緩衝區，杜絕前一姿勢殘留特徵干擾下一次手勢判定
        buffer, display_text = [], "請做手勢"

    # 在畫面上渲染預測中文字幕
    frame = draw_chinese_text(frame, display_text)
    # 跳出 OpenCV 即時推論視窗
    cv2.imshow("Sign Transformer 218-Dim (Precision Alignment)", frame)
    # 監聽鍵盤事件，按下 ESC 鍵（ASCII 碼 27）即退出推論迴圈
    if cv2.waitKey(1) & 0xFF == 27: break

# 釋放攝影機硬體、銷毀所有視窗
cap.release()
cv2.destroyAllWindows()
