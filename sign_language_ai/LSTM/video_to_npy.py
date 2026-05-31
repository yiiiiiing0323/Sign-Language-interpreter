import cv2
import mediapipe as mp
import numpy as np
import os

# 定義原始影片目錄與輸出特徵資料夾路徑
VIDEO_DIR = "videos"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True) # 建立輸出目錄，若已存在則跳過

# 初始化 MediaPipe 各個追蹤模組（雙手、人臉網格、身體骨架）
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# 建立偵測器實例
hands = mp_hands.Hands(max_num_hands=2)
face = mp_face.FaceMesh()
pose = mp_pose.Pose()

# 定義人臉與身體骨架要保留的特定節點 ID，用以精簡特徵長度
FACE_IDS = [33, 133, 362, 263, 1, 61, 291, 199]
POSE_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23]

# 核心常數定義
TOTAL_DIM = 218        # 合併後的總特徵維度 (人臉 24 + 雙手特徵 126 + 雙手角度 30 + 骨架 30 + 手肘角 2 + 速度相對 6)
SEQ_LEN = 30           # 每個時序片段固定的幀數長度
N_WINDOWS = 3          # 每支影片要切出的滑動視窗片段數量
ENERGY_THRESHOLD = 0.5 # 動態端點偵測的能量門檻（用來判斷手是否有在動）

# ── 💡 資料增強方法 (Data Augmentation Methods) ──────────────────────────────
def augment_noise(seq, scale=0.01):
    """
    為時序特徵矩陣加入微小的隨機高斯雜訊，模擬攝影機雜訊或手震情況
    """
    return seq + np.random.normal(0, scale, seq.shape).astype(np.float32)

def augment_feature_scale(seq, scale_range=(0.9, 1.1)):
    """
    將時序特徵進行隨機比例縮放，模擬不同身材或動作放大縮小的變化
    """
    scale = np.random.uniform(*scale_range) # 在範圍內隨機抽樣出一個縮放倍數
    return (seq * scale).astype(np.float32)

# ── 🛠️ 特徵擷取工具 (Feature Extraction Tools) ──────────────────────────
def angle_between_points(a, b, c):
    """
    利用空間向量計算三點形成的關節夾角（回傳弧度值值）
    """
    ba, bc = a - b, c - b
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6) # 乘積加微小值防除以零
    return np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))

def hand_angles(hand_lms, img_w, img_h):
    """
    計算手部五根手指內所有關節的夾角（每手共 15 個角度特徵）
    """
    fingers = [
        [0, 1, 2, 3, 4],     # 大拇指
        [0, 5, 6, 7, 8],     # 食指
        [0, 9, 10, 11, 12],  # 中指
        [0, 13, 14, 15, 16], # 無名指
        [0, 17, 18, 19, 20]  # 小拇指
    ]
    angles = []
    for f in fingers:
        # 將標準化比例坐標還原成基於影像寬高的真實像素坐標空間
        pts = np.array([
            [hand_lms[i].x * img_w, hand_lms[i].y * img_h, hand_lms[i].z * img_w]
            for i in f
        ])
        # 遍歷每根手指的各個關節點計算夾角
        for j in range(1, 4):
            angles.append(angle_between_points(pts[j-1], pts[j], pts[j+1]))
    return angles

def extract_all_frames(video_path):
    """
    核心函式：逐幀讀取影片並調用 MediaPipe 解析出 218 維的特徵陣列
    """
    cap = cv2.VideoCapture(video_path)
    all_frames = []           # 儲存整支影片的所有幀特徵
    prev_right_wrist_y = None # 記錄前一幀右手腕位置用以計算速度
    prev_left_wrist_y = None  # 記錄前一幀左手腕位置用以計算速度

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break # 影片讀取結束則跳出

        frame = cv2.flip(frame, 1) # 水平鏡像翻轉
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # 內部小工具：將標準化節點轉為像素坐標矩陣
        def get_px_coord(lm):
            return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w])

        # ── A. Pose 骨架特徵擷取 ──
        pres = pose.process(rgb)
        pose_feat = [0.0] * 30   # 預設 10 個點的 X, Y, Z 共 30 維
        elbow_ang = [0.0, 0.0]   # 預設左右手肘夾角
        norm_dist = 1.0          # 肩寬比例尺
        nose_x, l_ear_x, r_ear_x = 0.5 * img_w, 0.5 * img_w, 0.5 * img_w
        ref_pt = np.zeros(3)     # 位移基準點原點

        if pres.pose_landmarks:
            lms = pres.pose_landmarks.landmark
            ref_pt = get_px_coord(lms[0]) # 以鼻子為坐標原點
            p11 = get_px_coord(lms[11])   # 左肩
            p12 = get_px_coord(lms[12])   # 右肩
            norm_dist = np.linalg.norm(p11 - p12) + 1e-6 # 以肩膀寬度作為空間歸一化基準
            nose_x = get_px_coord(lms[0])[0]
            l_ear_x = get_px_coord(lms[7])[0]
            r_ear_x = get_px_coord(lms[8])[0]
            
            pose_feat = []
            for i in POSE_IDS:
                # 計算相對於鼻子的位移並除以肩寬進行尺度歸一化
                coord = (get_px_coord(lms[i]) - ref_pt) / norm_dist
                pose_feat += coord.tolist()
                
            # 計算手肘夾角特徵
            p13, p14 = get_px_coord(lms[13]), get_px_coord(lms[14])
            p15, p16 = get_px_coord(lms[15]), get_px_coord(lms[16])
            elbow_ang[0] = angle_between_points(p11, p13, p15) / np.pi
            elbow_ang[1] = angle_between_points(p12, p14, p16) / np.pi

        # ── B. Face 人臉特徵擷取 ──
        fres = face.process(rgb)
        face_feat = [0.0] * 24 # 預設 8 個點共 24 維
        if fres.multi_face_landmarks:
            face_feat = []
            for i in FACE_IDS:
                lm = fres.multi_face_landmarks[0].landmark[i]
                coord = (get_px_coord(lm) - ref_pt) / norm_dist
                face_feat += coord.tolist()

        # ── C. Hands 雙手特徵擷取 ──
        hres = hands.process(rgb)
        left_hand_feat  = [0.0] * 63  # 21 節點 * 3 軸 = 63 維
        right_hand_feat = [0.0] * 63
        left_finger_ang  = [0.0] * 15 # 15 關節角度
        right_finger_ang = [0.0] * 15
        hand_exist = False

        # 初始化動態時序增強特徵值
        right_delta_y, right_dist_x_nose, right_dist_x_ear = 0.0, 0.0, 0.0
        left_delta_y,  left_dist_x_nose,  left_dist_x_ear  = 0.0, 0.0, 0.0

        if hres.multi_hand_landmarks and hres.multi_handedness:
            hand_exist = True
            for idx, hand_lms in enumerate(hres.multi_hand_landmarks):
                label = hres.multi_handedness[idx].classification[0].label
                wrist_lm = hand_lms.landmark[0]
                wrist_pt = get_px_coord(wrist_lm)
                
                # 計算手部各點相對於自己手腕的相對位移特徵
                feat = []
                for lm in hand_lms.landmark:
                    coord = (get_px_coord(lm) - wrist_pt) / norm_dist
                    feat += coord.tolist()
                
                angles = hand_angles(hand_lms.landmark, img_w, img_h)
                
                if label == 'Left':
                    left_hand_feat = feat
                    left_finger_ang = angles
                    # 計算左手腕垂直位移速度
                    if prev_left_wrist_y is not None:
                        left_delta_y = (wrist_pt[1] - prev_left_wrist_y) / norm_dist
                    prev_left_wrist_y = wrist_pt[1]
                    # 計算左手腕與五官的相對水平距離距離
                    left_dist_x_nose = (wrist_pt[0] - nose_x) / norm_dist
                    left_dist_x_ear  = (wrist_pt[0] - l_ear_x) / norm_dist
                else:
                    right_hand_feat = feat
                    right_finger_ang = angles
                    # 計算右手腕垂直位移速度
                    if prev_right_wrist_y is not None:
                        right_delta_y = (wrist_pt[1] - prev_right_wrist_y) / norm_dist
                    prev_right_wrist_y = wrist_pt[1]
                    # 計算右手腕與五官的相對水平距離
                    right_dist_x_nose = (wrist_pt[0] - nose_x) / norm_dist
                    right_dist_x_ear  = (wrist_pt[0] - r_ear_x) / norm_dist

        # 若當前幀沒偵測到任何手，將速度追蹤器重置為 None
        if not hand_exist:
            prev_right_wrist_y = None
            prev_left_wrist_y = None

        # 整合 6 個額外動態速度與相對位置特徵
        extra_feats = [
            right_delta_y, right_dist_x_nose, right_dist_x_ear,
            left_delta_y,  left_dist_x_nose,  left_dist_x_ear,
        ]
        
        # 依據硬性規定的結構拼接為最終的一維特徵長度
        combined = (face_feat + left_hand_feat + right_hand_feat +
                    left_finger_ang + right_finger_ang +
                    pose_feat + elbow_ang + extra_feats)

        # 嚴格校驗：確保拼出來的特徵維度正好是 218 維
        assert len(combined) == TOTAL_DIM, f"維度錯誤：{len(combined)}"
        all_frames.append(combined)

    cap.release()
    return np.array(all_frames, dtype=np.float32)

def find_active_range(all_frames, threshold=ENERGY_THRESHOLD):
    """
    動態端點偵測 (VAD)：分析左右手特徵的絕對值總和，裁切掉前後沒有在動的空白影格
    """
    start_idx = 0
    end_idx = len(all_frames)

    # 正向尋找：找出第一個手部產生大於門檻值動作的影格作為起點
    for i, frame in enumerate(all_frames):
        hand_feat = frame[24:150] # 截取特徵中左右手節點共 126 維的區域
        if np.sum(np.abs(hand_feat)) > threshold:
            start_idx = i
            break

    # 反向尋找：從最後一幀往前找，找出最後一個有動作的影格作為終點
    for i in range(len(all_frames) - 1, -1, -1):
        hand_feat = all_frames[i][24:150]
        if np.sum(np.abs(hand_feat)) > threshold:
            end_idx = i + 1
            break

    return start_idx, end_idx

def sliding_windows(all_frames, seq_len=SEQ_LEN, n_windows=N_WINDOWS):
    """
    時序對齊與切分：從有效段落中均勻取得起始點，固定切割出 3 個長度為 30 幀的子片段
    """
    total = len(all_frames)

    # 若有效長度小於 30 幀，使用 np.linspace 產生拉伸索引，強行線性內插展開
    if total < seq_len:
        indices = np.linspace(0, total - 1, seq_len).astype(int)
        stretched = all_frames[indices]
        return [stretched.copy() for _ in range(n_windows)]

    # 若長度足夠，均勻取三個不同的起始點切出長度 30 幀的連續滑動視窗
    starts = np.linspace(0, total - seq_len, n_windows).astype(int)
    return [all_frames[s:s + seq_len] for s in starts]

# ── 🚀 主資料處理流程 (Main Processing Loop) ─────────────────────────────────
for label in os.listdir(VIDEO_DIR):
    label_dir = os.path.join(VIDEO_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    # 建立對應的特徵輸出子資料夾 (如 data/謝謝_A)
    out_label_dir = os.path.join(OUT_DIR, label)
    os.makedirs(out_label_dir, exist_ok=True)

    for v in os.listdir(label_dir):
        if not v.endswith(".mp4"):
            continue

        base_name = v.replace('.mp4', '')

        # 💡 檢查點機制：若此影片的第一個片段已存在，代表處理過，直接跳過 (中斷續傳功能)
        first_out = os.path.join(out_label_dir, f"{base_name}_w0.npy")
        if os.path.exists(first_out):
            print(f"⏭️ 跳過（已處理）: {v}")
            continue

        print(f"正在處理: {v} ...")

        # 1. 調用核心函式擷取該影片的完整原始特徵
        all_frames = extract_all_frames(os.path.join(label_dir, v))

        # 2. 端點偵測切除頭尾靜止空白幀
        start_idx, end_idx = find_active_range(all_frames)
        active_frames = all_frames[start_idx:end_idx]
        print(f"  總幀數:{len(all_frames)}, 有效範圍:{start_idx}~{end_idx}, 有效幀:{len(active_frames)}")

        # 3. 滑動視窗時序切分 (固定切成 3 個視窗)
        windows = sliding_windows(active_frames)

        # 4. 針對每個視窗衍生 1 筆原始、3 筆加噪、3 筆縮放，共 7 筆特徵檔案儲存 (強大資料增強)
        count = 0
        for i, window in enumerate(windows):
            saves = [
                (f"{base_name}_w{i}",         window),                     # 原始切片資料
                (f"{base_name}_w{i}_noise0",  augment_noise(window)),      # 雜訊變體 1
                (f"{base_name}_w{i}_noise1",  augment_noise(window)),      # 雜訊變體 2
                (f"{base_name}_w{i}_noise2",  augment_noise(window)),      # 雜訊變體 3
                (f"{base_name}_w{i}_fscale0", augment_feature_scale(window)), # 尺度縮放變體 1
                (f"{base_name}_w{i}_fscale1", augment_feature_scale(window)), # 尺度縮放變體 2
                (f"{base_name}_w{i}_fscale2", augment_feature_scale(window)), # 尺度縮放變體 3
            ]
            for fname, data in saves:
                out_path = os.path.join(out_label_dir, f"{fname}.npy")
                np.save(out_path, data) # 以二進位 NumPy 格式高效儲存特徵矩陣
                count += 1

        print(f"✅ {v} → 共產生 {count} 筆資料")

print("\n✅ 所有影片處理完成！")
