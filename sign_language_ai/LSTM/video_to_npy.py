import cv2
import mediapipe as mp
import numpy as np
import os
VIDEO_DIR = "videos"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=2)
face = mp_face.FaceMesh()
pose = mp_pose.Pose()
FACE_IDS = [33, 133, 362, 263, 1, 61, 291, 199]
# 0:鼻子, 7:左耳, 8:右耳, 11/12:肩膀, 13/14:手肘, 15/16:手腕, 23:髖 (共 10 點)
POSE_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23]
# 指尖 landmark ID
FINGERTIP_IDS = [4, 8, 12, 16, 20]
# 最終維度：
# face_feat:        24  (8點 × 3)
# left_hand_feat:   63  (21點 × 3)
# right_hand_feat:  63  (21點 × 3)
# left_finger_ang:  15  (5指 × 3關節)
# right_finger_ang: 15  (5指 × 3關節)
# pose_feat:        30  (10點 × 3)
# elbow_ang:         2  (左右手肘角，正規化)
# extra_feats:       6  (右手 delta_y + dist_x_nose + dist_x_rear
#                        左手 delta_y + dist_x_nose + dist_x_rear)
# ─────────────────────
# 合計:            218
TOTAL_DIM = 218
def angle_between_points(a, b, c):
    """計算 a-b-c 三點夾角（弧度）"""
    ba, bc = a - b, c - b
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))
def hand_angles(hand_lms):
    """
    計算每根手指各關節的彎曲角度（共 5 指 × 3 關節 = 15 維）
    使用相鄰三點夾角：(j-1, j, j+1)
    """
    fingers = [
        [0, 1, 2, 3, 4],    # 拇指
        [0, 5, 6, 7, 8],    # 食指
        [0, 9, 10, 11, 12], # 中指
        [0, 13, 14, 15, 16],# 無名指
        [0, 17, 18, 19, 20] # 小指
    ]
    angles = []
    for f in fingers:
        pts = np.array([[hand_lms[i].x, hand_lms[i].y, hand_lms[i].z] for i in f])
        for j in range(1, 4):
            angles.append(angle_between_points(pts[j-1], pts[j], pts[j+1]))
    return angles  # 15 維，單位 radian，範圍 [0, π]
def extract_landmarks(video_path, seq_len=30):
    cap = cv2.VideoCapture(video_path)
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    seq = []
    # --- 初始化上一幀手腕 Y 座標（用於計算位移）---
    prev_right_wrist_y = None
    prev_left_wrist_y = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # ── 1. Pose（基準正規化）────────────────────────────────────────
        pres = pose.process(rgb)
        pose_feat = [0.0] * 30   # 10點 × 3
        elbow_ang = [0.0, 0.0]
        norm_dist = 1.0
        nose_x, r_ear_x = 0.5, 0.5
        ref_pt = np.zeros(3)
        if pres.pose_landmarks:
            lms = pres.pose_landmarks.landmark
            # 參考點：鼻子 (ID 0)
            ref_pt = np.array([lms[0].x, lms[0].y, lms[0].z])
            # 正規化基準：肩寬
            p11 = np.array([lms[11].x, lms[11].y, lms[11].z])
            p12 = np.array([lms[12].x, lms[12].y, lms[12].z])
            norm_dist = np.linalg.norm(p11 - p12) + 1e-6
            # ✅ 修正 3：正確更新 nose_x / r_ear_x
            nose_x = lms[0].x
            r_ear_x = lms[8].x  # POSE_IDS 中 ID=8 為右耳
            pose_feat = []
            for i in POSE_IDS:
                coord = (np.array([lms[i].x, lms[i].y, lms[i].z]) - ref_pt) / norm_dist
                pose_feat += coord.tolist()
            # 手肘角（肩→手肘→手腕）
            p13 = np.array([lms[13].x, lms[13].y, lms[13].z])
            p14 = np.array([lms[14].x, lms[14].y, lms[14].z])
            p15 = np.array([lms[15].x, lms[15].y, lms[15].z])
            p16 = np.array([lms[16].x, lms[16].y, lms[16].z])
            # ✅ 修正 4：elbow_ang 正規化到 [0, 1]
            elbow_ang[0] = angle_between_points(p11, p13, p15) / np.pi
            elbow_ang[1] = angle_between_points(p12, p14, p16) / np.pi
        # ── 2. Face ──────────────────────────────────────────────────────
        fres = face.process(rgb)
        face_feat = [0.0] * 24  # 8點 × 3
        if fres.multi_face_landmarks:
            face_feat = []
            for i in FACE_IDS:
                lm = fres.multi_face_landmarks[0].landmark[i]
                # ✅ 修正 2：減去鼻子參考點後再除以肩寬
                coord = (np.array([lm.x, lm.y, lm.z]) - ref_pt) / norm_dist
                face_feat += coord.tolist()
        # ── 3. Hands ─────────────────────────────────────────────────────
        hres = hands.process(rgb)
        left_hand_feat  = [0.0] * 63
        right_hand_feat = [0.0] * 63
        left_finger_ang  = [0.0] * 15
        right_finger_ang = [0.0] * 15
        hand_exist = False
        # ✅ 修正 6：左右手各自的差異化特徵
        right_delta_y, right_dist_x_nose, right_dist_x_rear = 0.0, 0.0, 0.0
        left_delta_y,  left_dist_x_nose,  left_dist_x_rear  = 0.0, 0.0, 0.0
        if hres.multi_hand_landmarks and hres.multi_handedness:
            hand_exist = True
            for idx, hand_lms in enumerate(hres.multi_hand_landmarks):
                label = hres.multi_handedness[idx].classification[0].label
                wrist_lm = hand_lms.landmark[0]
                wrist_pt = np.array([wrist_lm.x, wrist_lm.y, wrist_lm.z])
                # 手部座標：以手腕為原點，肩寬正規化
                feat = []
                for lm in hand_lms.landmark:
                    coord = (np.array([lm.x, lm.y, lm.z]) - wrist_pt) / norm_dist
                    feat += coord.tolist()
                angles = hand_angles(hand_lms.landmark)
                if label == 'Left':
                    left_hand_feat = feat
                    left_finger_ang = angles
                    # ✅ 修正 6：左手差異化特徵
                    if prev_left_wrist_y is not None:
                        left_delta_y = (wrist_lm.y - prev_left_wrist_y) / norm_dist
                    prev_left_wrist_y = wrist_lm.y
                    left_dist_x_nose = (wrist_lm.x - nose_x) / norm_dist
                    left_dist_x_rear = (wrist_lm.x - r_ear_x) / norm_dist
                else:  # Right
                    right_hand_feat = feat
                    right_finger_ang = angles
                    if prev_right_wrist_y is not None:
                        right_delta_y = (wrist_lm.y - prev_right_wrist_y) / norm_dist
                    prev_right_wrist_y = wrist_lm.y
                    right_dist_x_nose = (wrist_lm.x - nose_x) / norm_dist
                    right_dist_x_rear = (wrist_lm.x - r_ear_x) / norm_dist
        if not hand_exist:
            prev_right_wrist_y = None
            prev_left_wrist_y = None
        # ── 4. 拼接 ──────────────────────────────────────────────────────
        # 24 + 63 + 63 + 15 + 15 + 30 + 2 + 6 = 218
        extra_feats = [
            right_delta_y, right_dist_x_nose, right_dist_x_rear,
            left_delta_y,  left_dist_x_nose,  left_dist_x_rear,
        ]
        combined = (face_feat + left_hand_feat + right_hand_feat +
                    left_finger_ang + right_finger_ang +
                    pose_feat + elbow_ang + extra_feats)
        assert len(combined) == TOTAL_DIM, f"維度錯誤：{len(combined)}"
        # ✅ 修正 7：所有幀都保留（無手部時用零填充）
        seq.append(combined)
    cap.release()
    # ── 5. 序列長度對齊 ───────────────────────────────────────────────
    if len(seq) == 0:
        return np.zeros((seq_len, TOTAL_DIM), dtype=np.float32)
    seq_arr = np.array(seq, dtype=np.float32)
    if len(seq_arr) >= seq_len:
        indices = np.linspace(0, len(seq_arr) - 1, seq_len).astype(int)
        final_seq = seq_arr[indices]
    else:
        pad = np.zeros((seq_len - len(seq_arr), TOTAL_DIM), dtype=np.float32)
        final_seq = np.vstack([seq_arr, pad])
    return final_seq
# ── 主流程 ────────────────────────────────────────────────────────────────
for label in os.listdir(VIDEO_DIR):
    label_dir = os.path.join(VIDEO_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for v in os.listdir(label_dir):
        if not v.endswith(".mp4"):
            continue
        out_name = f"{label}_{v.replace('.mp4', '')}.npy"
        out_path = os.path.join(OUT_DIR, out_name)
        if os.path.exists(out_path):
            continue
        data = extract_landmarks(os.path.join(label_dir, v))
        np.save(out_path, data)
        print(f"Saved: {out_name} | Shape: {data.shape} | Dim: {TOTAL_DIM}")