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
POSE_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23]

TOTAL_DIM = 218
SEQ_LEN = 30
N_WINDOWS = 3
ENERGY_THRESHOLD = 0.5

# ── 增強方法 ──────────────────────────────────────────────
def augment_noise(seq, scale=0.01):
    return seq + np.random.normal(0, scale, seq.shape).astype(np.float32)

def augment_feature_scale(seq, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return (seq * scale).astype(np.float32)

# ── 特徵擷取工具 ──────────────────────────────────────────
def angle_between_points(a, b, c):
    ba, bc = a - b, c - b
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))

def hand_angles(hand_lms, img_w, img_h):
    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    angles = []
    for f in fingers:
        pts = np.array([
            [hand_lms[i].x * img_w, hand_lms[i].y * img_h, hand_lms[i].z * img_w]
            for i in f
        ])
        for j in range(1, 4):
            angles.append(angle_between_points(pts[j-1], pts[j], pts[j+1]))
    return angles

def extract_all_frames(video_path):
    """擷取整支影片所有幀，不做長度對齊"""
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    prev_right_wrist_y = None
    prev_left_wrist_y = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        def get_px_coord(lm):
            return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w])

        # ── Pose ──
        pres = pose.process(rgb)
        pose_feat = [0.0] * 30
        elbow_ang = [0.0, 0.0]
        norm_dist = 1.0
        nose_x, l_ear_x, r_ear_x = 0.5 * img_w, 0.5 * img_w, 0.5 * img_w
        ref_pt = np.zeros(3)

        if pres.pose_landmarks:
            lms = pres.pose_landmarks.landmark
            ref_pt = get_px_coord(lms[0])
            p11 = get_px_coord(lms[11])
            p12 = get_px_coord(lms[12])
            norm_dist = np.linalg.norm(p11 - p12) + 1e-6
            nose_x = get_px_coord(lms[0])[0]
            l_ear_x = get_px_coord(lms[7])[0]
            r_ear_x = get_px_coord(lms[8])[0]
            pose_feat = []
            for i in POSE_IDS:
                coord = (get_px_coord(lms[i]) - ref_pt) / norm_dist
                pose_feat += coord.tolist()
            p13, p14 = get_px_coord(lms[13]), get_px_coord(lms[14])
            p15, p16 = get_px_coord(lms[15]), get_px_coord(lms[16])
            elbow_ang[0] = angle_between_points(p11, p13, p15) / np.pi
            elbow_ang[1] = angle_between_points(p12, p14, p16) / np.pi

        # ── Face ──
        fres = face.process(rgb)
        face_feat = [0.0] * 24
        if fres.multi_face_landmarks:
            face_feat = []
            for i in FACE_IDS:
                lm = fres.multi_face_landmarks[0].landmark[i]
                coord = (get_px_coord(lm) - ref_pt) / norm_dist
                face_feat += coord.tolist()

        # ── Hands ──
        hres = hands.process(rgb)
        left_hand_feat  = [0.0] * 63
        right_hand_feat = [0.0] * 63
        left_finger_ang  = [0.0] * 15
        right_finger_ang = [0.0] * 15
        hand_exist = False

        right_delta_y, right_dist_x_nose, right_dist_x_ear = 0.0, 0.0, 0.0
        left_delta_y,  left_dist_x_nose,  left_dist_x_ear  = 0.0, 0.0, 0.0

        if hres.multi_hand_landmarks and hres.multi_handedness:
            hand_exist = True
            for idx, hand_lms in enumerate(hres.multi_hand_landmarks):
                label = hres.multi_handedness[idx].classification[0].label
                wrist_lm = hand_lms.landmark[0]
                wrist_pt = get_px_coord(wrist_lm)
                feat = []
                for lm in hand_lms.landmark:
                    coord = (get_px_coord(lm) - wrist_pt) / norm_dist
                    feat += coord.tolist()
                angles = hand_angles(hand_lms.landmark, img_w, img_h)
                if label == 'Left':
                    left_hand_feat = feat
                    left_finger_ang = angles
                    if prev_left_wrist_y is not None:
                        left_delta_y = (wrist_pt[1] - prev_left_wrist_y) / norm_dist
                    prev_left_wrist_y = wrist_pt[1]
                    left_dist_x_nose = (wrist_pt[0] - nose_x) / norm_dist
                    left_dist_x_ear  = (wrist_pt[0] - l_ear_x) / norm_dist
                else:
                    right_hand_feat = feat
                    right_finger_ang = angles
                    if prev_right_wrist_y is not None:
                        right_delta_y = (wrist_pt[1] - prev_right_wrist_y) / norm_dist
                    prev_right_wrist_y = wrist_pt[1]
                    right_dist_x_nose = (wrist_pt[0] - nose_x) / norm_dist
                    right_dist_x_ear  = (wrist_pt[0] - r_ear_x) / norm_dist

        if not hand_exist:
            prev_right_wrist_y = None
            prev_left_wrist_y = None

        extra_feats = [
            right_delta_y, right_dist_x_nose, right_dist_x_ear,
            left_delta_y,  left_dist_x_nose,  left_dist_x_ear,
        ]
        combined = (face_feat + left_hand_feat + right_hand_feat +
                    left_finger_ang + right_finger_ang +
                    pose_feat + elbow_ang + extra_feats)

        assert len(combined) == TOTAL_DIM, f"維度錯誤：{len(combined)}"
        all_frames.append(combined)

    cap.release()
    return np.array(all_frames, dtype=np.float32)

def find_active_range(all_frames, threshold=ENERGY_THRESHOLD):
    """找到手部動作的開始和結束位置"""
    start_idx = 0
    end_idx = len(all_frames)

    for i, frame in enumerate(all_frames):
        hand_feat = frame[24:150]
        if np.sum(np.abs(hand_feat)) > threshold:
            start_idx = i
            break

    for i in range(len(all_frames) - 1, -1, -1):
        hand_feat = all_frames[i][24:150]
        if np.sum(np.abs(hand_feat)) > threshold:
            end_idx = i + 1
            break

    return start_idx, end_idx

def sliding_windows(all_frames, seq_len=SEQ_LEN, n_windows=N_WINDOWS):
    """均勻分配起始點，固定產生 n_windows 個片段"""
    total = len(all_frames)

    # 有效幀不足時用 linspace 拉伸
    if total < seq_len:
        indices = np.linspace(0, total - 1, seq_len).astype(int)
        stretched = all_frames[indices]
        return [stretched.copy() for _ in range(n_windows)]

    starts = np.linspace(0, total - seq_len, n_windows).astype(int)
    return [all_frames[s:s + seq_len] for s in starts]

# ── 主流程 ──────────────────────────────────────────────────
for label in os.listdir(VIDEO_DIR):
    label_dir = os.path.join(VIDEO_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    out_label_dir = os.path.join(OUT_DIR, label)
    os.makedirs(out_label_dir, exist_ok=True)

    for v in os.listdir(label_dir):
        if not v.endswith(".mp4"):
            continue

        base_name = v.replace('.mp4', '')

        # 💡 先檢查是否已處理過，已處理直接跳過
        first_out = os.path.join(out_label_dir, f"{base_name}_w0.npy")
        if os.path.exists(first_out):
            print(f"⏭️ 跳過（已處理）: {v}")
            continue

        print(f"正在處理: {v} ...")

        # 1. 擷取全部幀
        all_frames = extract_all_frames(os.path.join(label_dir, v))

        # 2. 找到有效範圍
        start_idx, end_idx = find_active_range(all_frames)
        active_frames = all_frames[start_idx:end_idx]
        print(f"  總幀數:{len(all_frames)}, 有效範圍:{start_idx}~{end_idx}, 有效幀:{len(active_frames)}")

        # 3. 滑動視窗切割
        windows = sliding_windows(active_frames)

        # 4. 每個視窗做增強後儲存
        count = 0
        for i, window in enumerate(windows):
            saves = [
                (f"{base_name}_w{i}",         window),
                (f"{base_name}_w{i}_noise0",  augment_noise(window)),
                (f"{base_name}_w{i}_noise1",  augment_noise(window)),
                (f"{base_name}_w{i}_noise2",  augment_noise(window)),
                (f"{base_name}_w{i}_fscale0", augment_feature_scale(window)),
                (f"{base_name}_w{i}_fscale1", augment_feature_scale(window)),
                (f"{base_name}_w{i}_fscale2", augment_feature_scale(window)),
            ]
            for fname, data in saves:
                out_path = os.path.join(out_label_dir, f"{fname}.npy")
                np.save(out_path, data)
                count += 1

        print(f"✅ {v} → 共產生 {count} 筆資料")

print("\n✅ 所有影片處理完成！")