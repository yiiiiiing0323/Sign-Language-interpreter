import numpy as np

# ----------------------------------------------------------------
# 工具函式：計算距離與角度
# ----------------------------------------------------------------
def calculate_dist_3d(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_dist_2d(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def is_finger_open(tip_idx, pip_idx, hand_lms):
    tip = hand_lms[tip_idx]
    pip = hand_lms[pip_idx]
    wrist = hand_lms[0]
    return calculate_dist_3d(tip, wrist) > calculate_dist_3d(pip, wrist)

# ----------------------------------------------------------------
# A 流核心：特徵萃取器 (Feature Extractor)
# ----------------------------------------------------------------
class AStreamFeatureExtractor:
    def __init__(self):
        # 🌟 動態軌跡記憶區 (擴充到 30 幀，約 1 秒，讓動態抓得更穩)
        self.wrist_dist_history = []  
        self.index_history = []       
        self.chest_dist_history = []  
        self.wrist_y_history = []     # 用來判斷「往下移動」(早上) & 單手 Y 軸
        self.wrist_x_history = []     # 🟢 新增：右手腕 X 軸歷史 (判斷向外移動)
        self.hand_z_history = []      # 用來判斷 Z 軸進出 (去、來)
        self.thumb_chest_dist_history = []   # 手腕是否持續停在胸前

        # 🔧 修正跑_A / 跑_B：改用雙手腕 Y 軸追蹤，不依賴食指
        self.r_wrist_y_history = []   # 右手腕 Y 軸歷史
        self.l_wrist_y_history = []   # 左手腕 Y 軸歷史
        self.r_wrist_x_history = []   # 🟢 新增：右手腕 X 軸歷史 (畫方框)
        self.l_wrist_x_history = []   # 🟢 新增：左手腕 X 軸歷史 (畫方框)

        # 🟢 新增 history buffers
        self.hand_z_small_history = []    # 說話_A：前後小幅移動（Z 軸，約 10 幀）
        self.writing_y_history = []       # 寫：追蹤往下移動 Y 軸
        self.nose_y_history = []          # 想到：追蹤頭部後仰（鼻子 Y 上移）
        self.index_bend_history = []      # 再見：追蹤食指彎曲狀態序列
        self.rotate_history = []          # 想：追蹤食指旋轉軌跡（X,Y 極值）

    def extract_features(self, pose_results, hand_results):
        current_features = {
            # 📌 基礎距離
            'dist_HAND_8_POSE_CHEST': 99.0, 
            'dist_HAND_8_FACE_1': 99.0,     
            'dist_HAND_0_POSE_CHEST': 99.0,
            'dist_HAND_POSE_CHEST': 99.0, 
            'dist_RIGHT_HAND_0_POSE_NOSE': 99.0,
            'dist_HAND_8_LEFT_EAR': 99.0,   
            'dist_HAND_8_RIGHT_EAR': 99.0,  
            'dist_HAND_0_POSE_CHIN': 99.0,  
            
            # 🟢 新增：臉部器官距離 (V分類專用)
            'dist_HAND_FACE_MOUTH': 99.0,
            'dist_HAND_FACE_LIPS': 99.0,
            'dist_HAND_FACE_EYE': 99.0,
            'dist_HAND_START_FACE_EYE': 99.0,
            'dist_LEFT_HAND_FACE_CHIN': 99.0,
            'dist_HAND_8_FACE_CHEEK_R': 99.0,      # 🟢 新增：右臉頰 (食指)
            'dist_HAND_12_FACE_CHEEK_R': 99.0,     # 🟢 新增：右臉頰 (中指)
            'dist_HAND_16_FACE_CHEEK_R': 99.0,     # 🟢 新增：右臉頰 (無名指)
            'dist_LEFT_HAND_8_FACE_CHEEK_L': 99.0, # 🟢 新增：左臉頰
            'dist_LEFT_HAND_8_TEMPLE_L': 99.0,     # 🟢 新增：左太陽穴
            
            # 📌 「早上」專屬特徵 (舊有功能保留)
            'dist_START_RIGHT_HAND_0_POSE_HEAD': 99.0,
            'move_downwards_RIGHT_HAND': False,
            'move_upwards_RIGHT_HAND': False,             # 🟢 新增：向上移動
            'palm_facing_in_RIGHT_HAND': False,
            'palm_facing_down_HAND': False, 
            'palm_facing_out_RIGHT_HAND': False,          # 🟢 新增：掌心朝外
            
            # 📌 嚴格手勢形狀 (包含 V 字型等)
            'is_open_HAND': False,
            'is_index_pointing_HAND': False,
            'is_fist_HAND': False,
            'is_L_shape_HAND': False,
            'is_V_shape_HAND': False,
            'is_thumb_extended_HAND': False,
            
            # 🟢 新增：單手細部形狀 (V分類與再見專用)
            'is_flat_HAND': False,
            'is_flat_RIGHT_HAND': False,
            'is_pinch_HAND': False,
            'is_pinch_all_HAND': False,
            'is_pinch_RIGHT_HAND': False,
            'is_C_shape_HAND': False,
            'is_crossed_fingers_HAND': False,
            'is_index_HAND': False,
            'is_Y_shape_HAND': False,                     # 🟢 新增：Y型/6
            'is_Y_shape_RIGHT_HAND': False,               
            'is_curved_down_RIGHT_HAND': False,           # 🟢 新增：彎曲朝下 (證明)
            'is_middle_finger_extended_HAND': False,      # 🟢 新增：單伸中指 (哥哥/弟弟)
            'is_middle_finger_extended_RIGHT_HAND': False,
            'is_ring_finger_extended_HAND': False,        # 🟢 新增：單伸無名指 (姐姐/妹妹)
            'is_ring_finger_extended_RIGHT_HAND': False,
            'is_pinky_extended_HAND': False,              # 🟢 新增：單伸小指
            'is_pinky_extended_RIGHT_HAND': False,
            'is_index_pinky_extended_HAND': False,        # 🟢 新增：牛角 (媽媽)
            'is_index_pinky_extended_RIGHT_HAND': False,
            'is_thumb_up_RIGHT_HAND': False,              # 🟢 新增：拇指朝上
            'is_thumb_down_RIGHT_HAND': False,            # 🟢 新增：拇指朝下
            
            # 📌 方向與對齊
            'vector_align_HAND_8_CAMERA_AXIS': 0.0,
            'vector_align_HAND_8_SIDE_AXIS': 0.0,
            'vector_align_HAND_8_DOWN_AXIS': 0.0, 
            'is_right_front_HAND_0_POSE_CHEST': False,
            'is_facing_RIGHT_HAND_KNUCKLE_POSE_NOSE': False,
            'detect_thumb_bending_HAND': False, 
            
            # 🟢 新增：Z 軸向量 (V分類 去、來)
            'vector_align_HAND_out': False,
            'vector_change_HAND_in_out': False,
            
            # 📌 動態特徵 (點擊、畫圓、平移)
            'is_static_HAND': False,
            'is_static_RIGHT_HAND': False,
            'detect_swipe_HAND_horizontal': False,
            'detect_small_swipe_HAND_horizontal': False, 
            'detect_circle_HAND': False,
            'detect_circle_RIGHT_HAND': False,            # 🟢 新增
            'detect_tap_HAND': False,
            'detect_small_move_outwards_RIGHT_HAND': False, # 🟢 新增：向外小幅度拉開
            
            # 🟢 新增：其他動態特徵 (V分類與再見)
            'detect_scoop_HAND': False,
            'detect_scoop_RIGHT_HAND': False,
            'detect_swipe_HAND_forward': False,
            'detect_finger_bend_repeat_HAND_8': False,
            'detect_move_HAND': False,
            'detect_wave_HAND_horizontal': False, # 再見專用
            
            # 📌 雙手特徵 (平安/平靜等)
            'dist_START_RIGHT_HAND_0_START_LEFT_HAND_0': 99.0,
            'is_above_RIGHT_HAND_LEFT_HAND': False,
            'move_apart_horizontally_RIGHT_HAND_LEFT_HAND': False,
            'is_thumb_extended_LEFT_HAND': False,
            'dist_RIGHT_HAND_8_LEFT_HAND_4': 99.0,
            
            # 🟢 新增：雙手互動特徵 (V分類 跑、幫忙、下雨等)
            'is_fist_BOTH_HANDS': False,
            'is_open5_BOTH_HANDS': False,
            'is_flat_down_BOTH_HANDS': False,
            'hands_facing_BOTH_HANDS': False,
            'is_thumb_up_LEFT_HAND': False,
            'is_curved_up_LEFT_HAND': False,
            'is_index_LEFT_HAND': False,
            'is_index_RIGHT_HAND': False,
            'is_static_LEFT_HAND': False,
            'dist_RIGHT_HAND_LEFT_HAND': 99.0,
            'dist_RIGHT_HAND_LEFT_THUMB': 99.0,
            'detect_alternate_swing_BOTH_HANDS': False,
            'detect_wave_BOTH_HANDS_vertical': False,
            'detect_clap_alternate_BOTH_HANDS': False,
            'is_in_front_RIGHT_HAND_LEFT_HAND': False,    # 🟢 新增：右手在左手前方 (Z軸)
            'detect_small_move_apart_repeat_BOTH_HANDS': False, # 🟢 新增：重複小幅拉開 (學生證)
            'dist_RIGHT_HAND_20_LEFT_HAND_20': 99.0,      # 🟢 新增：小指尖距離 (學生)
            'dist_RIGHT_HAND_12_LEFT_HAND_12': 99.0,      # 🟢 新增：中指尖距離
            'dist_LEFT_HAND_0_POSE_CHEST': 99.0,          # 🟢 新增：左手腕到胸口距離
            'palms_facing_in_BOTH_HANDS': False,          # 🟢 新增：雙手掌心朝內
            'palm_facing_in_LEFT_HAND': False,            # 🟢 新增：左手掌心朝內
            'palm_facing_up_LEFT_HAND': False,            # 🟢 新增：左手掌心朝上
            'is_index_BOTH_HANDS': False,                 # 🟢 新增：雙手皆食指
            'detect_draw_square_BOTH_HANDS': False,       # 🟢 新增：畫方形 (大學)
            'detect_roof_shape_BOTH_HANDS': False,        # 🟢 新增：房子狀 (學校)
            'tips_up_BOTH_HANDS': False,                  # 🟢 新增：雙手指尖朝上
            'is_Y_shape_BOTH_HANDS': False,               # 🟢 新增：雙手Y型
            'is_flat_LEFT_HAND': False,                   # 🟢 新增：左手平攤
            'is_flat_BOTH_HANDS': False,                  # 🟢 新增：雙手平攤
            'is_middle_finger_extended_BOTH_HANDS': False,# 🟢 新增：雙手中指
            'is_ring_finger_extended_BOTH_HANDS': False,  # 🟢 新增：雙手無名指
            'is_index_pinky_extended_LEFT_HAND': False,   # 🟢 新增：左手牛角
            
            # 🟢 新增：說話_A / 說話_B 專用
            'dist_HAND_8_MOUTH': 99.0,          # 食指尖到嘴巴距離
            'dist_HAND_4_MOUTH': 99.0,           # 拇指尖到嘴巴距離
            'palm_facing_left_RIGHT_HAND': False, # 掌心朝左（小指根在拇指根右側）
            'vector_align_HAND_8_LEFT_AXIS': False, # 同上別名（Excel 舊寫法相容）
            'detect_small_move_HAND_z': False,    # 小幅度前後移動（說話_A：食指來回）
            'detect_wave_fingers_HAND': False,    # 五指輕微參差擺動（說話_B）

            # 🟢 新增：寫 專用
            'dist_RIGHT_HAND_8_LEFT_HAND_0': 99.0,  # 右手食指尖到左手腕距離（寫字位置）
            'detect_writing_motion_HAND': False,      # 寫字狀：水平小幅擺動且往下移動

            # 🟢 新增：想 / 想到 專用
            'dist_HAND_8_TEMPLE_R': 99.0,        # 食指尖到右太陽穴距離
            'detect_finger_rotate_HAND_8': False, # 食指在太陽穴旋轉（想）
            'detect_head_tilt_back': False,        # 頭後仰（想到/思念）

            # 🟢 新增：等_A / 等_B 專用
            'is_four_fingers_closed_HAND': False,  # 食中無小四指併攏伸直（等_A 手型近似）
            'is_ILY_shape_HAND': False,             # 拇指+食指+小指伸直（等_B/守）
            'dist_HAND_0_POSE_SHOULDER_R': 99.0,   # 手腕到右肩距離（守/等_B 位置）

            # 🟢 新增：再見 專用
            'detect_index_open_to_bend_HAND': False, # 雙手食指從伸直拉開至彎曲（再見）
            'detect_index_bending_HAND': False,       # 同上別名（Excel 舊寫法相容）

            # 🟢 新增：來 專用
            'is_index_up': False,                    # 食指朝上（向量 Y 軸對齊）

            # 🟢 新增：精確雙手距離 / 左手形狀
            'dist_RIGHT_HAND_8_LEFT_HAND_8': 99.0,  # 右食指尖到左食指尖（再見 Step1）
            'is_flat_palm_LEFT_HAND': False,          # 左手五指伸直攤平（寫：左掌承接）
        }

        if not pose_results or not pose_results.pose_landmarks:
            return current_features

        p_lms = pose_results.pose_landmarks.landmark
        shoulder_l = p_lms[11]
        shoulder_r = p_lms[12]
        shoulder_width = calculate_dist_2d(shoulder_l, shoulder_r) + 1e-6
        
        chest_x = (shoulder_l.x + shoulder_r.x) / 2
        chest_y = (shoulder_l.y + shoulder_r.y) / 2
        chest_point = Point2D(chest_x, chest_y)
        
        nose = p_lms[0]
        left_ear = p_lms[7]
        right_ear = p_lms[8]
        
        chin_x = (p_lms[9].x + p_lms[10].x) / 2
        chin_y = (p_lms[9].y + p_lms[10].y) / 2
        chin_point = Point2D(chin_x, chin_y)
        
        # 🟢 新增臉部參考點
        mouth_point = chin_point # 近似
        eye_point = Point2D((p_lms[2].x + p_lms[5].x) / 2, (p_lms[2].y + p_lms[5].y) / 2)
        
        # 🟢 新增：臉頰與太陽穴近似點
        cheek_r = Point2D((nose.x + right_ear.x)/2, (nose.y + right_ear.y)/2)
        cheek_l = Point2D((nose.x + left_ear.x)/2, (nose.y + left_ear.y)/2)

        if hand_results and hand_results.multi_hand_landmarks:
            hand_lms = None
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for i, handedness in enumerate(hand_results.multi_handedness):
                    if handedness.classification[0].label == "Right":
                        hand_lms = hand_results.multi_hand_landmarks[i].landmark
                        break
            
            # =========================================================
            # 🌟 1. 單手判定 (Right Hand)
            # =========================================================
            if hand_lms is not None:
                wrist = hand_lms[0]
                index_tip = hand_lms[8]
                index_mcp = hand_lms[5] 
                thumb_tip = hand_lms[4]
                thumb_mcp = hand_lms[2]
                pinky_mcp = hand_lms[17]
                
                dist_thumb_to_pinky_base = calculate_dist_3d(thumb_tip, pinky_mcp)
                dist_palm_width = calculate_dist_3d(thumb_mcp, pinky_mcp)
                thumb_open = dist_thumb_to_pinky_base > dist_palm_width

                index_open = is_finger_open(8, 6, hand_lms)
                middle_open = is_finger_open(12, 10, hand_lms)
                ring_open = is_finger_open(16, 14, hand_lms)
                pinky_open = is_finger_open(20, 18, hand_lms)
                
                # 基礎形狀
                current_features['is_fist_HAND'] = not index_open and not middle_open and not ring_open and not pinky_open
                current_features['is_open_HAND'] = (index_open + middle_open + ring_open + pinky_open) >= 3
                current_features['is_index_pointing_HAND'] = index_open and not middle_open and not ring_open and not pinky_open
                current_features['is_V_shape_HAND'] = index_open and middle_open and not ring_open and not pinky_open
                current_features['is_L_shape_HAND'] = thumb_open and index_open and not middle_open and not ring_open and not pinky_open
                current_features['is_thumb_extended_HAND'] = bool(thumb_open)

                # 🟢 新增：單手細節特徵對應
                current_features['is_flat_HAND'] = current_features['is_open_HAND']
                current_features['is_flat_RIGHT_HAND'] = current_features['is_open_HAND']
                current_features['is_index_HAND'] = current_features['is_index_pointing_HAND']
                current_features['is_index_RIGHT_HAND'] = current_features['is_index_pointing_HAND']
                
                # 🟢 新增：特殊單獨指頭特徵對應
                current_features['is_Y_shape_HAND'] = thumb_open and pinky_open and not index_open and not middle_open and not ring_open
                current_features['is_Y_shape_RIGHT_HAND'] = current_features['is_Y_shape_HAND']
                current_features['is_middle_finger_extended_HAND'] = not thumb_open and not index_open and middle_open and not ring_open and not pinky_open
                current_features['is_middle_finger_extended_RIGHT_HAND'] = current_features['is_middle_finger_extended_HAND']
                current_features['is_ring_finger_extended_HAND'] = not thumb_open and not index_open and not middle_open and ring_open and not pinky_open
                current_features['is_ring_finger_extended_RIGHT_HAND'] = current_features['is_ring_finger_extended_HAND']
                current_features['is_pinky_extended_HAND'] = not thumb_open and not index_open and not middle_open and not ring_open and pinky_open
                current_features['is_pinky_extended_RIGHT_HAND'] = current_features['is_pinky_extended_HAND']
                current_features['is_index_pinky_extended_HAND'] = not thumb_open and index_open and not middle_open and not ring_open and pinky_open
                current_features['is_index_pinky_extended_RIGHT_HAND'] = current_features['is_index_pinky_extended_HAND']

                dist_thumb_index = calculate_dist_3d(thumb_tip, index_tip)
                current_features['is_pinch_HAND'] = dist_thumb_index < 0.05
                current_features['is_pinch_RIGHT_HAND'] = current_features['is_pinch_HAND']
                current_features['is_pinch_all_HAND'] = current_features['is_pinch_HAND'] and not middle_open
                current_features['is_C_shape_HAND'] = thumb_open and index_open and not middle_open and dist_thumb_index > 0.05
                current_features['is_crossed_fingers_HAND'] = index_open and middle_open and calculate_dist_3d(hand_lms[8], hand_lms[12]) < 0.03

                # 手心判定
                current_features['palm_facing_in_RIGHT_HAND'] = abs(thumb_mcp.x - pinky_mcp.x) < 0.30
                is_horizontal = abs(thumb_mcp.y - pinky_mcp.y) < 0.1
                is_flat_z = abs(thumb_mcp.z - pinky_mcp.z) < 0.08
                current_features['palm_facing_down_HAND'] = is_horizontal and is_flat_z
                current_features['palm_facing_out_RIGHT_HAND'] = hand_lms[17].z > hand_lms[5].z # 🟢 新增：簡單掌心朝外判定 (小指在食指後方)
                
                # 🟢 新增：證明 (彎曲朝下，非全握拳)
                current_features['is_curved_down_RIGHT_HAND'] = (not index_open) and (not middle_open) and current_features['palm_facing_down_HAND'] and not current_features['is_fist_HAND']

                # 距離判定
                current_features['dist_HAND_8_FACE_1'] = round(calculate_dist_2d(index_tip, nose) / shoulder_width, 3)
                current_features['dist_HAND_8_POSE_CHEST'] = round(calculate_dist_2d(index_tip, chest_point) / shoulder_width, 3)
                current_features['dist_HAND_0_POSE_CHEST'] = round(calculate_dist_2d(wrist, chest_point) / shoulder_width, 3)
                current_features['dist_HAND_POSE_CHEST'] = current_features['dist_HAND_0_POSE_CHEST']
                current_features['dist_HAND_8_LEFT_EAR'] = round(calculate_dist_2d(index_tip, left_ear) / shoulder_width, 3)
                current_features['dist_HAND_8_RIGHT_EAR'] = round(calculate_dist_2d(index_tip, right_ear) / shoulder_width, 3)
                current_features['dist_HAND_0_POSE_CHIN'] = round(calculate_dist_2d(wrist, chin_point) / shoulder_width, 3)

                dist_wrist_nose = round(calculate_dist_2d(wrist, nose) / shoulder_width, 3)
                current_features['dist_RIGHT_HAND_0_POSE_NOSE'] = dist_wrist_nose
                current_features['dist_START_RIGHT_HAND_0_POSE_HEAD'] = dist_wrist_nose
                
                # 🟢 新增：臉頰相關距離
                current_features['dist_HAND_8_FACE_CHEEK_R'] = round(calculate_dist_2d(index_tip, cheek_r) / shoulder_width, 3)
                current_features['dist_HAND_12_FACE_CHEEK_R'] = round(calculate_dist_2d(hand_lms[12], cheek_r) / shoulder_width, 3)
                current_features['dist_HAND_16_FACE_CHEEK_R'] = round(calculate_dist_2d(hand_lms[16], cheek_r) / shoulder_width, 3)

                # 🟢 新增：臉部部位距離
                current_features['dist_HAND_FACE_MOUTH'] = round(calculate_dist_2d(index_tip, mouth_point) / shoulder_width, 3)
                current_features['dist_HAND_FACE_LIPS'] = current_features['dist_HAND_FACE_MOUTH']
                current_features['dist_HAND_FACE_EYE'] = round(calculate_dist_2d(index_tip, eye_point) / shoulder_width, 3)
                current_features['dist_HAND_START_FACE_EYE'] = round(calculate_dist_2d(wrist, eye_point) / shoulder_width, 3)

                # 向量與方向
                dx = index_tip.x - index_mcp.x
                dy = index_tip.y - index_mcp.y
                dz = index_tip.z - index_mcp.z
                finger_length = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
                
                # 🟢 新增：拇指朝上朝下
                thumb_dy = thumb_tip.y - thumb_mcp.y
                current_features['is_thumb_up_RIGHT_HAND'] = current_features['is_thumb_extended_HAND'] and not index_open and thumb_dy < -0.05
                current_features['is_thumb_down_RIGHT_HAND'] = current_features['is_thumb_extended_HAND'] and not index_open and thumb_dy > 0.05

                current_features['vector_align_HAND_8_CAMERA_AXIS'] = round(-dz / finger_length, 3)
                current_features['vector_align_HAND_8_SIDE_AXIS'] = round(abs(dx) / finger_length, 3)
                current_features['vector_align_HAND_8_DOWN_AXIS'] = round(dy / finger_length, 3)

                # 食指朝上：Y 分量為負（MediaPipe Y 軸朝下）且主要方向是 Y 軸
                current_features['is_index_up'] = (dy < -0.5 * finger_length)

                current_features['is_facing_RIGHT_HAND_KNUCKLE_POSE_NOSE'] = current_features['is_fist_HAND']
                current_features['is_right_front_HAND_0_POSE_CHEST'] = (wrist.x > chest_point.x - 0.2) and (wrist.y > nose.y)

                # 🔧 謝謝拇指展開且往下彎
                dist_4_2 = calculate_dist_3d(hand_lms[4], hand_lms[2])
                dist_3_2 = calculate_dist_3d(hand_lms[3], hand_lms[2])
                is_thumb_nodding = dist_4_2 < (dist_3_2 * 1.4)
                
                # 手腕必須在胸口附近
                wrist_near_chest = current_features['dist_HAND_0_POSE_CHEST'] < 0.80
                
                # 暫存基礎條件
                _thumb_base = bool(thumb_open and is_thumb_nodding and wrist_near_chest)
                current_features['is_right_front_HAND_0_POSE_CHEST'] = (wrist.x > chest_point.x - 0.2) and (wrist.y > nose.y)

                # 單手動態追蹤
                self.index_history.append((index_tip.x, index_tip.y))
                if len(self.index_history) > 30: self.index_history.pop(0)

                self.chest_dist_history.append(current_features['dist_HAND_8_POSE_CHEST'])
                if len(self.chest_dist_history) > 30: self.chest_dist_history.pop(0)

                # 右手腕 Y 軸歷史（單手用：早上判定）
                self.wrist_y_history.append(wrist.y)
                if len(self.wrist_y_history) > 30: self.wrist_y_history.pop(0)
                
                # 🟢 新增：右手腕 X 軸歷史
                self.wrist_x_history.append(wrist.x)
                if len(self.wrist_x_history) > 30: self.wrist_x_history.pop(0)

                # 右手腕 Y 軸歷史（雙手用：跑_A / 跑_B 擺動判定）
                self.r_wrist_y_history.append(wrist.y)
                if len(self.r_wrist_y_history) > 20: self.r_wrist_y_history.pop(0)
                
                self.hand_z_history.append(wrist.z)
                if len(self.hand_z_history) > 30: self.hand_z_history.pop(0)

                self.thumb_chest_dist_history.append(current_features['dist_HAND_0_POSE_CHEST'])
                if len(self.thumb_chest_dist_history) > 20: self.thumb_chest_dist_history.pop(0)

                # 往下 / 往上 移動判定
                if len(self.wrist_y_history) >= 15:
                    y_diff = self.wrist_y_history[-1] - self.wrist_y_history[0]
                    if y_diff > 0.03:
                        current_features['move_downwards_RIGHT_HAND'] = True
                    if y_diff < -0.03:
                        current_features['move_upwards_RIGHT_HAND'] = True  # 🟢 新增
                    # 🟢 新增：Scoop (往上提)
                    current_features['detect_scoop_HAND'] = (self.wrist_y_history[0] - self.wrist_y_history[-1] > 0.02)
                    current_features['detect_scoop_RIGHT_HAND'] = current_features['detect_scoop_HAND']

                # 🟢 新增：向外小幅度拉開
                if len(self.wrist_x_history) >= 15:
                    x_diff = self.wrist_x_history[-1] - self.wrist_x_history[0]
                    current_features['detect_small_move_outwards_RIGHT_HAND'] = x_diff > 0.02

                # 🟢 新增：Z 軸進出判定 (去、來)
                if len(self.hand_z_history) >= 10:
                    z_diff = self.hand_z_history[-1] - self.hand_z_history[0]
                    current_features['detect_move_HAND'] = abs(z_diff) > 0.02
                    current_features['vector_align_HAND_out'] = z_diff > 0.02
                    current_features['detect_swipe_HAND_forward'] = z_diff > 0.02
                    current_features['vector_change_HAND_in_out'] = abs(z_diff) > 0.02

                # X, Y 位移判斷 (畫圓、平移、揮手)
                if len(self.index_history) >= 15:
                    xs = [p[0] for p in self.index_history]
                    ys = [p[1] for p in self.index_history]
                    x_range = max(xs) - min(xs)
                    y_range = max(ys) - min(ys)

                    is_static = (x_range < 0.08) and (y_range < 0.08)
                    current_features['is_static_HAND'] = is_static
                    current_features['is_static_RIGHT_HAND'] = is_static
                    current_features['detect_swipe_HAND_horizontal'] = (x_range > 0.13) and (x_range > y_range * 1.5)
                    current_features['detect_small_swipe_HAND_horizontal'] = (0.08 < x_range <= 0.13) and (x_range > y_range * 1.2)
                    current_features['detect_circle_HAND'] = (x_range > 0.05) and (y_range > 0.05)
                    current_features['detect_circle_RIGHT_HAND'] = current_features['detect_circle_HAND'] # 🟢 新增
                    
                    # 再見：張開手且水平來回晃動
                    current_features['detect_wave_HAND_horizontal'] = (x_range > 0.08) and (y_range < 0.08) and current_features['is_open_HAND']

                    # 謝謝_A：確認最近 10 幀手腕都持續在胸前附近 (放寬到 0.85)
                    stayed_near_chest = (
                        len(self.thumb_chest_dist_history) >= 10 and
                        all(d < 0.85 for d in self.thumb_chest_dist_history[-10:])
                    )
                    current_features['detect_thumb_bending_HAND'] = (
                        _thumb_base and
                        stayed_near_chest 
                    )

                # 點擊判定 (Tap)
                if len(self.chest_dist_history) >= 15:
                    hist = self.chest_dist_history
                    min_idx = np.argmin(hist)
                    if 3 < min_idx < len(hist) - 3:
                        if (hist[0] - hist[min_idx] > 0.015) and (hist[-1] - hist[min_idx] > 0.015):
                            current_features['detect_tap_HAND'] = True

            # =========================================================
            # 🌟 1.5 新增單手特徵計算
            # =========================================================
            if hand_lms is not None:

                # ── 掌心朝左判定 ──
                # 右手：拇指根(2)在小指根(17)左邊 = 掌心朝左
                current_features['palm_facing_left_RIGHT_HAND'] = (
                    hand_lms[2].x < hand_lms[17].x
                )
                # 別名（Excel 舊寫法相容）
                current_features['vector_align_HAND_8_LEFT_AXIS'] = current_features['palm_facing_left_RIGHT_HAND']

                # ── 嘴巴距離 (說話_A / 說話_B) ──
                # 用 MediaPipe Pose 的 mouth 近似點 = chin_point（已在上方定義）
                # 再用食指尖(8)和拇指尖(4)分別計算
                current_features['dist_HAND_8_MOUTH'] = round(
                    calculate_dist_2d(hand_lms[8], mouth_point) / shoulder_width, 3
                )
                current_features['dist_HAND_4_MOUTH'] = round(
                    calculate_dist_2d(hand_lms[4], mouth_point) / shoulder_width, 3
                )

                # ── 說話_A：小幅度前後移動（Z 軸振盪） ──
                self.hand_z_small_history.append(hand_lms[8].z)
                if len(self.hand_z_small_history) > 15:
                    self.hand_z_small_history.pop(0)
                if len(self.hand_z_small_history) >= 10:
                    z_vals = self.hand_z_small_history
                    z_range = max(z_vals) - min(z_vals)
                    # 小幅度：有振盪但不超過大幅度 threshold
                    current_features['detect_small_move_HAND_z'] = (0.02 < z_range <= 0.08)

                # ── 說話_B：五指輕微參差擺動（近似：open + 水平 x_range 小幅擺動） ──
                current_features['detect_wave_fingers_HAND'] = (
                    current_features['is_open_HAND'] and
                    current_features['detect_small_swipe_HAND_horizontal']
                )

                # ── 太陽穴距離（想 / 想到） ──
                # 右太陽穴近似 = 右耳 + 往內 / 往下偏移；用右耳(8)做近似
                temple_r = right_ear  # p_lms[8] 是右耳，最近似太陽穴
                current_features['dist_HAND_8_TEMPLE_R'] = round(
                    calculate_dist_2d(hand_lms[8], temple_r) / shoulder_width, 3
                )

                # ── 想：食指在太陽穴旋轉 ──
                # 以食指尖 (X,Y) 軌跡判斷：需同時有 X 和 Y 位移（近似圓形）
                self.rotate_history.append((hand_lms[8].x, hand_lms[8].y))
                if len(self.rotate_history) > 20: self.rotate_history.pop(0)
                if len(self.rotate_history) >= 15:
                    rx = [p[0] for p in self.rotate_history]
                    ry = [p[1] for p in self.rotate_history]
                    rx_range = max(rx) - min(rx)
                    ry_range = max(ry) - min(ry)
                    # 小圓圈：X 和 Y 都要動，且幅度接近
                    current_features['detect_finger_rotate_HAND_8'] = (
                        rx_range > 0.03 and ry_range > 0.03 and
                        abs(rx_range - ry_range) < 0.04
                    )

                # ── 想到/思念：頭後仰 ──
                # 鼻子 Y 座標在 MediaPipe 中，Y 越小 = 越高（螢幕坐標）
                # 頭後仰 → 鼻子 Y 變小
                self.nose_y_history.append(nose.y)
                if len(self.nose_y_history) > 20: self.nose_y_history.pop(0)
                if len(self.nose_y_history) >= 10:
                    # 後段比前段 Y 更小 = 頭往後仰
                    head_moved_up = self.nose_y_history[0] - self.nose_y_history[-1] > 0.03
                    current_features['detect_head_tilt_back'] = head_moved_up

                # ── 等_A：四指（食中無小）伸直、拇指彎曲收起 ──
                # 手型：食、中、無名、小指伸直，拇指微彎收在掌心
                current_features['is_four_fingers_closed_HAND'] = (
                    index_open and middle_open and ring_open and pinky_open and not thumb_open
                )

                # ── 等_B/守：拇指+食指+小指伸直（ILY 手勢） ──
                current_features['is_ILY_shape_HAND'] = (
                    thumb_open and index_open and pinky_open and
                    not middle_open and not ring_open
                )

                # ── 守/等_B：手腕到右肩距離 ──
                current_features['dist_HAND_0_POSE_SHOULDER_R'] = round(
                    calculate_dist_2d(hand_lms[0], shoulder_r) / shoulder_width, 3
                )

                # ── 寫字動作：水平小幅擺動 + 整體往下移動 ──
                self.writing_y_history.append(hand_lms[0].y)
                if len(self.writing_y_history) > 25: self.writing_y_history.pop(0)
                if len(self.writing_y_history) >= 15:
                    wy = self.writing_y_history
                    wy_drift_down = wy[-1] - wy[0] > 0.02     # 整體往下漂移
                    # 結合水平小幅擺動條件
                    current_features['detect_writing_motion_HAND'] = (
                        wy_drift_down and
                        current_features['detect_small_swipe_HAND_horizontal']
                    )

                # ── 再見：雙手食指伸直→分開→彎曲 ──
                # 此特徵需要雙手，先記錄右手食指彎曲狀態
                self.index_bend_history.append(index_open)
                if len(self.index_bend_history) > 30: self.index_bend_history.pop(0)

            # =========================================================
            # 🌟 2. 雙手特徵 (平安/平靜等 + 新增 V分類互動)
            # =========================================================
            if len(hand_results.multi_hand_landmarks) == 2 and hand_results.multi_handedness:
                hand_dict = {}
                for idx, hand_handedness in enumerate(hand_results.multi_handedness):
                    label = hand_handedness.classification[0].label
                    hand_dict[label] = hand_results.multi_hand_landmarks[idx].landmark

                if 'Right' in hand_dict and 'Left' in hand_dict:
                    r_lms = hand_dict['Right']
                    l_lms = hand_dict['Left']
                    
                    r_wrist = r_lms[0]
                    l_wrist = l_lms[0]
                    
                    l_thumb_tip = l_lms[4]
                    l_thumb_mcp = l_lms[2]
                    l_pinky_mcp = l_lms[17]
                    l_thumb_open = calculate_dist_3d(l_thumb_tip, l_pinky_mcp) > calculate_dist_3d(l_thumb_mcp, l_pinky_mcp)
                    current_features['is_thumb_extended_LEFT_HAND'] = l_thumb_open

                    # 🟢 新增：左手其他手指形狀
                    l_index_open = is_finger_open(8, 6, l_lms)
                    l_middle_open = is_finger_open(12, 10, l_lms)
                    l_ring_open = is_finger_open(16, 14, l_lms)
                    l_pinky_open = is_finger_open(20, 18, l_lms)
                    
                    # 嚴格判定左手單獨伸出的手指
                    current_features['is_thumb_up_LEFT_HAND'] = l_thumb_open and not l_index_open and not l_middle_open and not l_ring_open and not l_pinky_open
                    current_features['is_index_LEFT_HAND'] = l_index_open and not l_middle_open and not l_ring_open and not l_pinky_open and not l_thumb_open
                    current_features['is_curved_up_LEFT_HAND'] = l_index_open and l_middle_open
                    
                    # 🟢 新增：雙手綜合形狀
                    l_open_count = l_index_open + l_middle_open + l_ring_open + l_pinky_open
                    l_fist = (l_open_count == 0)
                    current_features['is_fist_BOTH_HANDS'] = current_features['is_fist_HAND'] and l_fist
                    
                    # 🟢 新增：左手單獨形狀擴充
                    current_features['is_flat_LEFT_HAND'] = (l_index_open and l_middle_open and l_ring_open and l_pinky_open)
                    current_features['is_index_pinky_extended_LEFT_HAND'] = not l_thumb_open and l_index_open and not l_middle_open and not l_ring_open and l_pinky_open

                    # 🟢 新增：雙手同步形狀
                    current_features['is_Y_shape_BOTH_HANDS'] = current_features['is_Y_shape_RIGHT_HAND'] and (l_thumb_open and l_pinky_open and not l_index_open and not l_middle_open and not l_ring_open)
                    current_features['is_flat_BOTH_HANDS'] = current_features['is_flat_RIGHT_HAND'] and current_features['is_flat_LEFT_HAND']
                    current_features['is_index_BOTH_HANDS'] = current_features['is_index_RIGHT_HAND'] and current_features['is_index_LEFT_HAND']
                    current_features['is_middle_finger_extended_BOTH_HANDS'] = current_features['is_middle_finger_extended_RIGHT_HAND'] and (not l_thumb_open and not l_index_open and l_middle_open and not l_ring_open and not l_pinky_open)
                    current_features['is_ring_finger_extended_BOTH_HANDS'] = current_features['is_ring_finger_extended_RIGHT_HAND'] and (not l_thumb_open and not l_index_open and not l_middle_open and l_ring_open and not l_pinky_open)

                    # 跑_B：放寬五指張開的條件 (只要雙手都至少開3指就算張開)
                    current_features['is_open5_BOTH_HANDS'] = current_features['is_open_HAND'] and (l_open_count >= 3)
                    current_features['is_flat_down_BOTH_HANDS'] = current_features['is_open5_BOTH_HANDS']

                    # =========================================================
                    # hands_facing_BOTH_HANDS — 掌心相對
                    # 右手大拇指根在右手小拇指根的左側（x 較小）左手大拇指根在左手小拇指根的右側（x 較大）
                    # = 兩手拇指側互相面對 = 掌心相對
                    # =========================================================
                    r_thumb_faces_left  = r_lms[2].x < r_lms[17].x   # 右手：拇指根在小指根左邊
                    l_thumb_faces_right = l_lms[2].x > l_lms[17].x   # 左手：拇指根在小指根右邊
                    current_features['hands_facing_BOTH_HANDS'] = r_thumb_faces_left and l_thumb_faces_right
                    
                    # 🟢 新增：雙手掌心朝向
                    current_features['palm_facing_in_LEFT_HAND'] = abs(l_lms[2].x - l_lms[17].x) < 0.30
                    current_features['palms_facing_in_BOTH_HANDS'] = current_features['palm_facing_in_RIGHT_HAND'] and current_features['palm_facing_in_LEFT_HAND']
                    current_features['palm_facing_up_LEFT_HAND'] = l_wrist.y > l_lms[12].y and not (abs(l_lms[5].z - l_lms[17].z) < 0.08)
                    current_features['tips_up_BOTH_HANDS'] = (r_wrist.y > r_lms[12].y) and (l_wrist.y > l_lms[12].y)

                    # =========================================================
                    # 左手腕 Y 軸歷史（跑_A / 跑_B 擺動判定）
                    # =========================================================
                    self.l_wrist_y_history.append(l_wrist.y)
                    if len(self.l_wrist_y_history) > 20: self.l_wrist_y_history.pop(0)
                    
                    # 🟢 新增：雙手腕 X 軸歷史 (畫方形)
                    self.r_wrist_x_history.append(r_wrist.x)
                    self.l_wrist_x_history.append(l_wrist.x)
                    if len(self.r_wrist_x_history) > 20: self.r_wrist_x_history.pop(0)
                    if len(self.l_wrist_x_history) > 20: self.l_wrist_x_history.pop(0)

                    # =========================================================
                    # detect_alternate_swing_BOTH_HANDS
                    # 直接追蹤雙手腕的 Y 軸 range，只要雙手腕都在動就算擺動
                    # =========================================================
                    if len(self.r_wrist_y_history) >= 10 and len(self.l_wrist_y_history) >= 10:
                        r_y_range = max(self.r_wrist_y_history) - min(self.r_wrist_y_history)
                        l_y_range = max(self.l_wrist_y_history) - min(self.l_wrist_y_history)
                        both_wrists_moving = (r_y_range > 0.04) and (l_y_range > 0.04)
                        
                        # 只要雙手沒有五指全開，且都在上下擺動，就判定為擺臂
                        current_features['detect_alternate_swing_BOTH_HANDS'] = (
                            not current_features['is_open5_BOTH_HANDS'] and both_wrists_moving
                        )

                    # =========================================================
                    # 🔧 修正：detect_wave_BOTH_HANDS_vertical
                    # 舊邏輯只追蹤單手腕 Y，且沒加掌心相對條件，容易與平靜混淆
                    # 新邏輯：雙手腕都要有 Y 軸位移，且需確認掌心相對
                    # =========================================================
                    if len(self.r_wrist_y_history) >= 10 and len(self.l_wrist_y_history) >= 10:
                        r_y = max(self.r_wrist_y_history) - min(self.r_wrist_y_history)
                        l_y = max(self.l_wrist_y_history) - min(self.l_wrist_y_history)
                        current_features['detect_wave_BOTH_HANDS_vertical'] = (
                            current_features['is_open5_BOTH_HANDS'] and
                            current_features['hands_facing_BOTH_HANDS'] and
                            (r_y > 0.04) and (l_y > 0.04)
                        )

                    # 右手食指尖(8) 到 左手大拇指尖(4) 的距離
                    r_index_tip = r_lms[8]
                    dist_r8_l4 = calculate_dist_2d(r_index_tip, l_thumb_tip) / shoulder_width
                    current_features['dist_RIGHT_HAND_8_LEFT_HAND_4'] = round(dist_r8_l4, 3)

                    # 🟢 右手食指尖(8) 到 左手手腕(0) 的距離（寫：右手在左掌心上）
                    dist_r8_l0 = calculate_dist_2d(r_lms[8], l_lms[0]) / shoulder_width
                    current_features['dist_RIGHT_HAND_8_LEFT_HAND_0'] = round(dist_r8_l0, 3)

                    # 🟢 右手食指尖(8) 到 左手食指尖(8) 的距離（再見 Step1 精確靠近）
                    dist_r8_l8 = calculate_dist_2d(r_lms[8], l_lms[8]) / shoulder_width
                    current_features['dist_RIGHT_HAND_8_LEFT_HAND_8'] = round(dist_r8_l8, 3)

                    # 🟢 左手五指伸直攤平（寫：左掌承接）—— 沿用上方已算好的 l_*_open
                    current_features['is_flat_palm_LEFT_HAND'] = (
                        l_index_open and l_middle_open and l_ring_open and l_pinky_open
                    )

                    # 雙手手腕與身體部位距離
                    dist_hands = calculate_dist_2d(r_wrist, l_wrist) / shoulder_width
                    current_features['dist_START_RIGHT_HAND_0_START_LEFT_HAND_0'] = round(dist_hands, 3)
                    current_features['dist_RIGHT_HAND_LEFT_HAND'] = round(dist_hands, 3)
                    current_features['dist_RIGHT_HAND_LEFT_THUMB'] = round(calculate_dist_2d(r_wrist, l_thumb_tip) / shoulder_width, 3)
                    current_features['dist_LEFT_HAND_FACE_CHIN'] = round(calculate_dist_2d(l_wrist, chin_point) / shoulder_width, 3)
                    current_features['is_above_RIGHT_HAND_LEFT_HAND'] = r_wrist.y < l_wrist.y
                    
                    # 🟢 新增：雙手綜合距離與前後相對關係
                    current_features['dist_RIGHT_HAND_20_LEFT_HAND_20'] = round(calculate_dist_2d(r_lms[20], l_lms[20]) / shoulder_width, 3)
                    current_features['dist_RIGHT_HAND_12_LEFT_HAND_12'] = round(calculate_dist_2d(r_lms[12], l_lms[12]) / shoulder_width, 3)
                    current_features['dist_LEFT_HAND_0_POSE_CHEST'] = round(calculate_dist_2d(l_wrist, chest_point) / shoulder_width, 3)
                    current_features['dist_LEFT_HAND_8_FACE_CHEEK_L'] = round(calculate_dist_2d(l_lms[8], cheek_l) / shoulder_width, 3)
                    current_features['dist_LEFT_HAND_8_TEMPLE_L'] = round(calculate_dist_2d(l_lms[8], left_ear) / shoulder_width, 3) # 用左耳近似
                    current_features['is_in_front_RIGHT_HAND_LEFT_HAND'] = r_wrist.z < (l_wrist.z - 0.05) # Z值較小代表較靠近鏡頭(前方)
                    
                    current_dist_x = abs(r_wrist.x - l_wrist.x) / shoulder_width

                    # 🟢 再見：雙手食指伸直拉開後彎曲（current_dist_x 已在上方定義）
                    r_index_open_now = is_finger_open(8, 6, r_lms)
                    l_index_open_now = is_finger_open(8, 6, l_lms)
                    if len(self.index_bend_history) >= 20:
                        was_open = any(self.index_bend_history[:10])
                        is_now_bent = not r_index_open_now and not l_index_open_now
                        hands_spread = current_dist_x > 0.5
                        current_features['detect_index_open_to_bend_HAND'] = (
                            was_open and is_now_bent and hands_spread
                        )
                        current_features['detect_index_bending_HAND'] = current_features['detect_index_open_to_bend_HAND']
                    
                    # 🟢 雙手水平拉開與重複拉開
                    self.wrist_dist_history.append(current_dist_x)
                    if len(self.wrist_dist_history) > 20:
                        self.wrist_dist_history.pop(0)
                        
                    if len(self.wrist_dist_history) >= 15:
                        if current_dist_x - min(self.wrist_dist_history) > 0.03:
                            current_features['move_apart_horizontally_RIGHT_HAND_LEFT_HAND'] = True
                        
                        # 學生證：重複拉開 (偵測距離歷史中是否有多次波峰波谷)
                        hist = self.wrist_dist_history
                        peaks = sum(1 for i in range(1, len(hist)-1) if hist[i] > hist[i-1] and hist[i] > hist[i+1])
                        current_features['detect_small_move_apart_repeat_BOTH_HANDS'] = peaks >= 1 and current_features['move_apart_horizontally_RIGHT_HAND_LEFT_HAND']
                        
                    # 🟢 幫忙_S 拍手判定
                    current_features['detect_clap_alternate_BOTH_HANDS'] = (
                        current_features['is_flat_down_BOTH_HANDS'] and
                        current_features['move_apart_horizontally_RIGHT_HAND_LEFT_HAND']
                    )
                    
                    # 🟢 大學：雙手畫方框
                    if len(self.r_wrist_x_history) >= 15:
                        rx_range = max(self.r_wrist_x_history) - min(self.r_wrist_x_history)
                        ry_range = max(self.r_wrist_y_history) - min(self.r_wrist_y_history)
                        current_features['detect_draw_square_BOTH_HANDS'] = rx_range > 0.05 and ry_range > 0.05
                    
                    # 🟢 學校：房子形狀
                    current_features['detect_roof_shape_BOTH_HANDS'] = (
                        current_features['tips_up_BOTH_HANDS'] and
                        current_features['dist_RIGHT_HAND_LEFT_HAND'] > 0.15 and
                        current_features['dist_RIGHT_HAND_12_LEFT_HAND_12'] < 0.08
                    )

                else: 
                    self.wrist_dist_history.clear()
                    self.l_wrist_y_history.clear()
                    self.r_wrist_x_history.clear()
                    self.l_wrist_x_history.clear()

        else: 
            self.wrist_dist_history.clear()
            self.index_history.clear()
            self.chest_dist_history.clear()
            self.wrist_y_history.clear()
            self.wrist_x_history.clear()
            self.hand_z_history.clear()
            self.r_wrist_y_history.clear()
            self.l_wrist_y_history.clear()
            self.r_wrist_x_history.clear()
            self.l_wrist_x_history.clear()
            self.thumb_chest_dist_history.clear()
            self.hand_z_small_history.clear()
            self.writing_y_history.clear()
            self.nose_y_history.clear()
            self.index_bend_history.clear()
            self.rotate_history.clear()

        return current_features