import sys
sys.dont_write_bytecode = True

import numpy as np
import mediapipe as mp
from collections import defaultdict
import time

from core.feature_registry import FEATURE_REGISTRY

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

def get_hand_size(hand_lms):
    # 用手腕到中指根部的距離代表手的視覺大小 (用來判斷前後移動)
    return calculate_dist_2d(hand_lms[0], hand_lms[9])

# ========================================
# ===== 新增：計算角度的工具函式 =====
# ========================================

def angle_between_points(a, b, c):
    """
    計算三點之間的夾角（弧度）
    
    參數:
        a, b, c: numpy array，形狀為 (3,) 代表 3D 座標
    
    回傳:
        弧度值（0 到 π）
    """
    ba = a - b
    bc = c - b
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return np.arccos(cos_angle)

def hand_angles(hand_lms, img_w, img_h):
    """
    計算五根手指的 15 個彎曲角度
    
    參數:
        hand_lms: MediaPipe 的 hand_landmarks (21 個點)
        img_w, img_h: 影像寬高（用於像素轉換）
    
    回傳:
        list of float，長度為 15
    """
    # 定義五根手指的關鍵點 index
    fingers = [
        [0, 1, 2, 3, 4],      # 大拇指
        [0, 5, 6, 7, 8],      # 食指
        [0, 9, 10, 11, 12],   # 中指
        [0, 13, 14, 15, 16],  # 無名指
        [0, 17, 18, 19, 20]   # 小指
    ]
    
    angles = []
    for f in fingers:
        # 將 MediaPipe 的歸一化座標轉成像素座標
        pts = np.array([
            [hand_lms[i].x * img_w, hand_lms[i].y * img_h, hand_lms[i].z * img_w]
            for i in f
        ])
        
        # 每根手指計算 3 個關節角度
        for j in range(1, 4):
            angles.append(angle_between_points(pts[j-1], pts[j], pts[j+1]))
    
    return angles  # 總共 5 根手指 * 3 個角度 = 15 個值

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
        self.vibrate_z_history = []       # 抖動：追蹤手腕 Z 軸振盪（冷/痛）
        self.nose_sway_history = []        # 飆車：追蹤鼻子 X 軸左右搖擺
        
        # 🌟🌟 針對新詞彙(划船、車等) 新增的動態歷史緩衝區 🌟🌟
        self.hand_size_history = []       # 右手大小歷史 (判斷前後移動)
        self.wrist_angle_history = []     # 右手腕角度 (判斷轉動，摩托車/閃光車)
        self.l_hand_size_history = []     # 左手大小歷史
        self.index_flap_history = []      # 食指開合歷史 (判斷物/東西)
        self.pinch_history = []           # 捏合歷史 (判斷包水餃)
        self.wrist_dist_history_z = []    # 雙手 Z 軸相對距離 (判斷前後交替)
        self.l_wrist_angle_history = []   # 左手腕角度歷史

        # 新增：時間戳記相關變數（解決 FPS 問題）=====
        self.left_wrist_pos_history = []   # ⭐ 格式：[(timestamp, x, y, z), ...]
        self.right_wrist_pos_history = []  # ⭐
        
        # 新增：AI 模型需要的變數 =====
        self.prev_left_wrist_y = None   # ⭐ 用於計算動態特徵
        self.prev_right_wrist_y = None  # ⭐
        
        # MediaPipe 的臉部與姿態點位 index
        self.FACE_IDS = [33, 133, 362, 263, 1, 61, 291, 199]  # ⭐ 8 個臉部關鍵點
        self.POSE_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23]  # ⭐ 10 個姿態關鍵點

    def extract_features(self, pose_results, hand_results, face_results, img_w, img_h):
        """
        提取特徵
    
        參數:
            pose_results: MediaPipe Pose 的結果
            hand_results: MediaPipe Hands 的結果
            face_results: MediaPipe Face Mesh 的結果  # ⭐ 新增
            img_w: 影像寬度（像素）  # ⭐ 新增
            img_h: 影像高度（像素）  # ⭐ 新增
    
        回傳:
            current_features: dict，原有的布林特徵
            ai_tensor: list，218 維的數值特徵給 LSTM 用  # ⭐ 新增
        """

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
            'dist_HAND_8_FACE_CHEEK_R': 99.0,      
            'dist_HAND_12_FACE_CHEEK_R': 99.0,     
            'dist_HAND_16_FACE_CHEEK_R': 99.0,     
            'dist_LEFT_HAND_8_FACE_CHEEK_L': 99.0, 
            'dist_LEFT_HAND_8_TEMPLE_L': 99.0,     
            
            # 📌 「早上」專屬特徵 (舊有功能保留)
            'dist_START_RIGHT_HAND_0_POSE_HEAD': 99.0,
            'move_downwards_RIGHT_HAND': False,
            'move_upwards_RIGHT_HAND': False,             
            'palm_facing_in_RIGHT_HAND': False,
            'palm_facing_down_HAND': False, 
            'palm_facing_out_RIGHT_HAND': False,          
            
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
            'is_Y_shape_HAND': False,                     
            'is_Y_shape_RIGHT_HAND': False,               
            'is_curved_down_RIGHT_HAND': False,           
            'is_middle_finger_extended_HAND': False,      
            'is_middle_finger_extended_RIGHT_HAND': False,
            'is_ring_finger_extended_HAND': False,        
            'is_ring_finger_extended_RIGHT_HAND': False,
            'is_pinky_extended_HAND': False,              
            'is_pinky_extended_RIGHT_HAND': False,
            'is_index_pinky_extended_HAND': False,        
            'is_index_pinky_extended_RIGHT_HAND': False,
            'is_thumb_up_RIGHT_HAND': False,              
            'is_thumb_down_RIGHT_HAND': False,            
            
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
            'detect_circle_RIGHT_HAND': False,            
            'detect_tap_HAND': False,
            'detect_small_move_outwards_RIGHT_HAND': False, 
            
            # 🟢 新增：其他動態特徵 (V分類與再見)
            'detect_scoop_HAND': False,
            'detect_scoop_RIGHT_HAND': False,
            'detect_swipe_HAND_forward': False,
            'detect_finger_bend_repeat_HAND_8': False,
            'detect_move_HAND': False,
            'detect_wave_HAND_horizontal': False, 
            
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
            'is_in_front_RIGHT_HAND_LEFT_HAND': False,    
            'detect_small_move_apart_repeat_BOTH_HANDS': False, 
            'dist_RIGHT_HAND_20_LEFT_HAND_20': 99.0,      
            'dist_RIGHT_HAND_12_LEFT_HAND_12': 99.0,      
            'dist_LEFT_HAND_0_POSE_CHEST': 99.0,          
            'palms_facing_in_BOTH_HANDS': False,          
            'palm_facing_in_LEFT_HAND': False,            
            'palm_facing_up_LEFT_HAND': False,            
            'is_index_BOTH_HANDS': False,                 
            'detect_draw_square_BOTH_HANDS': False,       
            'detect_roof_shape_BOTH_HANDS': False,        
            'tips_up_BOTH_HANDS': False,                  
            'is_Y_shape_BOTH_HANDS': False,               
            'is_flat_LEFT_HAND': False,                   
            'is_flat_BOTH_HANDS': False,                  
            'is_middle_finger_extended_BOTH_HANDS': False,
            'is_ring_finger_extended_BOTH_HANDS': False,  
            'is_index_pinky_extended_LEFT_HAND': False,   
            
            # 🟢 新增：說話_A / 說話_B 專用
            'dist_HAND_8_MOUTH': 99.0,          
            'dist_HAND_4_MOUTH': 99.0,           
            'palm_facing_left_RIGHT_HAND': False, 
            'vector_align_HAND_8_LEFT_AXIS': False, 
            'detect_small_move_HAND_z': False,    
            'detect_wave_fingers_HAND': False,    

            # 🟢 新增：寫 專用
            'dist_RIGHT_HAND_8_LEFT_HAND_0': 99.0,  
            'detect_writing_motion_HAND': False,      

            # 🟢 新增：想 / 想到 專用
            'dist_HAND_8_TEMPLE_R': 99.0,        
            'detect_finger_rotate_HAND_8': False, 
            'detect_head_tilt_back': False,        

            # 🟢 新增：等_A / 等_B 專用
            'is_four_fingers_closed_HAND': False,  
            'is_ILY_shape_HAND': False,             
            'dist_HAND_0_POSE_SHOULDER_R': 99.0,   

            # 🟢 新增：再見 專用
            'detect_index_open_to_bend_HAND': False, 
            'detect_index_bending_HAND': False,       

            # 🟢 新增：來 專用
            'is_index_up': False,                    

            # 🟢 新增：精確雙手距離 / 左手形狀
            'dist_RIGHT_HAND_8_LEFT_HAND_8': 99.0,  
            'is_flat_palm_LEFT_HAND': False,          

            # 🟢 新增：物/東西、路 專用
            'dist_HAND_0_POSE_SHOULDER_L': 99.0,     
            'dist_RIGHT_HAND_0_POSE_CHEST': 99.0,    
            'is_ILY_shape_LEFT_HAND': False,          

            # 🟢 新增：種類 專用
            'dist_RIGHT_HAND_20_LEFT_HAND_0': 99.0,  

            # 🟢 新增：結婚/夫婦 專用
            'dist_LEFT_HAND_4_RIGHT_HAND_20': 99.0,  
            'move_upwards_LEFT_HAND': False,          

            # 🟢 新增：常用別名（整合全 Excel 缺失）
            'detect_swipe_HAND_up': False,            
            'is_fist_RIGHT_HAND': False,              
            'is_index_pointing_RIGHT_HAND': False,    
            'vector_align_HAND_PALM_INWARD': False,   
            'vector_align_HAND_PALM_UPWARD': False,   
            'vector_align_HAND_PALM_OPPOSITE': False, 
            'dist_SHOULDER_L_SHOULDER_R': 0.0,        
            'dist_HAND_4_8': 99.0,                   
            'dist_HAND_POSE_NAVEL': 99.0,             
            'dist_HAND_POSE_SHOULDER': 99.0,          
            'dist_HAND_POSE_SHOULDER_RIGHT': 99.0,    
            'dist_RIGHT_HAND_LEFT_UPPER_ARM': 99.0,   
            'dist_RIGHT_HAND_BACK_LEFT_PALM': 99.0,   
            'dist_RIGHT_HAND_LEFT_FOREARM': 99.0,     

            # 🟢 新增：需要真正計算的動作/形狀特徵
            'is_O_shape_HAND': False,                 
            'is_claw_HAND': False,                    
            'is_fingers_bent_HAND': False,            
            'is_pinch_middle_HAND': False,            
            'is_V_shape_LEFT_HAND': False,            
            'is_fist_LEFT_HAND': False,               
            'detect_swipe_HAND_down': False,          
            'detect_swipe_HAND_outward': False,       
            'detect_vibrate_HAND': False,             
            'detect_flick_HAND': False,               
            'detect_arc_HAND': False,                 
            'vector_align_HAND_8_UP_AXIS': 0.0,      
            'dist_RIGHT_HAND_8_LEFT_HAND_V_GAP': 99.0, 
            'dist_RIGHT_HAND_LEFT_HAND_PALM': 99.0,  
            'dist_HAND_MIN_BODY_PART': 99.0,         
            'dist_HANDS_POSE_NECK': 99.0,             

            # ==========================================
            # 🌟 新增：針對最後一批詞彙擴充的專屬特徵
            # ==========================================
            'dist_RIGHT_HAND_4_FACE_FOREHEAD': 99.0,
            'dist_RIGHT_HAND_0_FACE_SIDE': 99.0,
            'dist_RIGHT_HAND_0_LEFT_CHEST': 99.0,
            'dist_RIGHT_HAND_0_LEFT_ELBOW': 99.0,
            'dist_RIGHT_HAND_0_LEFT_HAND_0': 99.0,
            'dist_RIGHT_HAND_TIPS_LEFT_HAND_0': 99.0,
            
            'move_right_RIGHT_HAND': False,    
            'move_left_RIGHT_HAND': False,     
            'move_forwards_RIGHT_HAND': False, 
            'move_backwards_RIGHT_HAND': False,
            'palm_facing_down_RIGHT_HAND': False, 
            'palm_facing_up_RIGHT_HAND': False,   
            'is_open_RIGHT_HAND': False, 
            'is_L_shape_RIGHT_HAND': False, 
            'is_C_shape_RIGHT_HAND': False, 
            'is_curved_RIGHT_HAND': False,  
            'is_U_shape_RIGHT_HAND': False,
            'is_hook_RIGHT_HAND': False,
            'is_double_hook_RIGHT_HAND': False,
            'is_airplane_shape_RIGHT_HAND': False,
            'is_circle_shape_RIGHT_HAND': False,
            'is_K_shape_RIGHT_HAND': False,

            'detect_swipe_RIGHT_HAND_vertical': False,
            'detect_wrist_rotation_RIGHT_HAND': False,
            'detect_index_flap_RIGHT_HAND': False,
            'detect_repeating_pinch_RIGHT_HAND': False,

            'is_below_RIGHT_HAND_LEFT_HAND': False, 
            'move_closer_RIGHT_HAND_LEFT_HAND': False, 
            'palms_facing_each_other': False, 

            'is_open_BOTH_HANDS': False,
            'is_open_LEFT_HAND': False,
            'is_L_shape_BOTH_HANDS': False,
            'is_L_shape_LEFT_HAND': False,
            'is_C_shape_LEFT_HAND': False,
            'is_double_hook_BOTH_HANDS': False,
            'is_K_shape_BOTH_HANDS': False,
            'is_pinch_LEFT_HAND': False,
            'fingers_pointing_down_LEFT_HAND': False,
            'palm_facing_down_LEFT_HAND': False,
            'palm_facing_down_BOTH_HANDS': False,
            'palm_facing_up_BOTH_HANDS': False,
            'move_forwards_BOTH_HANDS': False,
            'move_backwards_BOTH_HANDS': False,
            'detect_circle_BOTH_HANDS': False,
            'detect_alternating_circle_BOTH_HANDS': False,
            'detect_wrist_rotation_BOTH_HANDS': False,
            'move_alternating_depth_LEFT_HAND': False,
            'move_alternating_depth_BOTH_HANDS': False,

            # 🟢 新增：上下車 / 停塞車 / 飆車 專用
            'dist_RIGHT_HAND_0_POSE_HIP': 99.0,           # 右手腕到右臀距離（上下車骨盆位置）
            'detect_body_sway_large_amplitude': False,     # 身體大幅左右搖擺（飆車）
            'is_four_fingers_closed_BOTH_HANDS': False,    # 雙手四指伸直拇指收（停車/塞車/車禍）

            'palm_facing_in_BOTH_HANDS': False,
            'move_horizontal_apart_BOTH_HANDS': False,
            'palms_down_BOTH_HANDS': False,
            'detect_swipe_outwards_HAND': False,
            'vector_align_RIGHT_HAND_8_DOWN_AXIS': 0.0,
            'is_claw_RIGHT_HAND': False,
            'detect_swipe_RIGHT_HAND_horizontal': False,
            'detect_wiggle_BOTH_HANDS': False,
            'move_downwards_BOTH_HANDS': False,
            'tips_down_BOTH_HANDS': False,
            'detect_slap_HAND': False,
            'dist_HAND_FACE_234': 99.0,
            'dist_HAND_FACE_454': 99.0,
            'dist_START_HAND_POSE_CHEST': 99.0,
            'dist_RIGHT_HAND_8_POSE_NOSE': 99.0,
            'dist_RIGHT_HAND_0_RIGHT_SHOULDER': 99.0,
            'is_ring_finger_extended_LEFT_HAND': False, # 確保沒偵測到雙手時也有這個鍵值
        }
        current_features = FEATURE_REGISTRY.create_feature_dict(current_features)

        if not pose_results or not pose_results.pose_landmarks:
            return current_features, FEATURE_REGISTRY.empty_ai_tensor()

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

        # 🌟 新增：針對新詞彙設定身體部位參考點
        forehead_point = Point2D(nose.x, nose.y - 0.08)
        left_chest_point = Point2D((shoulder_l.x * 2 + chest_point.x)/3, chest_point.y)
        face_side_point = right_ear

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
                current_features['is_open_RIGHT_HAND'] = current_features['is_open_HAND']
                current_features['is_L_shape_RIGHT_HAND'] = current_features['is_L_shape_HAND']
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

                # 🌟 新增：詞彙動作專屬手型判定 (U、鉤、K、飛機等)
                current_features['is_U_shape_RIGHT_HAND'] = index_open and middle_open and not thumb_open and not ring_open and not pinky_open
                current_features['is_hook_RIGHT_HAND'] = (not index_open) and (calculate_dist_3d(hand_lms[8], hand_lms[0]) > calculate_dist_3d(hand_lms[6], hand_lms[0]))
                current_features['is_double_hook_RIGHT_HAND'] = not index_open and not middle_open and not thumb_open and not ring_open
                current_features['is_airplane_shape_RIGHT_HAND'] = thumb_open and index_open and pinky_open and not middle_open and not ring_open
                current_features['is_K_shape_RIGHT_HAND'] = thumb_open and index_open and middle_open and not ring_open and not pinky_open

                dist_thumb_index = calculate_dist_3d(thumb_tip, index_tip)
                current_features['is_pinch_HAND'] = dist_thumb_index < 0.05
                current_features['is_pinch_RIGHT_HAND'] = current_features['is_pinch_HAND']
                current_features['is_pinch_all_HAND'] = current_features['is_pinch_HAND'] and not middle_open
                current_features['is_C_shape_HAND'] = thumb_open and index_open and not middle_open and dist_thumb_index > 0.05
                current_features['is_C_shape_RIGHT_HAND'] = current_features['is_C_shape_HAND']
                current_features['is_crossed_fingers_HAND'] = index_open and middle_open and calculate_dist_3d(hand_lms[8], hand_lms[12]) < 0.03

                # 手心判定
                current_features['palm_facing_in_RIGHT_HAND'] = abs(thumb_mcp.x - pinky_mcp.x) < 0.30
                is_horizontal = abs(thumb_mcp.y - pinky_mcp.y) < 0.1
                is_flat_z = abs(thumb_mcp.z - pinky_mcp.z) < 0.08
                current_features['palm_facing_down_HAND'] = is_horizontal and is_flat_z
                current_features['palm_facing_down_RIGHT_HAND'] = current_features['palm_facing_down_HAND']
                current_features['palm_facing_out_RIGHT_HAND'] = hand_lms[17].z > hand_lms[5].z 
                
                # 掌心朝上 (沿用原本的 vector_align_HAND_PALM_UPWARD)
                current_features['vector_align_HAND_PALM_UPWARD'] = (hand_lms[9].y > hand_lms[0].y)
                current_features['palm_facing_up_RIGHT_HAND'] = current_features['vector_align_HAND_PALM_UPWARD']

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

                # 🌟 新增：針對額頭、側臉、左胸的距離計算
                current_features['dist_RIGHT_HAND_4_FACE_FOREHEAD'] = round(calculate_dist_2d(hand_lms[4], forehead_point) / shoulder_width, 3)
                current_features['dist_RIGHT_HAND_0_FACE_SIDE'] = round(calculate_dist_2d(wrist, face_side_point) / shoulder_width, 3)
                current_features['dist_RIGHT_HAND_0_LEFT_CHEST'] = round(calculate_dist_2d(wrist, left_chest_point) / shoulder_width, 3)
                # dist_RIGHT_HAND_8_POSE_NOSE 可以直接用 dist_HAND_8_FACE_1 代替
                current_features['dist_RIGHT_HAND_8_POSE_NOSE'] = current_features.get('dist_HAND_8_FACE_1', 99.0)
                # dist_RIGHT_HAND_0_RIGHT_SHOULDER 可以用 dist_HAND_POSE_SHOULDER_RIGHT 代替
                current_features['dist_RIGHT_HAND_0_RIGHT_SHOULDER'] = current_features.get('dist_HAND_POSE_SHOULDER_RIGHT', 99.0)

                # 🟢 新增：右手腕到右臀距離（上下車骨盆點 = p_lms[24]）
                hip_r = Point2D(p_lms[24].x, p_lms[24].y)
                current_features['dist_RIGHT_HAND_0_POSE_HIP'] = round(
                    calculate_dist_2d(wrist, hip_r) / shoulder_width, 3
                )

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
                
                # 🌟 新增：向左 / 向右 移動判定
                if len(self.wrist_x_history) >= 10:
                    x_diff = self.wrist_x_history[-1] - self.wrist_x_history[0]
                    current_features['move_right_RIGHT_HAND'] = x_diff > 0.04
                    current_features['move_left_RIGHT_HAND'] = x_diff < -0.04

                # 🌟 新增：向外小幅度拉開
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

                # 🌟 新增：利用視覺手掌大小判斷 Z 軸 (前後移動，高鐵_A 等)
                current_size = get_hand_size(hand_lms)
                self.hand_size_history.append(current_size)
                if len(self.hand_size_history) > 15: self.hand_size_history.pop(0)
                if len(self.hand_size_history) >= 10:
                    size_diff = self.hand_size_history[-1] - self.hand_size_history[0]
                    current_features['move_forwards_RIGHT_HAND'] = size_diff > 0.012
                    current_features['move_backwards_RIGHT_HAND'] = size_diff < -0.012

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
                    
                    # 🌟 新增：垂直滑動 (公車_B)
                    current_features['detect_swipe_RIGHT_HAND_vertical'] = (y_range > 0.1) and (x_range < 0.08)

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
                    
                # 🌟 新增：手腕轉動 (摩托車/閃光車) 計算食指根部與手腕的角度變異
                w_dx = hand_lms[5].x - hand_lms[0].x
                w_dy = hand_lms[5].y - hand_lms[0].y
                current_angle = np.arctan2(w_dy, w_dx)
                self.wrist_angle_history.append(current_angle)
                if len(self.wrist_angle_history) > 15: self.wrist_angle_history.pop(0)
                if len(self.wrist_angle_history) >= 10:
                    current_features['detect_wrist_rotation_RIGHT_HAND'] = np.var(self.wrist_angle_history[-10:]) > 0.03

                # 🌟 新增：食指上下擺動 (物/東西)
                self.index_flap_history.append(index_open)
                if len(self.index_flap_history) > 15: self.index_flap_history.pop(0)
                if len(self.index_flap_history) >= 10:
                    flaps = sum(1 for i in range(1, len(self.index_flap_history)) if self.index_flap_history[i] != self.index_flap_history[i-1])
                    current_features['detect_index_flap_RIGHT_HAND'] = flaps >= 2

                # 🌟 新增：重複捏合 (水餃)
                self.pinch_history.append(current_features['is_pinch_HAND'])
                if len(self.pinch_history) > 15: self.pinch_history.pop(0)
                if len(self.pinch_history) >= 10:
                    pinches = sum(1 for i in range(1, len(self.pinch_history)) if self.pinch_history[i] != self.pinch_history[i-1])
                    current_features['detect_repeating_pinch_RIGHT_HAND'] = pinches >= 2

                # 點擊判定 (Tap)
                if len(self.chest_dist_history) >= 15:
                    hist = self.chest_dist_history
                    min_idx = np.argmin(hist)
                    if 3 < min_idx < len(hist) - 3:
                        if (hist[0] - hist[min_idx] > 0.015) and (hist[-1] - hist[min_idx] > 0.015):
                            current_features['detect_tap_HAND'] = True

            # =========================================================
            # 🌟 1.5 新增單手特徵計算 (包含原本的說話/想到/等...)
            # =========================================================
            if hand_lms is not None:

                # ── 掌心朝左判定 ──
                current_features['palm_facing_left_RIGHT_HAND'] = (hand_lms[2].x < hand_lms[17].x)
                current_features['vector_align_HAND_8_LEFT_AXIS'] = current_features['palm_facing_left_RIGHT_HAND']

                # ── 嘴巴距離 (說話_A / 說話_B) ──
                current_features['dist_HAND_8_MOUTH'] = round(calculate_dist_2d(hand_lms[8], mouth_point) / shoulder_width, 3)
                current_features['dist_HAND_4_MOUTH'] = round(calculate_dist_2d(hand_lms[4], mouth_point) / shoulder_width, 3)

                # ── 說話_A：小幅度前後移動（Z 軸振盪） ──
                self.hand_z_small_history.append(hand_lms[8].z)
                if len(self.hand_z_small_history) > 15:
                    self.hand_z_small_history.pop(0)
                if len(self.hand_z_small_history) >= 10:
                    z_vals = self.hand_z_small_history
                    z_range = max(z_vals) - min(z_vals)
                    current_features['detect_small_move_HAND_z'] = (0.02 < z_range <= 0.08)

                # ── 說話_B：五指輕微參差擺動 ──
                current_features['detect_wave_fingers_HAND'] = (
                    current_features['is_open_HAND'] and
                    current_features['detect_small_swipe_HAND_horizontal']
                )

                # ── 太陽穴距離（想 / 想到） ──
                temple_r = right_ear  
                current_features['dist_HAND_8_TEMPLE_R'] = round(calculate_dist_2d(hand_lms[8], temple_r) / shoulder_width, 3)

                # ── 想：食指在太陽穴旋轉 ──
                self.rotate_history.append((hand_lms[8].x, hand_lms[8].y))
                if len(self.rotate_history) > 20: self.rotate_history.pop(0)
                if len(self.rotate_history) >= 15:
                    rx = [p[0] for p in self.rotate_history]
                    ry = [p[1] for p in self.rotate_history]
                    rx_range = max(rx) - min(rx)
                    ry_range = max(ry) - min(ry)
                    current_features['detect_finger_rotate_HAND_8'] = (
                        rx_range > 0.03 and ry_range > 0.03 and abs(rx_range - ry_range) < 0.04
                    )

                # ── 想到/思念：頭後仰 ──
                self.nose_y_history.append(nose.y)
                if len(self.nose_y_history) > 20: self.nose_y_history.pop(0)
                if len(self.nose_y_history) >= 10:
                    head_moved_up = self.nose_y_history[0] - self.nose_y_history[-1] > 0.03
                    current_features['detect_head_tilt_back'] = head_moved_up

                # ── 等_A：四指（食中無小）伸直、拇指彎曲收起 ──
                current_features['is_four_fingers_closed_HAND'] = (
                    index_open and middle_open and ring_open and pinky_open and not thumb_open
                )

                # ── 等_B/守：拇指+食指+小指伸直（ILY 手勢） ──
                current_features['is_ILY_shape_HAND'] = (
                    thumb_open and index_open and pinky_open and not middle_open and not ring_open
                )

                # ── 守/等_B：手腕到右肩距離 ──
                current_features['dist_HAND_0_POSE_SHOULDER_R'] = round(calculate_dist_2d(hand_lms[0], shoulder_r) / shoulder_width, 3)

                # ── 寫字動作：水平小幅擺動 + 整體往下移動 ──
                self.writing_y_history.append(hand_lms[0].y)
                if len(self.writing_y_history) > 25: self.writing_y_history.pop(0)
                if len(self.writing_y_history) >= 15:
                    wy = self.writing_y_history
                    wy_drift_down = wy[-1] - wy[0] > 0.02     
                    current_features['detect_writing_motion_HAND'] = (
                        wy_drift_down and current_features['detect_small_swipe_HAND_horizontal']
                    )

                # ── 再見：雙手食指伸直→分開→彎曲 ──
                self.index_bend_history.append(index_open)
                if len(self.index_bend_history) > 30: self.index_bend_history.pop(0)

                # ── 全 Excel 常用別名 ──
                current_features['detect_swipe_HAND_up']          = current_features['detect_scoop_HAND']
                current_features['is_fist_RIGHT_HAND']            = current_features['is_fist_HAND']
                current_features['is_index_pointing_RIGHT_HAND']  = current_features['is_index_pointing_HAND']
                current_features['vector_align_HAND_PALM_INWARD'] = current_features['palm_facing_in_RIGHT_HAND']
                current_features['vector_align_HAND_8_UP_AXIS']   = round(-current_features['vector_align_HAND_8_DOWN_AXIS'], 3)

                # ── 物/東西：右手腕到左肩距離 ──
                current_features['dist_HAND_0_POSE_SHOULDER_L'] = round(calculate_dist_2d(hand_lms[0], shoulder_l) / shoulder_width, 3)

                # ── 右手腕到胸口距離（別名） ──
                current_features['dist_RIGHT_HAND_0_POSE_CHEST'] = current_features['dist_HAND_0_POSE_CHEST']

                # ── 拇指尖到食指尖距離 ──
                current_features['dist_HAND_4_8'] = round(calculate_dist_3d(hand_lms[4], hand_lms[8]), 3)

                # ── 肩寬（供 A_003 大） ──
                current_features['dist_SHOULDER_L_SHOULDER_R'] = round(shoulder_width, 3)

                # ── 手腕到肚臍近似 ──
                navel_approx = Point2D(chest_point.x, chest_point.y + 0.20)
                current_features['dist_HAND_POSE_NAVEL'] = round(calculate_dist_2d(hand_lms[0], navel_approx) / shoulder_width, 3)

                # ── 右肩距離（別名） ──
                current_features['dist_HAND_POSE_SHOULDER']       = current_features['dist_HAND_0_POSE_SHOULDER_R']
                current_features['dist_HAND_POSE_SHOULDER_RIGHT'] = current_features['dist_HAND_0_POSE_SHOULDER_R']

                # ── 向下撥動 ──
                current_features['detect_swipe_HAND_down'] = current_features['move_downwards_RIGHT_HAND']

                # ── 向外撥動（X 軸向右移動）──
                if len(self.wrist_x_history) >= 10:
                    x_diff = self.wrist_x_history[-1] - self.wrist_x_history[0]
                    current_features['detect_swipe_HAND_outward'] = x_diff > 0.06

                # ── 抖動（Z 軸高頻振盪，冷/痛）──
                self.vibrate_z_history.append(hand_lms[0].z)
                if len(self.vibrate_z_history) > 15: self.vibrate_z_history.pop(0)
                if len(self.vibrate_z_history) >= 10:
                    z_std = float(np.std(self.vibrate_z_history))
                    current_features['detect_vibrate_HAND'] = z_std > 0.012

                # ── 彈指（少）：拇指+中指短暫靠近 ──
                dist_4_12 = calculate_dist_3d(hand_lms[4], hand_lms[12])
                current_features['is_pinch_middle_HAND'] = dist_4_12 < 0.05
                current_features['detect_flick_HAND'] = (dist_4_12 < 0.06 and not index_open and not ring_open)

                # ── 弧形移動（整天）：X 位移大、Y 位移中等 ──
                if len(self.index_history) >= 20:
                    xs = [p[0] for p in self.index_history]
                    ys = [p[1] for p in self.index_history]
                    arc_x = max(xs) - min(xs)
                    arc_y = max(ys) - min(ys)
                    current_features['detect_arc_HAND'] = arc_x > 0.15 and 0.03 < arc_y < 0.15

                # ── 特殊手型 ──
                current_features['is_O_shape_HAND'] = (
                    current_features['dist_HAND_4_8'] < 0.04 and not middle_open and not ring_open and not pinky_open
                )
                current_features['is_circle_shape_RIGHT_HAND'] = current_features['is_O_shape_HAND'] # 🌟新詞彙用
                
                current_features['is_claw_HAND'] = (
                    not index_open and not middle_open and not ring_open and not pinky_open and
                    not current_features['is_fist_HAND'] and
                    calculate_dist_3d(hand_lms[8], hand_lms[0]) > calculate_dist_3d(hand_lms[6], hand_lms[0]) * 0.8
                )
                
                current_features['is_fingers_bent_HAND'] = (not index_open and not middle_open and not current_features['is_fist_HAND'])
                current_features['is_curved_RIGHT_HAND'] = current_features['is_fingers_bent_HAND'] or current_features['is_claw_HAND']

            # =========================================================
            # 🌟 2. 雙手特徵 (平安/平靜等 + 新增 V分類互動 + 新詞彙擴充)
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
                    
                    current_features['is_fist_LEFT_HAND'] = l_fist
                    current_features['is_fist_BOTH_HANDS'] = current_features['is_fist_HAND'] and l_fist
                    
                    # 🟢 新增：左手單獨形狀擴充
                    current_features['is_flat_LEFT_HAND'] = (l_index_open and l_middle_open and l_ring_open and l_pinky_open)
                    current_features['is_index_pinky_extended_LEFT_HAND'] = not l_thumb_open and l_index_open and not l_middle_open and not l_ring_open and l_pinky_open

                    # 🌟 雙手形狀整合判定 (新詞彙擴充)
                    current_features['is_open_LEFT_HAND'] = current_features['is_flat_LEFT_HAND']
                    current_features['is_open_BOTH_HANDS'] = current_features['is_open_HAND'] and current_features['is_open_LEFT_HAND']
                    
                    current_features['is_C_shape_LEFT_HAND'] = l_thumb_open and l_index_open and not l_middle_open and not l_ring_open
                    current_features['is_L_shape_LEFT_HAND'] = l_thumb_open and l_index_open and not l_middle_open and not l_ring_open and not l_pinky_open
                    current_features['is_L_shape_BOTH_HANDS'] = current_features['is_L_shape_HAND'] and current_features['is_L_shape_LEFT_HAND']
                    
                    current_features['is_double_hook_LEFT_HAND'] = not l_index_open and not l_middle_open and not l_thumb_open
                    current_features['is_double_hook_BOTH_HANDS'] = current_features['is_double_hook_RIGHT_HAND'] and current_features['is_double_hook_LEFT_HAND']
                    
                    current_features['is_K_shape_LEFT_HAND'] = l_thumb_open and l_index_open and l_middle_open and not l_ring_open
                    current_features['is_K_shape_BOTH_HANDS'] = current_features['is_K_shape_RIGHT_HAND'] and current_features['is_K_shape_LEFT_HAND']
                    
                    current_features['is_pinch_LEFT_HAND'] = calculate_dist_3d(l_lms[4], l_lms[8]) < 0.05

                    # 🟢 新增：雙手同步形狀
                    current_features['is_Y_shape_BOTH_HANDS'] = current_features['is_Y_shape_RIGHT_HAND'] and (l_thumb_open and l_pinky_open and not l_index_open and not l_middle_open and not l_ring_open)
                    current_features['is_flat_BOTH_HANDS'] = current_features['is_flat_RIGHT_HAND'] and current_features['is_flat_LEFT_HAND']
                    current_features['is_index_BOTH_HANDS'] = current_features['is_index_RIGHT_HAND'] and current_features['is_index_LEFT_HAND']
                    current_features['is_middle_finger_extended_BOTH_HANDS'] = current_features['is_middle_finger_extended_RIGHT_HAND'] and (not l_thumb_open and not l_index_open and l_middle_open and not l_ring_open and not l_pinky_open)
                    current_features['is_ring_finger_extended_BOTH_HANDS'] = current_features['is_ring_finger_extended_RIGHT_HAND'] and (not l_thumb_open and not l_index_open and not l_middle_open and l_ring_open and not l_pinky_open)

                    # 跑_B：放寬五指張開的條件
                    current_features['is_open5_BOTH_HANDS'] = current_features['is_open_HAND'] and (l_open_count >= 3)
                    current_features['is_flat_down_BOTH_HANDS'] = current_features['is_open5_BOTH_HANDS']

                    # =========================================================
                    # hands_facing_BOTH_HANDS — 掌心相對
                    # =========================================================
                    r_thumb_faces_left  = r_lms[2].x < r_lms[17].x   
                    l_thumb_faces_right = l_lms[2].x > l_lms[17].x   
                    current_features['hands_facing_BOTH_HANDS'] = r_thumb_faces_left and l_thumb_faces_right
                    current_features['palms_facing_each_other'] = current_features['hands_facing_BOTH_HANDS']
                    
                    # 🟢 新增：雙手掌心朝向
                    current_features['palm_facing_in_LEFT_HAND'] = abs(l_lms[2].x - l_lms[17].x) < 0.30
                    current_features['palms_facing_in_BOTH_HANDS'] = current_features['palm_facing_in_RIGHT_HAND'] and current_features['palm_facing_in_LEFT_HAND']
                    
                    l_is_horizontal = abs(l_lms[2].y - l_lms[17].y) < 0.1
                    l_is_flat_z = abs(l_lms[2].z - l_lms[17].z) < 0.08
                    current_features['palm_facing_down_LEFT_HAND'] = l_is_horizontal and l_is_flat_z
                    current_features['palm_facing_down_BOTH_HANDS'] = current_features['palm_facing_down_RIGHT_HAND'] and current_features['palm_facing_down_LEFT_HAND']
                    
                    current_features['palm_facing_up_LEFT_HAND'] = l_wrist.y > l_lms[12].y and not l_is_flat_z
                    current_features['palm_facing_up_BOTH_HANDS'] = current_features['palm_facing_up_RIGHT_HAND'] and current_features['palm_facing_up_LEFT_HAND']
                    
                    current_features['tips_up_BOTH_HANDS'] = (r_wrist.y > r_lms[12].y) and (l_wrist.y > l_lms[12].y)
                    current_features['fingers_pointing_down_LEFT_HAND'] = l_lms[8].y > l_wrist.y

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
                    # =========================================================
                    if len(self.r_wrist_y_history) >= 10 and len(self.l_wrist_y_history) >= 10:
                        r_y_range = max(self.r_wrist_y_history) - min(self.r_wrist_y_history)
                        l_y_range = max(self.l_wrist_y_history) - min(self.l_wrist_y_history)
                        both_wrists_moving = (r_y_range > 0.04) and (l_y_range > 0.04)
                        
                        current_features['detect_alternate_swing_BOTH_HANDS'] = (
                            not current_features['is_open5_BOTH_HANDS'] and both_wrists_moving
                        )

                    # =========================================================
                    # 🔧 修正：detect_wave_BOTH_HANDS_vertical
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

                    # 🟢 右手食指尖(8) 到 左手手腕(0) 的距離
                    dist_r8_l0 = calculate_dist_2d(r_lms[8], l_lms[0]) / shoulder_width
                    current_features['dist_RIGHT_HAND_8_LEFT_HAND_0'] = round(dist_r8_l0, 3)
                    current_features['dist_RIGHT_HAND_TIPS_LEFT_HAND_0'] = current_features['dist_RIGHT_HAND_8_LEFT_HAND_0']

                    # 🟢 右手食指尖(8) 到 左手食指尖(8) 的距離
                    dist_r8_l8 = calculate_dist_2d(r_lms[8], l_lms[8]) / shoulder_width
                    current_features['dist_RIGHT_HAND_8_LEFT_HAND_8'] = round(dist_r8_l8, 3)

                    # 🟢 左手五指伸直攤平
                    current_features['is_flat_palm_LEFT_HAND'] = (l_index_open and l_middle_open and l_ring_open and l_pinky_open)

                    # 🟢 新增：左手 ILY 手型（路）
                    current_features['is_ILY_shape_LEFT_HAND'] = (l_thumb_open and l_index_open and l_pinky_open and not l_middle_open and not l_ring_open)

                    # 🟢 新增：左手 V 型
                    current_features['is_V_shape_LEFT_HAND'] = (l_index_open and l_middle_open and not l_ring_open and not l_pinky_open and not l_thumb_open)

                    # 🟢 新增：is_ring_finger_extended_LEFT_HAND
                    current_features['is_ring_finger_extended_LEFT_HAND'] = (not l_thumb_open and not l_index_open and not l_middle_open and l_ring_open and not l_pinky_open)

                    # 🟢 新增：vector_align_HAND_PALM_OPPOSITE
                    current_features['vector_align_HAND_PALM_OPPOSITE'] = current_features['hands_facing_BOTH_HANDS']

                    # 🟢 新增：左手腕向上移動
                    if len(self.l_wrist_y_history) >= 10:
                        ly_diff = self.l_wrist_y_history[-1] - self.l_wrist_y_history[0]
                        current_features['move_upwards_LEFT_HAND'] = ly_diff < -0.03

                    # 🟢 雙手部位間距離
                    dist_l4_r20 = calculate_dist_2d(l_lms[4], r_lms[20]) / shoulder_width
                    current_features['dist_LEFT_HAND_4_RIGHT_HAND_20'] = round(dist_l4_r20, 3)

                    dist_r20_l0 = calculate_dist_2d(r_lms[20], l_lms[0]) / shoulder_width
                    current_features['dist_RIGHT_HAND_20_LEFT_HAND_0'] = round(dist_r20_l0, 3)

                    left_upper_arm = Point2D((shoulder_l.x + p_lms[13].x) / 2, (shoulder_l.y + p_lms[13].y) / 2)
                    current_features['dist_RIGHT_HAND_LEFT_UPPER_ARM'] = round(calculate_dist_2d(r_wrist, left_upper_arm) / shoulder_width, 3)

                    left_palm_back = Point2D((l_lms[0].x + l_lms[9].x) / 2, (l_lms[0].y + l_lms[9].y) / 2)
                    current_features['dist_RIGHT_HAND_BACK_LEFT_PALM'] = round(calculate_dist_2d(r_wrist, left_palm_back) / shoulder_width, 3)

                    left_forearm = Point2D((l_lms[0].x + p_lms[13].x) / 2, (l_lms[0].y + p_lms[13].y) / 2)
                    current_features['dist_RIGHT_HAND_LEFT_FOREARM'] = round(calculate_dist_2d(r_wrist, left_forearm) / shoulder_width, 3)

                    current_features['dist_RIGHT_HAND_LEFT_HAND_PALM'] = round(calculate_dist_2d(r_wrist, l_lms[9]) / shoulder_width, 3)

                    current_features['dist_RIGHT_HAND_8_LEFT_HAND_V_GAP'] = round(calculate_dist_2d(r_lms[8], r_lms[12]) / shoulder_width, 3)

                    neck_point = Point2D((shoulder_l.x + shoulder_r.x) / 2, shoulder_l.y - 0.05)
                    dist_neck_r = calculate_dist_2d(r_wrist, neck_point) / shoulder_width
                    dist_neck_l = calculate_dist_2d(l_wrist, neck_point) / shoulder_width
                    current_features['dist_HANDS_POSE_NECK'] = round(min(dist_neck_r, dist_neck_l), 3)

                    current_features['dist_HAND_MIN_BODY_PART'] = round(min(
                        current_features['dist_HAND_0_POSE_CHEST'],
                        current_features['dist_HAND_8_FACE_1'],
                        current_features['dist_HAND_POSE_NAVEL']
                    ), 3)

                    # 雙手手腕與身體部位距離
                    dist_hands = calculate_dist_2d(r_wrist, l_wrist) / shoulder_width
                    current_features['dist_START_RIGHT_HAND_0_START_LEFT_HAND_0'] = round(dist_hands, 3)
                    current_features['dist_RIGHT_HAND_LEFT_HAND'] = round(dist_hands, 3)
                    current_features['dist_RIGHT_HAND_0_LEFT_HAND_0'] = round(dist_hands, 3)
                    current_features['dist_RIGHT_HAND_LEFT_THUMB'] = round(calculate_dist_2d(r_wrist, l_thumb_tip) / shoulder_width, 3)
                    current_features['dist_LEFT_HAND_FACE_CHIN'] = round(calculate_dist_2d(l_wrist, chin_point) / shoulder_width, 3)
                    
                    current_features['is_above_RIGHT_HAND_LEFT_HAND'] = r_wrist.y < l_wrist.y
                    current_features['is_below_RIGHT_HAND_LEFT_HAND'] = r_wrist.y > l_wrist.y + 0.05
                    
                    left_elbow = Point2D(p_lms[13].x, p_lms[13].y)
                    current_features['dist_RIGHT_HAND_0_LEFT_ELBOW'] = round(calculate_dist_2d(r_wrist, left_elbow) / shoulder_width, 3)
                    
                    # 🟢 新增：雙手綜合距離與前後相對關係
                    current_features['dist_RIGHT_HAND_20_LEFT_HAND_20'] = round(calculate_dist_2d(r_lms[20], l_lms[20]) / shoulder_width, 3)
                    current_features['dist_RIGHT_HAND_12_LEFT_HAND_12'] = round(calculate_dist_2d(r_lms[12], l_lms[12]) / shoulder_width, 3)
                    current_features['dist_LEFT_HAND_0_POSE_CHEST'] = round(calculate_dist_2d(l_wrist, chest_point) / shoulder_width, 3)
                    current_features['dist_LEFT_HAND_8_FACE_CHEEK_L'] = round(calculate_dist_2d(l_lms[8], cheek_l) / shoulder_width, 3)
                    current_features['dist_LEFT_HAND_8_TEMPLE_L'] = round(calculate_dist_2d(l_lms[8], left_ear) / shoulder_width, 3) 
                    current_features['is_in_front_RIGHT_HAND_LEFT_HAND'] = r_wrist.z < (l_wrist.z - 0.05) 
                    
                    current_dist_x = abs(r_wrist.x - l_wrist.x) / shoulder_width

                    # 🟢 再見：雙手食指伸直拉開後彎曲
                    r_index_open_now = is_finger_open(8, 6, r_lms)
                    l_index_open_now = is_finger_open(8, 6, l_lms)
                    if len(self.index_bend_history) >= 20:
                        was_open = any(self.index_bend_history[:10])
                        is_now_bent = not r_index_open_now and not l_index_open_now
                        hands_spread = current_dist_x > 0.5
                        current_features['detect_index_open_to_bend_HAND'] = (was_open and is_now_bent and hands_spread)
                        current_features['detect_index_bending_HAND'] = current_features['detect_index_open_to_bend_HAND']
                    
                    # 🟢 雙手水平拉開與重複拉開 / 靠近
                    self.wrist_dist_history.append(current_dist_x)
                    if len(self.wrist_dist_history) > 20:
                        self.wrist_dist_history.pop(0)
                        
                    if len(self.wrist_dist_history) >= 15:
                        if current_dist_x - min(self.wrist_dist_history) > 0.03:
                            current_features['move_apart_horizontally_RIGHT_HAND_LEFT_HAND'] = True
                        if self.wrist_dist_history[0] - self.wrist_dist_history[-1] > 0.03:
                            current_features['move_closer_RIGHT_HAND_LEFT_HAND'] = True
                        
                        # 學生證：重複拉開
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

                    # 🌟 雙手動態：前後移動與深度交替 (車、高捷)
                    l_size = get_hand_size(l_lms)
                    self.l_hand_size_history.append(l_size)
                    if len(self.l_hand_size_history) > 15: self.l_hand_size_history.pop(0)
                    if len(self.l_hand_size_history) >= 10 and len(self.hand_size_history) >= 10:
                        r_diff = self.hand_size_history[-1] - self.hand_size_history[0]
                        l_diff = self.l_hand_size_history[-1] - self.l_hand_size_history[0]
                        current_features['move_forwards_BOTH_HANDS'] = (r_diff > 0.012) and (l_diff > 0.012)
                        current_features['move_backwards_BOTH_HANDS'] = (r_diff < -0.012) and (l_diff < -0.012)

                    self.wrist_dist_history_z.append(r_wrist.z - l_wrist.z)
                    if len(self.wrist_dist_history_z) > 15: self.wrist_dist_history_z.pop(0)
                    if len(self.wrist_dist_history_z) >= 10:
                        current_features['move_alternating_depth_BOTH_HANDS'] = np.var(self.wrist_dist_history_z[-10:]) > 0.005
                        current_features['move_alternating_depth_LEFT_HAND'] = current_features['move_alternating_depth_BOTH_HANDS']

                    # 🌟 左手腕轉動判定 (閃光車)
                    l_dx = l_lms[5].x - l_lms[0].x
                    l_dy = l_lms[5].y - l_lms[0].y
                    l_current_angle = np.arctan2(l_dy, l_dx)
                    self.l_wrist_angle_history.append(l_current_angle)
                    if len(self.l_wrist_angle_history) > 15: self.l_wrist_angle_history.pop(0)
                    
                    detect_left_rot = len(self.l_wrist_angle_history) >= 10 and np.var(self.l_wrist_angle_history[-10:]) > 0.03
                    current_features['detect_wrist_rotation_BOTH_HANDS'] = current_features['detect_wrist_rotation_RIGHT_HAND'] and detect_left_rot
                    
                    # 雙手畫圓 (腳踏車、划船)
                    current_features['detect_circle_BOTH_HANDS'] = current_features['detect_circle_RIGHT_HAND']
                    current_features['detect_alternating_circle_BOTH_HANDS'] = current_features['detect_circle_RIGHT_HAND']

                    # 🟢 新增：雙手四指伸直拇指收（停車/塞車/車禍）
                    l_thumb_open = is_finger_open(4, 2, l_lms)
                    l_index_open = is_finger_open(8, 6, l_lms)
                    l_middle_open = is_finger_open(12, 10, l_lms)
                    l_ring_open   = is_finger_open(16, 14, l_lms)
                    l_pinky_open  = is_finger_open(20, 18, l_lms)
                    l_four_closed = l_index_open and l_middle_open and l_ring_open and l_pinky_open and not l_thumb_open
                    current_features['is_four_fingers_closed_BOTH_HANDS'] = (
                        current_features['is_four_fingers_closed_HAND'] and l_four_closed
                    )

                    # 🟢 新增：身體大幅左右搖擺（飆車）
                    # 追蹤鼻子 X 軸振盪幅度
                    self.nose_sway_history.append(nose.x)
                    if len(self.nose_sway_history) > 30: self.nose_sway_history.pop(0)
                    if len(self.nose_sway_history) >= 20:
                        sway_range = max(self.nose_sway_history) - min(self.nose_sway_history)
                        current_features['detect_body_sway_large_amplitude'] = sway_range > 0.06

                else: 
                    self.wrist_dist_history.clear()
                    self.l_wrist_y_history.clear()
                    self.r_wrist_x_history.clear()
                    self.l_wrist_x_history.clear()
                    self.wrist_dist_history_z.clear()
                    self.l_hand_size_history.clear()
                    self.l_wrist_angle_history.clear()

        else: 
            # 清除所有歷史軌跡
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
            self.vibrate_z_history.clear()
            self.nose_sway_history.clear()
            
            # 🌟 清除新詞彙的緩衝區
            self.hand_size_history.clear()
            self.wrist_angle_history.clear()
            self.l_hand_size_history.clear()
            self.index_flap_history.clear()
            self.pinch_history.clear()
            self.wrist_dist_history_z.clear()
            self.l_wrist_angle_history.clear()
    
        # 初始化所有特徵為 0
        pose_feat = [0.0] * 30
        face_feat = [0.0] * 24
        l_hand_feat = [0.0] * 63
        r_hand_feat = [0.0] * 63
        l_finger_ang = [0.0] * 15
        r_finger_ang = [0.0] * 15
        elbow_ang = [0.0, 0.0]
        extra_feats = [0.0] * 6
    
        norm_dist = 1.0
        ref_pt = np.zeros(3)
        nose_x = 0.5 * img_w
        l_ear_x = 0.5 * img_w
        r_ear_x = 0.5 * img_w
    
        def get_px_coord(lm):
            return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w])
    
    
        # --- A. 提取姿態特徵 (30 維) ---
        if pose_results and pose_results.pose_landmarks:
            lms = pose_results.pose_landmarks.landmark
            ref_pt = get_px_coord(lms[0])
        
            p11 = get_px_coord(lms[11])
            p12 = get_px_coord(lms[12])
            norm_dist = np.linalg.norm(p11 - p12) + 1e-6
        
            nose_x = get_px_coord(lms[0])[0]
            l_ear_x = get_px_coord(lms[7])[0]
            r_ear_x = get_px_coord(lms[8])[0]
        
            pose_feat = []
            for i in self.POSE_IDS:
                pose_feat += ((get_px_coord(lms[i]) - ref_pt) / norm_dist).tolist()
        
            p13 = get_px_coord(lms[13])
            p14 = get_px_coord(lms[14])
            p15 = get_px_coord(lms[15])
            p16 = get_px_coord(lms[16])
            elbow_ang = [
                angle_between_points(p11, p13, p15) / np.pi,
                angle_between_points(p12, p14, p16) / np.pi
            ]
    
    
        # --- B. 提取臉部特徵 (24 維) ---
        if face_results and face_results.multi_face_landmarks:
            face_feat = []
            for i in self.FACE_IDS:
                lm = face_results.multi_face_landmarks[0].landmark[i]
                face_feat += ((get_px_coord(lm) - ref_pt) / norm_dist).tolist()
    
    
        # --- C. 提取手部特徵 (126 維) + 動態特徵 (6 維) ---
        hand_exist = False
        current_time = time.time()
    
        if hand_results and hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            hand_exist = True
        
            for idx, hand_lms in enumerate(hand_results.multi_hand_landmarks):
                label = hand_results.multi_handedness[idx].classification[0].label
                wrist = hand_lms.landmark[0]
                wrist_pt = get_px_coord(wrist)
            
                feat = []
                for lm in hand_lms.landmark:
                    feat += ((get_px_coord(lm) - wrist_pt) / norm_dist).tolist()
            
                angles = hand_angles(hand_lms.landmark, img_w, img_h)
            
                if label == 'Left':
                    l_hand_feat = feat
                    l_finger_ang = angles
                
                    if self.prev_left_wrist_y is not None:
                        extra_feats[3] = (wrist_pt[1] - self.prev_left_wrist_y) / norm_dist
                    self.prev_left_wrist_y = wrist_pt[1]
                
                    extra_feats[4] = (wrist_pt[0] - nose_x) / norm_dist
                    extra_feats[5] = (wrist_pt[0] - l_ear_x) / norm_dist
                
                    self.left_wrist_pos_history.append((current_time, wrist_pt[0], wrist_pt[1], wrist_pt[2]))
            
                else:
                    r_hand_feat = feat
                    r_finger_ang = angles
                
                    if self.prev_right_wrist_y is not None:
                        extra_feats[0] = (wrist_pt[1] - self.prev_right_wrist_y) / norm_dist
                    self.prev_right_wrist_y = wrist_pt[1]
                
                    extra_feats[1] = (wrist_pt[0] - nose_x) / norm_dist
                    extra_feats[2] = (wrist_pt[0] - r_ear_x) / norm_dist
                
                    self.right_wrist_pos_history.append((current_time, wrist_pt[0], wrist_pt[1], wrist_pt[2]))
    
        if not hand_exist:
            self.prev_left_wrist_y = None
            self.prev_right_wrist_y = None
    
    
    # ========================================
    # ===== 新增：計算手部移動速度（防止過渡動作誤判）=====
    # ========================================
    
        TIME_WINDOW = 1.5
        self.left_wrist_pos_history = [
            (t, x, y, z) for t, x, y, z in self.left_wrist_pos_history
            if (current_time - t) <= TIME_WINDOW
        ]
        self.right_wrist_pos_history = [
            (t, x, y, z) for t, x, y, z in self.right_wrist_pos_history
            if (current_time - t) <= TIME_WINDOW
        ]
    
        left_velocity = 0.0
        if len(self.left_wrist_pos_history) >= 2:
            t1, x1, y1, z1 = self.left_wrist_pos_history[-2]
            t2, x2, y2, z2 = self.left_wrist_pos_history[-1]
            dt = t2 - t1
            if dt > 0:
                left_velocity = np.sqrt((x2-x1)**2 + (y2-y1)**2) / dt
    
        right_velocity = 0.0
        if len(self.right_wrist_pos_history) >= 2:
            t1, x1, y1, z1 = self.right_wrist_pos_history[-2]
            t2, x2, y2, z2 = self.right_wrist_pos_history[-1]
            dt = t2 - t1
            if dt > 0:
                right_velocity = np.sqrt((x2-x1)**2 + (y2-y1)**2) / dt
    
        VELOCITY_THRESHOLD = 1.5
        current_features['is_moving_fast_LEFT'] = left_velocity > VELOCITY_THRESHOLD
        current_features['is_moving_fast_RIGHT'] = right_velocity > VELOCITY_THRESHOLD
    
        # 修正原有的靜態手勢判定
        if 'is_static_LEFT_HAND' in current_features:
            current_features['is_static_LEFT_HAND'] = (
                current_features['is_static_LEFT_HAND'] and 
                not current_features['is_moving_fast_LEFT']
            )
    
        if 'is_static_RIGHT_HAND' in current_features:
            current_features['is_static_RIGHT_HAND'] = (
                current_features['is_static_RIGHT_HAND'] and 
                not current_features['is_moving_fast_RIGHT']
            )
    
        # --- D. 合併所有特徵 (總共 218 維) ---
        ai_tensor = (
            face_feat +
            l_hand_feat +
            r_hand_feat +
            l_finger_ang +
            r_finger_ang +
            pose_feat +
            elbow_ang +
            extra_feats
        )
        if len(ai_tensor) < FEATURE_REGISTRY.tensor_dim:
            ai_tensor = ai_tensor + [0.0] * (FEATURE_REGISTRY.tensor_dim - len(ai_tensor))
        elif len(ai_tensor) > FEATURE_REGISTRY.tensor_dim:
            ai_tensor = ai_tensor[:FEATURE_REGISTRY.tensor_dim]

        return current_features, ai_tensor
