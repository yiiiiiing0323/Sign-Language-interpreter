import bpy
import json
import os
import math

def create_empty_if_not_exists(name, size=0.05):
    """
    檢查場景中是否存在指定名稱的 Empty (十字線) 物件。
    若無則使用 Blender 底層 API 建立，避免 Context 報錯。
    """
    if name not in bpy.data.objects:
        empty_obj = bpy.data.objects.new(name, None)
        empty_obj.empty_display_type = 'PLAIN_AXES'
        empty_obj.empty_display_size = size
        bpy.context.collection.objects.link(empty_obj)
    return bpy.data.objects[name]

def update_sign_word_text(word_string):
    """
    根據檔名自動在 3D 空間中生成立體文字，
    用於前端展示目前正在播放的手語詞彙。
    """
    obj_name = "SignWord_Text"
    if obj_name not in bpy.data.objects:
        text_data = bpy.data.curves.new(name=f"{obj_name}_Data", type='FONT')
        text_obj = bpy.data.objects.new(obj_name, text_data)
        bpy.context.collection.objects.link(text_obj)
        text_obj.location = (0.5, 0, 1.8) 
        text_obj.rotation_euler = (math.radians(90), 0, 0)
        text_data.extrude = 0.02
        text_data.size = 0.3
    else:
        text_obj = bpy.data.objects[obj_name]
        text_data = text_obj.data
    text_data.body = f"[{word_string}]"

def get_xyz(point_data):
    """
    萬用解析器：相容新版結構化 Dict 與舊版平坦 List，
    穩定擷取三維特徵點座標，增強系統容錯率。
    """
    if isinstance(point_data, dict):
        return point_data.get("x", 0.0), point_data.get("y", 0.0), point_data.get("z", 0.0)
    elif isinstance(point_data, (list, tuple)):
        if len(point_data) >= 4: return point_data[1], point_data[2], point_data[3]
        elif len(point_data) >= 3: return point_data[0], point_data[1], point_data[2]
    return 0.0, 0.0, 0.0

def load_hand_mocap(json_filepath):
    """
    核心重定向函數：讀取 AI 視覺模型輸出的特徵資料，
    進行反正規化與空間座標系轉換後，寫入 Blender 關鍵影格。
    """
    if not os.path.exists(json_filepath):
        print(f"找不到檔案: {json_filepath}")
        return
    
    # 擷取檔名作為 UI 顯示詞彙
    filename = os.path.basename(json_filepath)
    word_label = filename.split('_')[0]
    update_sign_word_text(word_label)
        
    with open(json_filepath, 'r', encoding='utf-8') as f:
        mocap_data = json.load(f)
        
    # 自動偵測 JSON 版本並轉化為統一陣列格式
    if isinstance(mocap_data, dict):
        frames_list = mocap_data.get("frames", [])
    else:
        frames_list = mocap_data

    # 建立左右手腕的局部空間容器，並設定手動校準之最佳歐拉角
    l_wrist_box = create_empty_if_not_exists("Left_Wrist_Box", size=0.1)
    l_wrist_box.rotation_euler = (math.radians(202), math.radians(-160), math.radians(90))
    r_wrist_box = create_empty_if_not_exists("Right_Wrist_Box", size=0.1)
    r_wrist_box.rotation_euler = (math.radians(202), math.radians(-160), math.radians(90)) # 若右手數值不同請於此修改

    # 建立 21 個手部特徵點並綁定至手腕容器之下
    l_points = [create_empty_if_not_exists(f"L_Hand_{i}") for i in range(21)]
    r_points = [create_empty_if_not_exists(f"R_Hand_{i}") for i in range(21)]
    for i in range(21):
        l_points[i].parent = l_wrist_box
        r_points[i].parent = r_wrist_box

    # 遍歷影格寫入相對座標
    for frame_idx, frame_data in enumerate(frames_list):
        if isinstance(frame_data, dict):
            l_hand_raw = frame_data.get("left_hand")
        else:
            l_slice = frame_data[24:87]
            l_hand_raw = None if all(abs(v) < 0.05 for v in l_slice) else [l_slice[i*3:i*3+3] for i in range(21)]

        if l_hand_raw:
            l_wrist_x, l_wrist_y, l_wrist_z = get_xyz(l_hand_raw[0])
            for i in range(21):
                x, y, z = get_xyz(l_hand_raw[i])
                
                # 空間轉換矩陣：反正規化 (x0.3) 與 Y/Z 軸向對調
                scale = 0.3  
                final_x = (x - l_wrist_x) * scale
                final_y = (z - l_wrist_z) * scale       
                final_z = -(y - l_wrist_y) * scale      
                
                l_points[i].location = (final_x, final_y, final_z)
                l_points[i].keyframe_insert(data_path="location", frame=frame_idx + 1)

    print("✅ 動畫重定向完成！")

# 執行區塊 (路徑請依實際環境修改)
json_file_path = r"JSON檔案的路徑.json"
load_hand_mocap(json_file_path)