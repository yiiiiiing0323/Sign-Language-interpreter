import bpy

def apply_hand_constraints():
    """
    物理約束驅動：為虛擬人手部骨骼動態添加 DAMPED_TRACK 約束器。
    強制骨骼之局部 Y 軸追蹤目標特徵點，避免萬向鎖破圖。
    """
    # 取得目前場景中的骨架物件 (需根據實際模型名稱修改)
    armature_name = "Armature" 
    if armature_name not in bpy.data.objects:
        print("找不到骨架！")
        return
        
    armature = bpy.data.objects[armature_name]
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # 建立左右手指名稱的對應表 (VRM 標準名稱 -> 特徵點 Index)
    # 註：這裡請替換成你實際寫好的骨頭名稱對應字典
    l_finger_mapping = {
        "J_Bip_L_Thumb1": "L_Hand_1",
        # ... 略 ...
    }
    
    for bone_name, target_name in l_finger_mapping.items():
        if bone_name in armature.pose.bones and target_name in bpy.data.objects:
            bone = armature.pose.bones[bone_name]
            target_empty = bpy.data.objects[target_name]
            
            # 清除舊的同名約束器避免疊加
            for c in bone.constraints:
                if c.name == 'Auto_Track':
                    bone.constraints.remove(c)
                    
            # 新增 Damped Track 約束器
            constraint = bone.constraints.new('DAMPED_TRACK')
            constraint.name = 'Auto_Track'
            constraint.target = target_empty
            # 將追蹤軸向設為 Y 軸 (骨骼的延伸方向)
            constraint.track_axis = 'TRACK_Y'
            
    print("✅ 物理約束器綁定完成！")

apply_hand_constraints()