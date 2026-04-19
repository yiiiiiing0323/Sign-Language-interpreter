import os

VIDEO_DIR = "videos"

def rename_videos_in_subfolders(root_dir):
    # 支援的影片格式
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    # 1. 遍歷第一層子資料夾 (Labels)
    subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)
        print(f"正在處理資料夾: {folder}")
        
        # 2. 取得該子資料夾內的所有影片檔案
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        # 排序檔案，確保編號邏輯一致
        video_files.sort()

        # 3. 獨立編號並重新命名
        for index, filename in enumerate(video_files, start=1):
            old_path = os.path.join(folder_path, filename)
            
            # 取得副檔名
            ext = os.path.splitext(filename)[1]
            
            # 定義新檔名 (例如編號為 0001, 0002...)
            new_filename = f"{index:04d}{ext}"
            new_path = os.path.join(folder_path, new_filename)
            
            # 執行重新命名
            try:
                os.rename(old_path, new_path)
                # print(f"  {filename} -> {new_filename}") # 若需要查看細節可取消註解
            except Exception as e:
                print(f"  重新命名 {filename} 失敗: {e}")
        
        print(f"--- 資料夾 '{folder}' 處理完成，共編號 {len(video_files)} 個影片 ---\n")

# 執行
if os.path.exists(VIDEO_DIR):
    rename_videos_in_subfolders(VIDEO_DIR)
else:
    print(f"找不到資料夾: {VIDEO_DIR}")