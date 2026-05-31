import os

VIDEO_DIR = "videos"

def rename_videos_in_subfolders(root_dir):
    # 支援的影片格式
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    # 1. 遍歷第一層子資料夾 (Labels)
    subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)
        print(f"📁 正在處理資料夾: {folder}")
        
        # 2. 取得該子資料夾內的所有影片檔案
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        # 分類：記錄「已經命名好的數字」與「需要重新命名的檔案」
        existing_numbers = []
        files_to_rename = []

        for filename in video_files:
            name_part, ext = os.path.splitext(filename)
            # 如果檔名(不含副檔名)是純數字，代表已經命名好了
            if name_part.isdigit():
                existing_numbers.append(int(name_part))
            else:
                files_to_rename.append(filename)

        # 如果沒有檔案需要改名，直接跳過這個資料夾
        if not files_to_rename:
            print(f"--- 💡 資料夾 '{folder}' 內的所有檔案皆已命名好，無須處理 ---\n")
            continue

        # 排序需要改名的檔案，確保每次執行順序一致
        files_to_rename.sort()

        # 決定下一個可用的起始編號
        # 如果資料夾裡已經有編號，就從「最大編號 + 1」開始；如果沒有，就從 1 開始
        next_index = max(existing_numbers) + 1 if existing_numbers else 1

        # 3. 針對未命名的檔案進行獨立編號並重新命名
        for filename in files_to_rename:
            old_path = os.path.join(folder_path, filename)
            ext = os.path.splitext(filename)[1]
            
            # 確保新檔名絕對不會與現有檔案衝突
            while True:
                new_filename = f"{next_index:04d}{ext}"
                new_path = os.path.join(folder_path, new_filename)
                
                if not os.path.exists(new_path):
                    break # 確定檔名沒有人使用，跳出迴圈
                next_index += 1 # 如果剛好撞名，就把編號加 1 繼續找
            
            # 執行重新命名
            try:
                os.rename(old_path, new_path)
                print(f"  ✅ {filename} -> {new_filename}")
                next_index += 1 # 成功改名後，將編號往前推進
            except Exception as e:
                print(f"  ❌ 重新命名 {filename} 失敗: {e}")
        
        print(f"--- 資料夾 '{folder}' 處理完成，共新增編號 {len(files_to_rename)} 個影片 ---\n")

# 執行
if os.path.exists(VIDEO_DIR):
    rename_videos_in_subfolders(VIDEO_DIR)
else:
    print(f"找不到資料夾: {VIDEO_DIR}")