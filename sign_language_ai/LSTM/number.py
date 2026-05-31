import os

# 定義影片存放的根目錄名稱
VIDEO_DIR = "videos"

def rename_videos_in_subfolders(root_dir):
    # 定義支援的影片副檔名格式（統一轉小寫比對避免遺漏）
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    # 1. 遍歷第一層子資料夾 (Labels)
    # 利用清單推導式（List Comprehension）篩選出根目錄下所有的子資料夾
    subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for folder in subfolders:
        # 組合出目前處理子資料夾的完整路徑（例如：videos/謝謝_A）
        folder_path = os.path.join(root_dir, folder)
        print(f"📁 正在處理資料夾: {folder}")
        
        # 2. 取得該子資料夾內的所有影片檔案
        # 篩選符合副檔名清單且結尾相符的檔案
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        # 分類群組：用來記錄「已經命名好的純數字檔案」與「其餘需要重新命名的亂碼檔案」
        existing_numbers = []
        files_to_rename = []

        for filename in video_files:
            # 切割出檔名（不含副檔名）與副檔名本身
            name_part, ext = os.path.splitext(filename)
            # 如果檔名部分全部由數字組成（isdigit()），代表該影片之前已經被命名好編號了
            if name_part.isdigit():
                existing_numbers.append(int(name_part)) # 轉成整數存入現存編號清單
            else:
                files_to_rename.append(filename) # 否則丟進待改名清單

        # 安全檢查：如果這個資料夾內全都是命好名的純數字檔案，則直接跳過此資料夾
        if not files_to_rename:
            print(f"--- 💡 資料夾 '{folder}' 內的所有檔案皆已命名好，無須處理 ---\n")
            continue

        # 排序需要改名的檔案，確保在不同作業系統上每次執行的讀取重新命名順序一致
        files_to_rename.sort()

        # 決定下一個可用的起始編號
        # 如果資料夾裡已經有純數字編號，就從「最大編號 + 1」開始接續往下編（增量機制）；如果沒有，就從 1 開始
        next_index = max(existing_numbers) + 1 if existing_numbers else 1

        # 3. 針對未命名的檔案進行獨立編號並重新命名
        for filename in files_to_rename:
            old_path = os.path.join(folder_path, filename) # 舊檔案的完整路徑
            ext = os.path.splitext(filename)[1]           # 取得原副檔名（如 .mp4）
            
            # 防撞名無窮迴圈：確保產生出的新檔名絕對不會與此資料夾現有的任何舊檔案衝突
            while True:
                # 格式化為 4 位數，不足前面補 0（例如：編號 5 變成 0005.mp4）
                new_filename = f"{next_index:04d}{ext}"
                new_path = os.path.join(folder_path, new_filename) # 新檔案的完整路徑
                
                # 檢查該路徑在磁碟上是否已存在
                if not os.path.exists(new_path):
                    break # 確定這檔名沒人使用，安全無誤，跳出 while 迴圈
                next_index += 1 # 如果不幸剛好撞名，就把編號加 1，繼續進入下一次檢查
            
            # 執行實際的重新命名操作
            try:
                os.rename(old_path, new_path)
                print(f"  ✅ {filename} -> {new_filename}")
                next_index += 1 # 成功完成一個檔案改名後，將編號往前推進 1 號
            except Exception as e:
                # 萬一遇到權限不足或檔案被佔用等異常，印出錯誤訊息而不中斷整體程式運作
                print(f"  ❌ 重新命名 {filename} 失敗: {e}")
        
        print(f"--- 資料夾 '{folder}' 處理完成，共新增編號 {len(files_to_rename)} 個影片 ---\n")

# 主程式執行入口
if os.path.exists(VIDEO_DIR):
    rename_videos_in_subfolders(VIDEO_DIR)
else:
    print(f"找不到資料夾: {VIDEO_DIR}")
