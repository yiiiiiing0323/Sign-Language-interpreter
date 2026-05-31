import os
import re
import pandas as pd

# 定義特徵資料存放的根目錄名稱
data_dir = "data"
# 建立一個空清單，用來儲存所有資料列（每筆資料包含檔案路徑與標籤）
rows = []

# 第一層迴圈：遍歷 data 資料夾底下的所有子項目（通常是各個手語詞彙的標籤資料夾）
for label in os.listdir(data_dir):
    # 組合出子項目的完整路徑（例如：data/謝謝_A）
    label_dir = os.path.join(data_dir, label)
    
    # 檢查該路徑是否為資料夾，如果不是（例如是獨立檔案），則跳過不處理
    if not os.path.isdir(label_dir):
        continue
        
    # 第二層迴圈：遍歷該標籤資料夾底下的所有檔案
    for f in os.listdir(label_dir):
        # 檢查檔案副檔名是否為 .npy（MediaPipe 提取出的時序特徵檔），如果不是則跳過
        if not f.endswith(".npy"):
            continue
            
        # 組合出檔案相對於 data 目錄的相對路徑（例如：謝謝_A/1.npy）
        relative_path = os.path.join(label, f)
        
        # 💡 使用正則表達式（re.sub）進行標籤清洗
        # 尋找底線後面接「單個英文字母」的樣式（例如：_A, _B, _a），並將其替換為空字串（去掉）
        clean_label = re.sub(r'_[A-Za-z]', '', label)
        
        # 將整理好的相對路徑與清洗後的乾淨標籤，以字典形式新增到 rows 清單中
        rows.append({"filename": relative_path, "label": clean_label})

# 將儲存所有資料的清單轉換成 Pandas 的 DataFrame 結構
df = pd.DataFrame(rows)

# 將 DataFrame 匯出為 CSV 檔案
# index=False 代表不寫入 Pandas 自動生成的索引編號列
# encoding="utf-8-sig" 確保匯出的 CSV 在 Windows Excel 中開啟時不會出現亂碼
df.to_csv("labels.csv", index=False, encoding="utf-8-sig")

# 於終端機印出成功提示訊息，並顯示本次共處理並寫入了多少筆特徵資料
print(f"✅ labels.csv 已生成，共 {len(rows)} 筆")
