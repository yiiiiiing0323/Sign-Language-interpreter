import os
import csv

DATA_DIR = "data"
OUT_CSV = "labels.csv"

rows = []

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".npy"):
        # 取檔名前綴當 label
        label = fname.split("_")[0]
        rows.append([fname, label])

# 寫入 CSV
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)

print(f"labels.csv 已產生，共 {len(rows)} 筆資料")
