import os
import re
import pandas as pd

data_dir = r"E:\HandSignProject\LSTM\data"
rows = []

for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if not os.path.isdir(label_dir):
        continue
    for f in os.listdir(label_dir):
        if not f.endswith(".npy"):
            continue
        relative_path = os.path.join(label, f)
        
        # 💡 把 _A、_B、_C 這種 _單個英文字母 去掉
        clean_label = re.sub(r'_[A-Za-z]$', '', label)
        
        rows.append({"filename": relative_path, "label": clean_label})

df = pd.DataFrame(rows)
df.to_csv("labels.csv", index=False, encoding="utf-8-sig")
print(f"✅ labels.csv 已生成，共 {len(rows)} 筆")