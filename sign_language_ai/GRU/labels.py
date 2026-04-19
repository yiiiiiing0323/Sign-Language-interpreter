import os
import pandas as pd

data_dir = "data"
rows = []

for f in os.listdir(data_dir):
    if f.endswith(".npy"):
        label = f.split("_")[0]  # filename 開頭當 label
        rows.append({"filename": f, "label": label})

df = pd.DataFrame(rows)
df.to_csv("labels.csv", index=False, encoding="utf-8-sig")
print("labels.csv 已生成")
