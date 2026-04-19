import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import glob

# -----------------------------
# Dataset 類別
# -----------------------------
class SignDataset(Dataset):
    def __init__(self, data_dir, seq_len=30):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.files = glob.glob(os.path.join(data_dir, "*.npy"))
        
        if not self.files:
            raise FileNotFoundError(f"在 {data_dir} 資料夾中找不到 .npy 檔案")
            
        # 從檔名解析標籤
        self.labels = sorted(list({os.path.basename(f).split("_")[0] for f in self.files}))
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        
        # 確保動作區段長度固定 30 幀
        if data.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, pad])
        else:
            data = data[:self.seq_len]
        
        label_name = os.path.basename(file_path).split("_")[0]
        return torch.tensor(data, dtype=torch.float32), self.label2idx[label_name]

# -----------------------------
# GRU 模型定義
# -----------------------------

class SignGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        # GRU 適合處理長度較短的時序數據，參數比 LSTM 少，訓練更快
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        # 取最後一個時間步的輸出進行分類
        return self.fc(out[:, -1, :])

# -----------------------------
# 訓練主程式
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "data"
    
    dataset = SignDataset(data_dir)
    # 建議將 batch_size 稍微調大一點（如 16 或 32），有助於正規化後的數值穩定收斂
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # --- 關鍵修正：自動抓取特徵維度 ---
    # 不要寫死 332，這裡會自動變成 206
    sample_data, _ = dataset[0]
    INPUT_DIM = sample_data.shape[1] 
    
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NUM_CLASSES = len(dataset.labels)

    print(f"偵測到特徵維度: {INPUT_DIM}, 類別總數: {NUM_CLASSES}")

    model = SignGRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"開始訓練 GRU... 裝置: {device}")
    model.train()
    for epoch in range(50): # 正規化後可以多訓練幾輪（建議 50-60）
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # 儲存模型
    torch.save(model.state_dict(), "sign_gru.pth")
    
    # 儲存標籤映射
    idx2label = {i: l for l, i in dataset.label2idx.items()}
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(idx2label, f, ensure_ascii=False, indent=4)
        
    print(f"訓練完成！模型已儲存為 sign_gru.pth，標籤映射已更新。")

if __name__ == "__main__":
    main()