import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json

# -----------------------------
# 1. Dataset (改為 CSV 讀取模式)
# -----------------------------
class SignDataset(Dataset):
    def __init__(self, csv_file, data_dir, seq_len=30):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.seq_len = seq_len
        # 自動從 CSV 的 label 欄位獲取類別清單並排序
        self.labels = sorted(self.df["label"].unique())
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data_path = os.path.join(self.data_dir, row["filename"])
        
        # 讀取 npy
        try:
            data = np.load(data_path)
        except Exception as e:
            print(f"讀取錯誤 {data_path}: {e}")
            # 若讀取失敗回傳全 0 矩陣作為保險（實務上應確保資料完整）
            return torch.zeros((self.seq_len, 215)), 0

        # 補齊或截斷至 seq_len (30 幀)
        if data.shape[0] < self.seq_len:
            pad_len = self.seq_len - data.shape[0]
            pad = np.zeros((pad_len, data.shape[1]), dtype=np.float32)
            data = np.vstack([data, pad])
        elif data.shape[0] > self.seq_len:
            data = data[:self.seq_len]

        label = self.label2idx[row["label"]]
        return torch.tensor(data, dtype=torch.float32), label

# -----------------------------
# 2. Collate function
# -----------------------------
def collate_fn(batch):
    sequences, labels = zip(*batch)
    return torch.stack(sequences), torch.tensor(labels)

# -----------------------------
# 3. Transformer Model
# -----------------------------
class SignTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.d_model = 256
        self.embedding = nn.Linear(input_dim, self.d_model)
        
        # 位置編碼
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, self.d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # x: (batch, 30, input_dim)
        x = self.embedding(x)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = x.mean(dim=1)  # 全局平均池化
        return self.classifier(x)

# -----------------------------
# 4. Training Main
# -----------------------------
def main():
    data_dir = "data"
    csv_file = "labels.csv" # 確保此檔案存在
    seq_len = 30

    if not os.path.exists(csv_file):
        print(f"錯誤：找不到標籤檔 {csv_file}")
        return

    # 初始化 Dataset 與 DataLoader
    dataset = SignDataset(csv_file, data_dir, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # 自動偵測輸入維度
    sample_data, _ = dataset[0]
    INPUT_DIM = sample_data.shape[1] 
    NUM_CLASSES = len(dataset.labels)
    
    print(f"偵測到特徵維度: {INPUT_DIM}")
    print(f"類別清單: {dataset.labels}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SignTransformer(INPUT_DIM, NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    epochs = 50 # Transformer 需要較穩定的訓練次數
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    # --- 儲存模型與標籤 ---
    torch.save(model.state_dict(), "sign_transformer.pth")
    print("Transformer 模型已儲存至 sign_transformer.pth")

    idx2label = {i: l for l, i in dataset.label2idx.items()}
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(idx2label, f, ensure_ascii=False, indent=4)
    print("標籤映射 (label_map.json) 已儲存")

if __name__ == "__main__":
    main()