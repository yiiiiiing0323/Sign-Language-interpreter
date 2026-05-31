import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json

# -----------------------------
# 1. Dataset
# -----------------------------
class SignDataset(Dataset):
    def __init__(self, csv_file, data_dir, seq_len=30):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.labels = sorted(self.df["label"].unique())
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data_path = os.path.join(self.data_dir, row["filename"])
        
        try:
            data = np.load(data_path)
        except Exception as e:
            print(f"讀取錯誤 {data_path}: {e}")
            return torch.zeros((self.seq_len, 215)), 0

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
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, self.d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = x.mean(dim=1)  # 全局平均池化
        return self.classifier(x)

# -----------------------------
# 4. Training Main
# -----------------------------
def main():
    data_dir = r"E:\HandSignProject\LSTM\data"
    csv_file = "labels.csv"
    seq_len = 30
    epochs = 60 # 統一設為 60

    if not os.path.exists(csv_file):
        print(f"錯誤：找不到標籤檔 {csv_file}")
        return

    dataset = SignDataset(csv_file, data_dir, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    sample_data, _ = dataset[0]
    INPUT_DIM = sample_data.shape[1] 
    NUM_CLASSES = len(dataset.labels)
    
    print(f"偵測到特徵維度: {INPUT_DIM}")
    print(f"類別清單: {dataset.labels}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SignTransformer(INPUT_DIM, NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # --- 初始化最佳 Loss 紀錄 ---
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 計算本輪平均 Loss
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        # --- 儲存最強模型邏輯 ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_transformer_model.pth")
            print(f"🌟 發現更強的模型！已更新儲存 (Loss: {best_loss:.4f})")

    # --- 儲存最終模型與標籤 ---
    torch.save(model.state_dict(), "sign_transformer.pth")
    print("\n訓練結束，最終模型已儲存至 sign_transformer.pth")

    idx2label = {i: l for l, i in dataset.label2idx.items()}
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(idx2label, f, ensure_ascii=False, indent=4)
    print("標籤映射 (label_map.json) 已儲存")

if __name__ == "__main__":
    main()