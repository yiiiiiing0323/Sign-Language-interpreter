import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import glob
import re
# -----------------------------
# Dataset 類別
# -----------------------------
class SignDataset(Dataset):
    def __init__(self, data_dir, seq_len=30):
        self.data_dir = data_dir
        self.seq_len = seq_len
        
        # 💡 掃兩層資料夾：data/標籤/檔案.npy
        self.files = glob.glob(os.path.join(data_dir, "*", "*.npy"))
        
        if not self.files:
            raise FileNotFoundError(f"在 {data_dir} 資料夾中找不到 .npy 檔案")
        # 💡 清理資料夾名稱，去掉 _英文字母
        def clean_label(folder_name):
            return re.sub(r'_[A-Za-z]', '', folder_name)
        
        raw_labels = {os.path.basename(os.path.dirname(f)) for f in self.files}    
        # 💡 從資料夾名稱取得標籤
        self.labels = sorted(list({os.path.basename(os.path.dirname(f)) for f in self.files}))
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"讀取錯誤 {file_path}: {e}")
            return torch.zeros((self.seq_len, 218)), 0

        if data.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, pad])
        else:
            data = data[:self.seq_len]
        
        # 💡 從資料夾名稱取得標籤
        label_name = os.path.basename(os.path.dirname(file_path))
        return torch.tensor(data, dtype=torch.float32), self.label2idx[label_name]

# -----------------------------
# GRU 模型定義
# -----------------------------
class SignGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# -----------------------------
# 訓練主程式
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = r"E:\HandSignProject\LSTM\data"
    epochs = 60
    
    dataset = SignDataset(data_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

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
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_sign_gru_model.pth")
            print(f"🌟 發現更強的模型！已更新儲存 (Loss: {best_loss:.4f})")

    torch.save(model.state_dict(), "sign_gru.pth")
    
    # 💡 儲存標籤映射（推論時會用到）
    idx2label = {i: l for l, i in dataset.label2idx.items()}
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(idx2label, f, ensure_ascii=False, indent=4)
        
    print(f"\n訓練完成！最終模型已儲存為 sign_gru.pth，標籤映射已更新。")

if __name__ == "__main__":
    main()