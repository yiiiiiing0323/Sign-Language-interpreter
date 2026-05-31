import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json

# -----------------------------
# 資料集處理 (Dataset)
# -----------------------------
class SignDataset(Dataset):
    """
    自定義手語特徵資料集類別，繼承自 PyTorch 的 Dataset 結構
    """
    def __init__(self, csv_file, data_dir, seq_len=30):
        # 讀取包含檔案路徑與標籤對應關係的 CSV 檔案
        self.df = pd.read_csv(csv_file)
        # 儲存特徵檔案的根目錄路徑
        self.data_dir = data_dir
        # 設定模型要求的固定時序長度（預設為 30 幀）
        self.seq_len = seq_len
        # 取得 CSV 檔案中所有不重複的手語中文標籤，並進行排序
        self.labels = sorted(self.df["label"].unique())
        # 建立「文字標籤轉數字索引」的字典映射表（例如：{"謝謝": 0, "你好": 1}）
        self.label2idx = {l:i for i,l in enumerate(self.labels)}

    def __len__(self):
        # 回傳資料集內資料的總筆數
        return len(self.df)

    def __getitem__(self, idx):
        # 依據索引編號取得該筆資料的 CSV 資料列
        row = self.df.iloc[idx]
        # 組合出該筆特徵檔案的完整磁碟路徑
        data_path = os.path.join(self.data_dir, row["filename"])
        # 載入 NumPy 時序特徵矩陣，形狀為 (該影片總幀數, 特徵維度218)
        data = np.load(data_path)

        # --- 時序特徵長度對齊機制（補齊或截斷） ---
        # 情況 A：如果實際影片幀數不足 30 幀，則在矩陣尾端補零
        if data.shape[0] < self.seq_len:
            pad_len = self.seq_len - data.shape[0] # 計算需要補齊的幀數
            # 建立形狀為 (補齊幀數, 218維) 的全零矩陣
            pad = np.zeros((pad_len, data.shape[1]), dtype=np.float32)
            # 使用 vstack 進行垂直拼接，補足到 30 幀
            data = np.vstack([data, pad])
        # 情況 B：如果實際影片幀數大於 30 幀，則直接截取前 30 幀
        elif data.shape[0] > self.seq_len:
            data = data[:self.seq_len]

        # 將該筆資料的中文文字標籤，轉換成對應的整數數字索引
        label = self.label2idx[row["label"]]
        # 回傳轉換成 PyTorch 張量 (Tensor) 的特徵矩陣以及對應的數字標籤
        return torch.tensor(data, dtype=torch.float32), label

# -----------------------------
# 批次打包函式 (Collate function)
# -----------------------------
def collate_fn(batch):
    """
    自定義 DataLoader 提取批次資料時的打包邏輯
    """
    # 將一個 Batch 內多個 (sequence, label) 的元組進行解包分離
    sequences, labels = zip(*batch)
    # 使用 torch.stack 將特徵矩陣串接成 (Batch_Size, 30, 218) 的高維張量，並將標籤打包成一維整數張量
    return torch.stack(sequences), torch.tensor(labels)

# -----------------------------
# LSTM / GRU 模型定義 (Model)
# -----------------------------
class SignRNN(nn.Module):
    """
    具備切換功能的循環神經網路模型架構，支援 LSTM 與 GRU 兩種核心單元
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, rnn_type="LSTM"):
        super().__init__()
        # 將傳入的模型類型字串統一轉大寫方便比對
        self.rnn_type = rnn_type.upper()
        # 根據指定類型建立對應的神經網路層
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 分類用的全連接輸出層，將隱藏層維度映射到最終的手語總類別數
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 餵入特徵進行前向傳播，out 的輸出形狀為 (Batch_Size, 30, hidden_dim)
        out, _ = self.rnn(x)
        # 記憶提取：取最後一個時間步（也就是第 30 幀，最末尾的隱藏狀態）進行語意分類
        out = out[:, -1, :] 
        # 送入全連接層計算分類機率分數
        out = self.fc(out)
        return out

# -----------------------------
# 訓練主程式 (Training)
# -----------------------------
def main():
    # 初始化設定相關參數
    data_dir = "data"
    csv_file = "labels.csv"
    seq_len = 30

    # 實例化自定義資料集
    dataset = SignDataset(csv_file, data_dir, seq_len=seq_len)
    # 建立 DataLoader 控制器，設定批次大小(batch_size)為 32、開啟洗牌打亂功能(shuffle)並套用 collate_fn
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 動態取得特徵維度長度（此專案中為 218 維）
    INPUT_DIM = dataset[0][0].shape[1]
    HIDDEN_DIM = 128      # 隱藏層神經元數量
    NUM_LAYERS = 2       # 循環網路的疊加層數
    NUM_CLASSES = len(dataset.labels) # 最終要辨識的手語詞彙總數

    # 硬體加速偵測
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 本程式預設使用 LSTM 進行訓練實驗，並搬移至相應運算裝置
    model = SignRNN(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, rnn_type="LSTM").to(device)

    # 定義 Adam 優化器，設定學習率 (Learning Rate) 為 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 定義多元分類專用的交叉熵損失函數
    criterion = nn.CrossEntropyLoss()
    # 初始化最佳損失值為正無窮大，用來記錄最優模型權重
    best_loss = float('inf')
    
    # 啟動 60 個週期的訓練迴圈
    for epoch in range(60):
        total_loss = 0 # 累加當前 Epoch 的總損失值
        # 自 DataLoader 迭代提取批次資料
        for x, y in loader:
            # 將特徵與標籤張量送入對應運算硬體（GPU 或 CPU）
            x, y = x.to(device), y.to(device)

            # 梯度歸零（防止上一個批次的梯度殘留影響本次計算）
            optimizer.zero_grad()
            # 進行前向傳播，取得模型預測的分數
            out = model(x)
            # 計算當前預測結果與真實標籤之間的 Loss 損失值
            loss = criterion(out, y)
            # 反向傳播（Backward），計算各參數的梯度
            loss.backward()
            # 依據計算出的梯度更新模型內的所有參數權重
            optimizer.step()

            # 累加當前 Batch 的 Loss 數值
            total_loss += loss.item()
            
        # 計算此 Epoch 的整體平均損失值
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        
        # 最佳權重存檔機制：若當前的平均 Loss 比歷史紀錄還要小，則觸發自動更新
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 儲存目前 Loss 最小的最佳模型狀態字典（權重檔）
            torch.save(model.state_dict(), "best_sign_model.pth")
            print(f"🌟 發現更強的模型！已更新儲存 (Loss: {best_loss:.4f})")
            
    # 60 個 Epoch 訓練結束後，儲存最後一輪完成的最終模型權重
    torch.save(model.state_dict(), "sign_lstm.pth")
    print("模型已儲存")

    # --- 儲存標籤映射字典檔案 ---
    # 將原始的「標籤轉數字字典」反轉成「數字索引轉中文標籤」的形式（例如：{0: "謝謝", 1: "你好"}）
    idx2label = {i:l for l,i in dataset.label2idx.items()}
    # 將該字典以 JSON 格式持久化寫入磁碟，供即時攝影機 Demo 端讀取解碼
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(idx2label, f, ensure_ascii=False) # ensure_ascii=False 確保寫入非 ASCII 字元時中文字不會變成亂碼
    print("標籤映射已儲存")

if __name__ == "__main__":
    main()
