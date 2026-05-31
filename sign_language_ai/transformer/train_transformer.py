import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json

# -----------------------------
# 1. Dataset 類別
# -----------------------------
class SignDataset(Dataset):
    """
    自定義手語特徵資料集類別，負責讀取特徵檔案並進行時序長度對齊
    """
    def __init__(self, csv_file, data_dir, seq_len=30):
        # 讀取記載檔案相對路徑與手語文字標籤的 CSV 對應表
        self.df = pd.read_csv(csv_file)
        # 儲存特徵資料存放的 Windows 絕對路徑
        self.data_dir = data_dir
        # 設定模型硬性規定的固定時序長度（預設 30 幀）
        self.seq_len = seq_len
        # 取得資料集中所有不重複且排序過的手語標籤清單
        self.labels = sorted(self.df["label"].unique())
        # 建立「中文標籤對應數字索引」的映射字典（例如：{"謝謝": 0, "你好": 1}）
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        # 回傳資料集內資料的總筆數
        return len(self.df)

    def __getitem__(self, idx):
        # 依據索引編號取得該筆資料的 CSV 橫列資料
        row = self.df.iloc[idx]
        # 組合出該筆特徵 .npy 檔案在硬碟中的完整路徑
        data_path = os.path.join(self.data_dir, row["filename"])
        
        # 嘗試加載 NumPy 矩陣，並進行檔案異常損壞捕獲處理
        try:
            data = np.load(data_path)
        except Exception as e:
            print(f"讀取錯誤 {data_path}: {e}")
            # 若發生讀取異常，回傳全零的預設特徵矩陣與類別 0，避免訓練程序中斷
            return torch.zeros((self.seq_len, 215)), 0

        # --- 時序特徵長度對齊機制（固定補齊或截斷至 30 幀） ---
        if data.shape[0] < self.seq_len:
            # 實際影格數不足 30 幀，計算缺少的長度
            pad_len = self.seq_len - data.shape[0]
            # 建立形狀為 (缺少影格數, 特徵維度) 的全零填充矩陣
            pad = np.zeros((pad_len, data.shape[1]), dtype=np.float32)
            # 在特徵矩陣尾端進行垂直拼接（Padding）
            data = np.vstack([data, pad])
        elif data.shape[0] > self.seq_len:
            # 實際影格數大於 30 幀，直接裁剪截斷前 30 幀（Truncate）
            data = data[:self.seq_len]

        # 將該資料列的中文標籤轉換為數字索引
        label = self.label2idx[row["label"]]
        # 回傳轉換為 PyTorch 張量的特徵矩陣與數字標籤
        return torch.tensor(data, dtype=torch.float32), label

# -----------------------------
# 2. 批次組合函式 (Collate function)
# -----------------------------
def collate_fn(batch):
    """
    自定義 DataLoader 打包批次資料時的堆疊邏輯
    """
    # 將一個 Batch 內多個 (sequence, label) 的元組進行解包分離
    sequences, labels = zip(*batch)
    # 將多個特徵張量堆疊成 (Batch_Size, 30, 特徵維度) 的高維張量，並將標籤組裝成一維整數張量
    return torch.stack(sequences), torch.tensor(labels)

# -----------------------------
# 3. Transformer 模型定義
# -----------------------------
class SignTransformer(nn.Module):
    """
    基於自注意力機制（Self-Attention）的手語動作分類 Transformer 模型
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.d_model = 256 # 設定 Transformer 內部隱藏層特徵維度寬度
        
        # 1. 嵌入層（Linear Embedding）：將輸入的原始特徵線性映射升維到 256 維
        self.embedding = nn.Linear(input_dim, self.d_model)
        
        # 2. 位置編碼（Positional Embedding）：由於自注意力機制不具備時序方向感，
        # 宣告一個可學習的張量參數 (1, 30幀, 256維)，與嵌入特徵相加，賦予時間順序資訊
        self.pos_embedding = nn.Parameter(torch.randn(1, 30, self.d_model))
        
        # 3. 自注意力編碼層：設定特徵維度為 256，多頭注意力機制頭數（nhead）為 8，並開啟 batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        # 堆疊 3 層 Transformer 編碼器
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 4. 分類層：全連接線性層，將經自注意力聚合後的 256 維特徵映射到手語總類別數
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # x 的輸入形狀為 (Batch_Size, 30, 原始特徵維度)
        x = self.embedding(x)          # 線性升維至 (Batch_Size, 30, 256)
        x = x + self.pos_embedding     # 加上可學習的時序位置編碼
        x = self.encoder(x)            # 送入 Transformer 編碼器計算時序上下文關聯
        x = x.mean(dim=1)              # 全局平均池化 (Global Mean Pooling)：沿著時間軸（dim=1）計算平均值，壓縮為 (Batch_Size, 256)
        return self.classifier(x)      # 送入最終線性分類層輸出各類別分數

# -----------------------------
# 4. 訓練主程式 (Training Main)
# -----------------------------
def main():
    # 初始化訓練設定參數
    data_dir = r"E:\HandSignProject\LSTM\data"
    csv_file = "labels.csv"
    seq_len = 30
    epochs = 60 # 統一設為 60 個週期

    # 檔案存在檢查安全性機制
    if not os.path.exists(csv_file):
        print(f"錯誤：找不到標籤檔 {csv_file}")
        return

    # 實例化自定義手語資料集
    dataset = SignDataset(csv_file, data_dir, seq_len=seq_len)
    # 建立 DataLoader 批次加載控制器，設定 Batch 大小為 16，隨機洗牌，並掛載自定義 collate_fn
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # 動態取得特徵維度寬度（此專案中通常對齊為 218 維，資料損壞兜底回傳為 215 維）
    sample_data, _ = dataset[0]
    INPUT_DIM = sample_data.shape[1] 
    NUM_CLASSES = len(dataset.labels) # 動態取得當前要辨識的手語中文詞彙總類別數
    
    print(f"偵測到特徵維度: {INPUT_DIM}")
    print(f"類別清單: {dataset.labels}")

    # 硬體加速偵測（優先調用 NVIDIA 顯示卡 CUDA）
    device = "cuda" if torch.torch.cuda.is_available() else "cpu"
    # 實例化 Transformer 模型並移置到相應運算硬體上
    model = SignTransformer(INPUT_DIM, NUM_CLASSES).to(device)

    # 宣告 Adam 優化器，設定較為保守穩健的學習率為 0.0005 (5e-4) 以利自注意力層穩定收斂
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # 宣告多元分類專用的交叉熵損失函數
    criterion = nn.CrossEntropyLoss()

    # --- 初始化最佳 Loss 紀錄 ---
    best_loss = float('inf') # 設為正無窮大，用於追蹤訓練期間表現最優異的模型

    # 啟動 60 個 Epoch 的訓練主迴圈
    for epoch in range(epochs):
        model.train() # 將模型切換為訓練模式（確保隨機行為正常運作）
        total_loss = 0
        
        # 疊代讀取批次特徵與標籤
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 歷史梯度清除歸零
            optimizer.zero_grad()
            # 前向傳播計算模型分數
            out = model(x)
            # 計算當前預測與真實答案之間的 Loss 損失值
            loss = criterion(out, y)
            # 反向傳播計算梯度
            loss.backward()
            # 優化器根據梯度更新 Transformer 所有內部權重與位置參數
            optimizer.step()
            
            # 累加批次損失
            total_loss += loss.item()

        # 計算本輪 Epoch 的平均 Loss 損失值
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        # --- 儲存最強模型邏輯 ---
        # 檢查點觸發機制：若當前 Epoch 的平均 Loss 創歷史新低，則自動覆蓋儲存最佳權重
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 持久化寫入當前表現最佳的模型狀態字典權重檔
            torch.save(model.state_dict(), "best_transformer_model.pth")
            print(f"🌟 發現更強的模型！已更新儲存 (Loss: {best_loss:.4f})")

    # --- 儲存最終模型與標籤 ---
    # 60 個週期完成後，保存最後一輪結束時的最終模型權重
    torch.save(model.state_dict(), "sign_transformer.pth")
    print("\n訓練結束，最終模型已儲存至 sign_transformer.pth")

    # 反轉字典：將原有的「標籤轉數字字典」翻轉為「數字索引解碼中文標籤」的字典（例如：{0: "謝謝", 1: "你好"}）
    idx2label = {i: l for l, i in dataset.label2idx.items()}
    # 將解碼映射表以 JSON 格式寫入磁碟，縮排設為 4 個空格，確保 ensure_ascii=False 讓中文正常顯示
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(idx2label, f, ensure_ascii=False, indent=4)
    print("標籤映射 (label_map.json) 已儲存")

if __name__ == "__main__":
    main()
