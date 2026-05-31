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
    """
    自定義手語特徵資料集類別，透過直接掃描硬碟資料夾動態建立標籤
    """
    def __init__(self, data_dir, seq_len=30):
        self.data_dir = data_dir
        self.seq_len = seq_len
        
        # 💡 使用 glob 掃描兩層資料夾：data/標籤資料夾/所有.npy 檔案
        # 這會直接抓取特定路徑下符合結構的所有特徵檔，免去讀取 CSV 的步驟
        self.files = glob.glob(os.path.join(data_dir, "*", "*.npy"))
        
        # 安全檢查：若完全沒掃到任何特徵檔，拋出檔案不存在異常提示
        if not self.files:
            raise FileNotFoundError(f"在 {data_dir} 資料夾中找不到 .npy 檔案")
            
        # 💡 清理資料夾名稱，去掉隨附的流派後綴（例如：_A、_B、_C 變回純文字標籤）
        def clean_label(folder_name):
            return re.sub(r'_[A-Za-z]', '', folder_name)
        
        # 利用集合（Set）推導式抓取所有檔案的父資料夾名稱，取得不重複的原始資料夾清單
        raw_labels = {os.path.basename(os.path.dirname(f)) for f in self.files}    
        
        # 💡 從資料夾名稱取得排序後的標籤清單，確保每次執行的標籤順序一致
        self.labels = sorted(list({os.path.basename(os.path.dirname(f)) for f in self.files}))
        # 建立「文字標籤對應數字索引」的字典（形狀如：{"謝謝_A": 0, "你好_A": 1}）
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        # 回傳本次掃描到的特徵檔案總總數
        return len(self.files)

    def __getitem__(self, idx):
        # 依據索引取得對應的特徵檔案路徑
        file_path = self.files[idx]
        
        # 嘗試加載 NumPy 矩陣，若檔案損壞則進行異常捕獲處理
        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"讀取錯誤 {file_path}: {e}")
            # 若檔案崩潰，回傳一個全零的時間步矩陣，避免整體訓練流程中斷
            return torch.zeros((self.seq_len, 218)), 0

        # --- 時序長度對齊機制（固定截斷或補零至 30 幀） ---
        if data.shape[0] < self.seq_len:
            # 實際長度不足 30 幀，計算缺少的影格數
            pad_len = self.seq_len - data.shape[0]
            # 建立形狀為 (缺少幀數, 218) 的全零矩陣
            pad = np.zeros((pad_len, data.shape[1]), dtype=np.float32)
            # 在尾端進行垂直拼接補齊
            data = np.vstack([data, pad])
        else:
            # 實際長度大於 30 幀，直接截取前 30 幀
            data = data[:self.seq_len]
        
        # 💡 從該檔案的父資料夾名稱取得其所屬的手語標籤名稱
        label_name = os.path.basename(os.path.dirname(file_path))
        # 回傳轉為 PyTorch 張量的特徵矩陣，與字典中查詢到的對應類別數字編號
        return torch.tensor(data, dtype=torch.float32), self.label2idx[label_name]

# -----------------------------
# GRU 模型定義
# -----------------------------
class SignGRU(nn.Module):
    """
    基於 218 維特徵輸入的閘道循環單元（GRU）手語分類器
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        # 建立雙層 GRU 循環網路，batch_first=True 設定張量首位為 Batch 大小
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全連接分類層，將隱藏狀態映射到最終的手語總類別數
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 前向傳播，out 的形狀為 (Batch_Size, 30, hidden_dim)
        out, _ = self.gru(x)
        # 時序記憶提取：擷取最後一個時間步（第 30 幀）的隱藏特徵送入線性層分類
        return self.fc(out[:, -1, :])

# -----------------------------
# 訓練主程式
# -----------------------------
def main():
    # 自動偵測運算裝置（優先調用 CUDA 顯示卡加速）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 設定特徵檔案存放的磁碟絕對路徑
    data_dir = r"E:\HandSignProject\LSTM\data"
    epochs = 60 # 設定總訓練週期
    
    # 實例化資料夾直讀資料集
    dataset = SignDataset(data_dir)
    # 打包成可迭代的批次控制器，設定批次大小為 32，隨機洗牌資料
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 取出第一筆特徵資料，動態分析特徵維度長度（通常為 218 維）
    sample_data, _ = dataset[0]
    INPUT_DIM = sample_data.shape[1]
    
    HIDDEN_DIM = 128                  # 設定隱藏層寬度
    NUM_LAYERS = 2                    # 設定 GRU 堆疊層數
    NUM_CLASSES = len(dataset.labels) # 動態取得掃描到的手語詞彙類別總數

    print(f"偵測到特徵維度: {INPUT_DIM}, 類別總數: {NUM_CLASSES}")

    # 實例化 GRU 模型並部署至指定硬體
    model = SignGRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)
    # 定義 Adam 優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 定義分類專用交叉熵損失函數
    criterion = nn.CrossEntropyLoss()

    print(f"開始訓練 GRU... 裝置: {device}")
    
    best_loss = float('inf') # 初始化最佳 Loss 為無限大
    
    # 開啟 60 輪的訓練迴圈
    for epoch in range(epochs):
        model.train() # 將模型設為訓練模式（開啟 Dropout 與 BatchNorm 等行為）
        total_loss = 0
        
        # 批次反向傳播訓練
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad() # 歷史梯度清空
            out = model(x)        # 前向傳播預測
            loss = criterion(out, y) # 計算損失值
            loss.backward()       # 反向傳播計算各參數梯度
            optimizer.step()      # 更新權重參數
            
            total_loss += loss.item() # 累加批次損失
            
        # 計算本週期的平均 Loss 損失
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        # 最佳權重即時保存保存機制
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存當前平均損失最低的最佳權重字典
            torch.save(model.state_dict(), "best_sign_gru_model.pth")
            print(f"🌟 發現更強的模型！已更新儲存 (Loss: {best_loss:.4f})")

    # 訓練完成後，儲存最後一輪的最終模型狀態字典
    torch.save(model.state_dict(), "sign_gru.pth")
    
    # 💡 儲存標籤映射（推論解碼時會用到）
    # 將原始的「標籤轉數字字典」反轉成「數字索引對應文字標籤」的解碼映射字典
    idx2label = {i: l for l, i in dataset.label2idx.items()}
    # 以 JSON 格式持久化寫入磁碟，格式化縮排設為 4 空格
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(idx2label, f, ensure_ascii=False, indent=4)
        
    print(f"\n訓練完成！最終模型已儲存為 sign_gru.pth，標籤映射已更新。")

if __name__ == "__main__":
    main()
