# Sign-Language-interpreter
[video_to_npy.py](sign_language_ai/LSTM/video_to_npy.py)
主要功能：手語影片資料預處理。將原始的錄影檔格式（.mp4）轉換成 AI 模型可以讀取的數值矩陣（.npy），方便後續進行訓練。
運作流程：
影像辨識：使用 MediaPipe 找出影片中每一影格的面部、雙手與身體關鍵點。
特徵計算：除了基本的座標外，還會自動計算手指彎曲角度、手肘角度以及手部相對於頭部的位移距離。
數據正規化：自動依照每個人的肩寬縮放座標，確保模型不會因為人站得太遠或太近而辨識錯誤。
格式統一：將所有影片長度固定壓縮或填充為 30 幀（Frames），並合併成一個 218 維的特徵向量。
輸出入說明：
輸入：讀取 videos/ 資料夾中各個類別的手語影片。
輸出：在 data/ 資料夾產出對應的 .npy 檔案（每個檔案包含 30 組 218 維的數據）。
[number.py](sign_language_ai/LSTM/number.py)
[labels.py](sign_language_ai/LSTM/labels.py)
[train_lstm.py](sign_language_ai/LSTM/train_lstm.py)
[realtime_demo_lstm.py](sign_language_ai/LSTM/realtime_demo_lstm.py)
[train_gru.py](sign_language_ai/GRU/train_gru.py)
[inference_gru.py](sign_language_ai/GRU/inference_gru.py)
[train_transformer.py](sign_language_ai/Transformer/train_transformer.py)
[realtime_demo.py](sign_language_ai/Transformer/realtime_demo.py)



