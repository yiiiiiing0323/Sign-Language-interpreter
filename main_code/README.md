# 台灣手語辨識專題 README

這個專題是一套 Windows 上執行的台灣手語辨識系統。系統同時使用兩條辨識來源：

- A 流：MediaPipe 特徵擷取 + LSTM 模型辨識。
- B 流：Excel 規則引擎，使用人工定義的手語規則判斷。
- Fusion：把 A 流與 B 流結果整合成最後輸出詞。
- Compound Phrase：把連續詞合併成複合詞，例如「爸爸/父親 + 弟弟 -> 叔叔」。

目前主程式是 `main.py`，執行方式：

```powershell
conda activate thrid
python main.py
```

## 專案架構

```text
專題/
  main.py                    # 主程式：攝影機、MediaPipe、A/B 流、fusion、畫面顯示
  a_stream.py                # A 流特徵擷取：MediaPipe landmarks -> feature dict + AI tensor
  b_stream.py                # B 流規則引擎：讀 database.xlsx 工作表3，判斷靜態/動態規則
  fusion.py                  # 融合器：整合 AI 結果與規則結果，處理同義詞/混淆詞/平滑
  compound_phrase.py         # 複合詞處理：把已輸出的詞串接成更長詞
  database.xlsx              # 規則資料庫：B 流規則與複合詞來源
  sign_lstm.pth              # LSTM 模型權重
  label_map.json             # 模型輸出 index -> 詞彙標籤
  model_contract.json        # 模型契約：檢查模型、label_map、tensor 格式是否一致
  requirements.txt           # Python 套件版本
  core/
    feature_registry.py      # A 流/B 流/AI tensor 的特徵契約與 tensor layout 版本
    safe_rule_engine.py      # 安全規則 evaluator，不使用 eval()
    word_normalization.py    # 詞彙正規化：_A/_B、斜線候選詞、候選群組
    confidence.py            # 信心分數與 temporal smoother
    logging_config.py        # log 設定
  tests/
    test_pipeline_logic.py   # 關鍵邏輯測試
```

## 執行流程

1. `main.py` 開啟攝影機。
2. MediaPipe 取得 hands、pose、face landmarks。
3. `a_stream.py` 產生兩種資料：
   - `current_features`：給 B 流 Excel 規則使用。
   - `ai_tensor`：長度 218 的數值向量，給 LSTM 使用。
4. A 流把最近 30 幀 `ai_tensor` 放進 `sequence_buffer`，滿 30 幀後丟給 `sign_lstm.pth`。
5. B 流把 `current_features` 套進 `database.xlsx` 的規則。
6. `fusion.py` 合併 A 流與 B 流輸出。
7. `compound_phrase.py` 檢查最後幾個詞能不能組成複合詞。
8. `main.py` 把結果顯示在畫面上。

## 各程式角色

### `main.py`

主控整個流程：

- 初始化 MediaPipe hands、pose、face。
- 初始化 `AStreamFeatureExtractor`、`BStreamGestureMatcher`、`DecisionFusion`、`CompoundPhraseResolver`。
- 載入 `label_map.json`、`model_contract.json`、`sign_lstm.pth`。
- 檢查模型契約是否一致，不一致時停用 AI 流，但 B 流仍可運作。
- 管理 AI 的 30 幀 buffer。
- LSTM 每幀取 top-5 機率：只有最高分 `> 0.65` 才會採用為 `ai_result`；只有在最高分已經過這個門檻時，才會進一步檢查 top-5 裡有沒有屬於 `fusion.py` `confusable_groups` 的候選詞並合併。這個順序很重要——低信心雜訊幀不能單純因為候選分數彼此接近，就繞過 0.65 門檻產生看似篤定的合併結果。
- 當手、姿勢或 handedness 追蹤中斷時，清除 AI buffer、B 流 sequence 狀態、A 流動作歷史。
- 呼叫 fusion，並把 B 流結果類型 `static` / `sequence` 傳給 fusion，避免動態規則完成後又被二次平滑刷掉。

### `a_stream.py`

A 流特徵擷取器：

- 讀取 MediaPipe landmarks。
- 產生 `current_features` 給 B 流使用。
- 產生 218 維 `ai_tensor` 給 LSTM 使用。
- 維護多種 movement history，例如揮動、上下移動、旋轉、震動等動態特徵。
- `reset_motion_history()` 會在追蹤中斷（沒有偵測到 pose）時清空全部動作歷史，避免上一段手勢殘留到下一段。
- 除此之外，`extract_features()` 內部也會逐手判斷：只要「這一幀」沒偵測到左手或右手，就分別清空該手的 `prev_left_wrist_y` / `prev_right_wrist_y`，不用等到兩手都消失。這樣單手暫時被遮擋、幾幀後又出現時，不會拿消失前很久的舊座標算出一個失真的位移量。

注意：如果 Excel 規則需要新特徵，通常要先確認 `a_stream.py` 有產生這個 feature 名稱。

### `b_stream.py`

B 流規則辨識器：

- 固定讀取 `database.xlsx` 的 `工作表3`。
- 必要欄位：
  - `ID`
  - `中文`
  - `MediaPipe 關鍵特徵`
- 支援靜態規則，例如：

```text
is_V_shape_HAND == True and palm_facing_backward_HAND == True
```

- 支援動態 sequence 規則，例如：

```text
sequence([step1_condition], [step2_condition])
```

- 靜態規則需要連續 3 幀穩定才輸出。
  - 每條靜態規則的穩定幀數是**各自獨立累積**的（用 `gesture_id` 當 key 存在 `self.static_stability`），不是全域共用一個計數器。
  - 這代表某一幀即使是別的規則（尤其是 sequence）贏得輸出，也不會把其他不相關靜態手勢已經累積的穩定度洗掉；只要那條規則自己的條件持續成立，進度就會繼續累積，不會被迫重新從 0 開始。
- 動態 sequence 完成後立即輸出，不再被 3 幀穩定條件刷掉。
- `sequence_timeout = 2.5` 秒，兩個 step 間隔太久會重置。
- `sequence_emit_cooldown = 1.0` 秒，避免同一個 sequence 連續重複輸出。
- 若多個規則同時命中，不再單純用 Excel 上下順序決定，而是依序比較：
  - sequence 優先於 static
  - Excel 欄位 `優先級` / `Priority` / `排序` / `Order`
  - 規則特徵數量，也就是 specificity
  - confidence
- 如果完全同分，會輸出候選群組，例如 `[上方詞/下方詞]`，避免上方規則永遠壓住下方規則。

#### 這個排序/候選群組邏輯寫在哪、怎麼修改

- 排序依據（`sort_key`）是在 `evaluate_frame_with_confidence()` 裡組成 `(kind, priority, specificity, confidence)`；`priority`/`specificity` 分別來自 `_rule_priority()`、`_rule_specificity()`。想加新的比較因子（例如使用頻率、歷史正確率），就在這個 tuple 裡加一項。
- 真正「比較兩個候選、決定誰贏或要不要合併成候選群組」的地方是 `_pick_better_candidate()`。`sort_key` 不同時直接選分數高的；完全相同時走最後的「Equal」分支，把兩邊的詞合併起來。想改「同分時的行為」（例如同分時永遠選 Excel 上面那條，而不是合併成候選群組），就是改這個分支。
- 候選群組實際的字串格式化（把 `["上方詞", "下方詞"]` 變成 `"[上方詞/下方詞]"`）在 `core/word_normalization.py` 的 `format_candidates()`，跟 AI 流無關，純粹是 B 流內部規則互相比較的結果。

### `fusion.py`

A/B 流融合器：

- 如果 A/B 流輸出相同或同義，合併為同一詞。
- 如果 A/B 流是容易混淆的詞，輸出候選群組。
- 如果 B 流結果已經是 `static` 或 `sequence`，fusion 會讓它立即通過，不再要求額外連續幀。
- 同義詞語意群組維護在這裡，例如「爸爸 / 父親」這類真正語意相同的詞。

建議分工：

- 語意同義詞：放在 `fusion.py` 的 `similar_groups`。
- 動作容易混淆但語意不同：放在 `fusion.py` 的 `confusable_groups`。
- 格式正規化，例如 `_A/_B`、`車_A_N/巴士`、`[先生/謝謝]`：由 `core/word_normalization.py` 處理。

#### 如何調整 AI 流與 B 流的信心權重

AI(LSTM) 跟 B 流(Excel 規則) 的信心分數怎麼混合，跟上面 B 流內部「多規則同分」是兩個完全不同的機制，要分開看：

- `core/confidence.py` 的 `weighted_confidence(rule_score, ai_score, rule_weight=0.4, ai_weight=0.6)` 是唯一的權重混合公式，預設 AI 佔 0.6、B 流佔 0.4。
- `fusion.py` 的 `fuse_with_confidence()` 裡有 3 種情境呼叫它，權重不一定用預設值：
  - AI/B 流輸出相同詞（`BOTH`）或同義詞（`SYNONYM`）：用預設 `rule_weight=0.4, ai_weight=0.6`。
  - 混淆詞（`CONFUSABLE`）：明確覆寫成 `rule_weight=0.5, ai_weight=0.5`。
  - AI/B 流輸出完全不同的詞、且 B 流不是穩定的 `static`/`sequence` 時：**不是用 `weighted_confidence()` 混合分數**，而是用 `fuse_with_confidence()` 裡的本地常數 `LOGIC_WEIGHT = 1.0`、`AI_WEIGHT = 1.0` 直接比較誰的加權分數高，分數差在 `0.15` 以內才會輸出候選群組。
- 想調整「AI 跟規則整體誰比較可信」：改 `core/confidence.py` 裡 `weighted_confidence` 的預設值（會影響 `BOTH`/`SYNONYM` 兩種情境）。
- 只想調整「AI/B 流完全不同詞時誰該贏」：改 `fusion.py` 裡的 `LOGIC_WEIGHT`/`AI_WEIGHT` 或 `0.15` 那個門檻，跟 `weighted_confidence` 的預設值無關。

### `compound_phrase.py`

複合詞處理器：

- 從 `database.xlsx` 讀取複合詞來源。
- 預設讀 `工作表3`。
- 來源欄位預設是 `Unnamed: 0`。
- 輸出欄位是 `中文`。
- 支援 `+` 串接，例如：

```text
爸爸/父親+弟弟 -> 叔叔
```

- 會共用 `core/word_normalization.py`，所以 `_A/_B` 與斜線候選詞可以被正確比對。
- 會優先匹配較長的複合詞，避免短詞先蓋掉長詞。

### `core/safe_rule_engine.py`

安全規則 evaluator：

- 用 Python AST 解析規則。
- 不使用 `eval()`。
- 只允許安全的布林、比較、數值與變數名稱。
- 如果 Excel 規則引用 A 流沒有提供的 feature，會 fail closed，也就是不命中。
- 這可避免類似 `dist_typo < 0.15` 因缺值被當成 0 而誤判成功。

### `core/feature_registry.py`

特徵契約：

- 定義 AI tensor 維度，目前是 `218`。
- 定義 `AI_TENSOR_LAYOUT_VERSION`。
- B 流會用 A 流輸出的 `current_features` 檢查 Excel 規則引用的特徵是否都存在。
- 如果未來改 AI tensor 排列方式，一定要更新 `AI_TENSOR_LAYOUT_VERSION`，並重新訓練或確認模型契約。

### `core/word_normalization.py`

詞彙正規化：

- `我_A` -> `我`
- `我_B` -> `我`
- `車_A_N/巴士` -> `車`
- `[先生/謝謝]` 保留為候選群組格式
- 給 `fusion.py` 與 `compound_phrase.py` 共用。

## Excel 規則設計

### B 流規則欄位

`database.xlsx` 的 `工作表3` 至少需要：

- `ID`：規則 ID，例如 `N_044`、`A_012`。
- `中文`：命中後要輸出的中文詞。
- `MediaPipe 關鍵特徵`：規則條件。

可選欄位：

- `優先級`
- `Priority`
- `排序`
- `Order`

這些欄位可用來解決多個規則同時命中時的優先順序。

### 靜態規則

靜態規則是單一條件式：

```text
is_flat_HAND == True and dist_HAND_8_FACE_1 < 0.5
```

行為：

- 需要連續 3 幀同一詞命中。
- 適合手型、手掌方向、固定位置。

### 動態規則

動態規則使用 `sequence(...)`：

```text
sequence([第一階段條件], [第二階段條件])
```

行為：

- 每個 `[...]` 是一個 step。
- step 完成後會記住目前進度。
- 最後一步命中後立即輸出。
- 不需要再連續 3 幀。

### 規則撰寫注意事項

- 布林特徵建議明確寫 `== True` 或 `== False`。
- 數值特徵才使用 `<`、`>`、`<=`、`>=`。
- 若是 `vector_align_HAND_PALM_UPWARD` 這類目前由程式輸出的布林特徵，請寫：

```text
vector_align_HAND_PALM_UPWARD == True
```

- 不要寫成：

```text
vector_align_HAND_PALM_UPWARD > 0.7
```

除非 A 流真的輸出的是數值。

## 模型與資料檔關係

AI 流需要三個檔案互相一致：

- `sign_lstm.pth`：模型權重。
- `label_map.json`：模型輸出 index 對應的詞。
- `model_contract.json`：模型契約檢查。

目前 `main.py` 載入模型時會檢查：

- `input_dim` 是否等於 `218`。
- `sequence_length` 是否等於 `30`。
- `class_count` 是否等於模型 `fc.weight` 的輸出類別數。
- `tensor_layout` 是否等於 `core/feature_registry.py` 的 `AI_TENSOR_LAYOUT_VERSION`。
- `label_map_sha256` 是否等於目前 `label_map.json` 的 SHA-256。

如果任何一項不一致，AI 流會停用，系統會只使用 B 流。

## 如何替換新模型

### 情況 A：新模型使用同一批類別、同一個 label_map

這是最簡單的情況。

1. 備份舊模型：

```powershell
Copy-Item sign_lstm.pth sign_lstm.old.pth
```

2. 把新模型檔改名成：

```text
sign_lstm.pth
```

3. 覆蓋專案內舊的 `sign_lstm.pth`。
4. 如果輸入維度、sequence length、類別數都沒變，`label_map.json` 與 `model_contract.json` 通常不用改。
5. 執行：

```powershell
python main.py
```

如果看到 `LSTM 模型已載入`，代表契約通過。

### 情況 B：新模型類別有新增、刪除或順序改變

這時一定要更新 `label_map.json`，而且順序必須跟訓練模型時的 class index 完全一致。

`label_map.json` 格式：

```json
{
  "0": "你好",
  "1": "謝謝",
  "2": "爸爸"
}
```

重要規則：

- key 必須是字串數字：`"0"`、`"1"`、`"2"`。
- value 是該 index 對應的詞。
- index 順序必須跟訓練時 dataset 的 `class_to_idx` 或 label encoder 一致。
- 不可以依照 Excel 順序隨便重排。

訓練完新模型後，請從訓練程式輸出同一份 mapping。建議訓練時直接輸出：

```python
import json

with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump({str(i): label for i, label in idx_to_label.items()}, f, ensure_ascii=False, indent=2)
```

如果你的訓練程式是 `class_to_idx = {"你好": 0, "謝謝": 1}`，可轉成：

```python
idx_to_label = {idx: label for label, idx in class_to_idx.items()}
```

### 情況 C：模型輸入格式改變

如果你有改：

- AI tensor 維度
- 218 維的排列順序
- sequence length
- LSTM hidden size / layer 結構

就不是單純替換 `sign_lstm.pth`。你需要同步修改 `main.py` 的模型定義或載入參數，並更新 `core/feature_registry.py` 的 `AI_TENSOR_LAYOUT_VERSION` 與 `model_contract.json`。

目前 `main.py` 的 LSTM 架構是：

```text
input_dim = 218
hidden_dim = 128
num_layers = 2
sequence_length = 30
output_classes = model fc.weight rows
```

## 如何更新 `model_contract.json`

當新模型或 `label_map.json` 改變後，請更新 `model_contract.json`。

可以用這段指令產生需要填入的值：

```powershell
python -c "import json, hashlib, torch; from core.feature_registry import AI_TENSOR_DIM, AI_TENSOR_LAYOUT_VERSION; sd=torch.load('sign_lstm.pth', map_location='cpu'); print(json.dumps({'input_dim': AI_TENSOR_DIM, 'sequence_length': 30, 'class_count': sd['fc.weight'].shape[0], 'tensor_layout': AI_TENSOR_LAYOUT_VERSION, 'label_map_sha256': hashlib.sha256(open('label_map.json','rb').read()).hexdigest()}, ensure_ascii=False, indent=2))"
```

把輸出的 JSON 覆蓋到 `model_contract.json`。

如果只是調整 Excel 規則，不需要更新 `model_contract.json`。

## 新模型替換檢查清單

替換前：

- 確認新模型檔名最後是 `sign_lstm.pth`。
- 確認新模型架構仍是 `SignRNN(input_dim=218, hidden_dim=128, num_layers=2, num_classes=類別數)`。
- 確認新模型訓練使用 30 幀序列。
- 確認 `label_map.json` 來自同一次訓練，不要手動猜順序。

替換後：

- 更新 `label_map.json`。
- 更新 `model_contract.json`。
- 執行 `python main.py`。
- 若 AI 載入失敗，看錯誤中 `模型契約不一致` 的欄位。
- 若 AI 載入成功但辨識詞錯位，優先檢查 `label_map.json` 順序。

## 常見問題

### 為什麼新模型載入失敗？

常見原因：

- `model_contract.json` 沒更新。
- `label_map.json` 的 SHA-256 和 `model_contract.json` 不一致。
- 新模型類別數和 `model_contract.json` 的 `class_count` 不一致。
- 新模型不是目前 `SignRNN` 架構。
- 新模型不是用 218 維、30 幀訓練。

### 為什麼 AI 載入成功但輸出詞不對？

最常見原因是 `label_map.json` 的 index 順序錯了。

例如模型第 0 類其實是「你好」，但 `label_map.json` 寫成：

```json
{
  "0": "謝謝"
}
```

那模型預測第 0 類時就會顯示成「謝謝」。

### 同義詞要加在哪裡？

- 「語意真的相同」：加在 `fusion.py` 的 `similar_groups`。
- 「動作相似但意思不同」：加在 `fusion.py` 的 `confusable_groups`。
- `_A/_B`、斜線、候選群組這種格式問題：不要加同義詞，交給 `core/word_normalization.py`。

### Excel 新增規則後一直出現未知特徵？

代表 `database.xlsx` 使用了 A 流沒有提供的 feature 名稱。

處理方式：

1. 檢查拼字是否錯。
2. 檢查左右手命名是否一致，例如 `LEFT_HAND`、`RIGHT_HAND`、`HAND`。
3. 若確定是新特徵，先在 `a_stream.py` 補出該 feature。
4. 再重新執行。

### 多個規則同時命中怎麼辦？

B 流現在不會單純使用 Excel 上方規則。它會看：

- sequence/static
- 優先級
- 規則特徵數量
- confidence

如果真的完全同分，輸出候選群組，例如 `[詞A/詞B]`。

## 測試

執行目前的邏輯測試：

```powershell
python -m unittest discover -s tests -v
```

測試涵蓋：

- 未知特徵 fail closed。
- `_A/_B` 與斜線詞正規化。
- 同義詞融合。
- 動態 sequence 完成後立即輸出。
- 靜態規則保留 3 幀防抖。
- 追蹤中斷後 sequence 狀態重置。
- 下方更具體規則可勝過上方泛用規則。
- 完全同分規則輸出候選群組。

## 環境

建議環境：

- Windows 10 或 Windows 11
- Conda env name: `thrid`
- Python `3.9.25`

安裝：

```powershell
conda create -n thrid python=3.9.25
conda activate thrid
python -m pip install -r requirements.txt
```

目前主要套件：

- `opencv-python`
- `mediapipe`
- `numpy`
- `pandas`
- `openpyxl`
- `Pillow`
- `requests`
- `python-dotenv`
- `torch`

## Gemini

Gemini 翻譯目前是可選功能。

如果要啟用，需要 `.env`：

```text
GEMINI_API_KEY=your_api_key
```

目前主程式預設可在不啟用 Gemini 的情況下運作。

## 交接給 AI 時建議貼的摘要

這個專題是台灣手語辨識系統，主程式 `main.py`。A 流在 `a_stream.py` 使用 MediaPipe 產生 `current_features` 與 218 維 `ai_tensor`；AI LSTM 使用 `sign_lstm.pth`、`label_map.json`、`model_contract.json`，30 幀序列輸入。B 流在 `b_stream.py` 讀 `database.xlsx` 的 `工作表3`，用安全 AST evaluator 判斷 Excel 規則，靜態規則需 3 幀穩定，`sequence([...], [...])` 動態規則完成即輸出。`fusion.py` 合併 AI/B 結果，語意同義詞在 `similar_groups`，混淆詞在 `confusable_groups`，格式正規化在 `core/word_normalization.py`。複合詞由 `compound_phrase.py` 處理。新增 Excel feature 時必須確認 A 流有提供同名 feature。替換新模型時要同步確認 `sign_lstm.pth`、`label_map.json`、`model_contract.json` 一致，尤其 `label_map.json` 的 index 順序必須跟訓練時相同。
