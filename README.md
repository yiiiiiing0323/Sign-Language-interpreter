# Sign-Language-interpreter

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Frontend: Three.js](https://img.shields.io/badge/Frontend-Three.js-green)
![Status: In-Development](https://img.shields.io/badge/Status-In--Development-yellow)

本系統旨在提供**台灣手語 (TSL)** 與**中文（文字/語音）**之間的即時雙向翻譯，促進聽障人士與一般人士在日常生活中的溝通。

> 📢 **專案開發階段聲明**：本專案目前處於**分模態核心開發與實驗階段**。各核心模組（雙流辨識演算法、LLM 語意轉換、3D 動畫生成）正同步進行獨立算法驗證與測試，後續將透過 WebSocket / API 進行全系統串接整合。

---

## 📌 系統簡介與開發背景

### 開發背景
現有溝通工具多依賴文字輸入或視訊聯繫，對於以手語為母語的聽障族群而言，轉換至文字邏輯仍存在思維門檻。目前市面上的手語辨識多針對美國手語 (ASL)，台灣手語 (TSL) 的即時翻譯工具相對匱乏。此外，手語的語法結構（如語序倒裝、表情配合）與口語中文存在顯著差異，需要更深層的語意轉換機制。

### 開發動機
本專案的核心動機在於開發一個屬於台灣本土的雙向翻譯平台，讓手語使用者能以熟悉的肢體語言表達，同時讓不具備手語能力的聽人能透過語音或文字無障礙地理解對方的意思，進而提升醫療、櫃台服務及日常生活的互動品質。

### 系統特色與創新性
* **🔄 即時雙向翻譯（規劃中）**：支援「手語錄入轉文字/語音」與「文字/語音輸入轉手語」雙向並行。
* **🧠 創新「雙流手勢辨識機制」（核心開發中）**：為了兼顧辨識的泛化能力與精準度，系統整合了兩大辨識主流：
  * **【A 流】AI 深度學習辨識**：利用時序模型學習複雜的手語句型與上下文語意。
  * **【B 流】幾何特徵規則辨識**：透過計算手指關節角度、空間向量與相對距離等物理幾何特徵，針對關鍵單詞進行精準的物理校正。
* **🙌 多模態特徵整合**：利用 MediaPipe Holistic 擷取手部（21 點骨架）、臉部表情（口型、眉毛）與全身姿態資料，提供雙流辨識模組足夠的特徵基底。
* **🤖 擬真 3D 視覺化（動作庫建立中）**：使用 Blender 進行骨架綁定，並預計透過 Three.js 於網頁端驅動 3D 虛擬角色（Avatar）呈現手語。

---

## 🏗️ 系統架構與設計藍圖 (System Architecture)

本系統採獨立模組化並行開發。後端採用 **AI 與幾何特徵雙軌並流（Dual-stream）** 的辨識機制，並導入 **LLM（大型語言模型）** 進行精準的台灣手語與中文語序重組，實現流暢的雙向閉環（Closed-loop）互動：

```mermaid
graph TD
    %% 節點樣式定義
    classDef input fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef core fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef stream fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;

    %% 1. 手語輸入與辨識端
    subgraph TSL_to_Chinese ["1. 手語輸入與辨識端 : TSL 轉中文"]
        A["視訊畫面輸入"] --> B["MediaPipe 特徵擷取"]
        B --> C1["【A 流】AI 辨識端<br>Transformer/GRU/LSTM "]
        B --> C2["【B 流】幾何辨識端<br>關節角度 / 向量物理規則"]
        C1 --> D["LLM 語序重組"]
        C2 --> D
        D --> E["中文文字 / 語音輸出"]
    end

    %% 2. 手語雙向反饋與動畫端
    subgraph Chinese_to_TSL ["2. 手語雙向反饋與動畫端 : 中文轉 TSL"]
        F["用戶中文文字 / 語音輸入"] --> G["ASR 語音辨識"]
        G --> H["字幕 ／ 動畫同步模組"]
        E -->|同步輸出至手語反饋模組| H
        H --> I["3D Avatar 手語呈現"]
    end

    %% 套用樣式表
    class A,F input;
    class B,D,H core;
    class E,I output;
    class C1,C2 stream;
    %% 套用樣式表
    class A,F input;
    class B,D,H core;
    class E,I output;
    class C1,C2 stream;
```

### 各模組功能與現階段分工說明

1. **使用者介面層（Web 前端 UI 草案）**
   * **功能描述**：提供網頁端視訊畫面、翻譯結果顯示區，並內嵌 Three.js 3D 虛擬角色。
   * **開發技術**：HTML5、CSS3、JavaScript、Three.js。

2. **MediaPipe 特徵擷取模組**
   * **功能描述**：利用 `MediaPipe Holistic` 追蹤並擷取手部 21 點骨架、臉部表情與全身姿態，將其序列化為 JSON 格式之時序關鍵點資料，作為後端雙流辨識的共同底數據。
   * **開發技術**：Python、OpenCV、MediaPipe。

3. **手語辨識模組 —【A 流：AI 深度學習】**
   * **功能描述**：接收特徵點的時序資料，經由深度學習模型進行複雜句型與語意上下文的特徵識別，輸出初步的手語詞彙（Gloss）。
   * **開發技術**：PyTorch、Transformer / GRU 模型。

4. **手語辨識模組 —【B 流：幾何特徵規則】**
   * **功能描述**：直接計算手指關節間的角度、空間向量與相對距離，利用 Rule-based（基於規則）的物理演算法進行特定關鍵單詞的精準物理校正與補強。
   * **開發技術**：Python、NumPy（幾何向量計算）。

5. **大型語言模型語意重組引擎（LLM Engine）**
   * **功能描述**：利用 LLM 強大的自然語言理解能力，將 A/B 雙流辨識出的手語詞彙（Gloss），依據台灣手語文法進行語序倒裝調整與潤飾，轉譯為符合自然中文文法的語句輸出。
   * **開發技術**：LLM API 串接、Prompt Tuning。

6. **3D 手語動畫生成與字幕同步模組**
   * **功能描述**：在 Blender 中完成 3D 角色骨架綁定（Armature Rig）與手語動作 Keyframe 片段製作。本模組具備**雙軌驅動能力**：除了能接收外部輸入的中文轉為手語外，也能將系統辨識完經 LLM 重組後的中文再次轉為手語動畫，供聽障使用者進行**即時雙向反饋驗證**。
   * **開發技術**：Blender（動作庫製作）、Three.js（網頁端動態 Clip 組合播放與字幕控制器）。

   ## 📂 專案目錄結構 (Project Structure)

目前專案目錄依據組員功能分工進行切分，各組員於對應資料夾下進行獨立開發、資料收集與演算法實驗，以便於未來進行模組化串接：
Sign-Language-interpreter/

└── README.md
