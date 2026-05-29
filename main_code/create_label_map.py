import json

# 你確定百分之百都有訓練到的完整 43 個詞彙清單
words = [
    "他", "我_A", "我_B", "我倆", "我們", "你", "好", "謝謝_A", "謝謝_B", "再見",
    "早上", "平安", "幫助", "大", "小", "多", "少", "冷", "熱", "難過",
    "快樂", "累", "生氣", "跑_A", "陪伴", "吃", "吃飯", "看", "偷看", "去",  
    "來", "說話", "寫", "先生", "太太", "夫妻", "女生", "物", "水餃", "晚上",
    "家", "路", "這"
]

# 按字母排序（這個排序順序是 Unicode 決定的，跟當初訓練時一樣）
sorted_words = sorted(words)

# 建立 label_map
label_map = {str(i): word for i, word in enumerate(sorted_words)}

# 儲存對照表（裡面維持有 _A _B，確保主程式即時辨識時不會出錯）
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("✅ label_map.json 已建立")
print(f"📦 目前清單總字數：{len(label_map)} 個")

# 🟢 繞過模型加載，直接為你強制印出乾淨的提示！
print(f"\n📝 最終排序後的詞彙對照提示（已過濾 _A、_B，請安心對照）：")
print("-" * 50)
for i in range(len(label_map)):
    raw_word = label_map[str(i)]
    
    # 自動轉換機制：如果帶有 _A 或 _B，去除底線與後置代號
    clean_word = raw_word.split('_')[0] if '_' in raw_word else raw_word
    
    if raw_word != clean_word:
        print(f"  {i}: {clean_word}  (內部標籤: {raw_word})")
    else:
        print(f"  {i}: {clean_word}")
print("-" * 50)