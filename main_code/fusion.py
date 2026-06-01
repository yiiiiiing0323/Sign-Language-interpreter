"""
決策融合模組
負責整合 A流(AI) 和 B流(邏輯) 的結果
"""

import sys
sys.dont_write_bytecode = True

import logging

from core.confidence import TemporalSmoother, weighted_confidence


logger = logging.getLogger(__name__)


class DecisionFusion:
    """
    A 流 AI 與 B 流規則的決策融合器。

    輸入：
    - ai_result: LSTM 產生的 (word, confidence)
    - logic_result: B 流規則產生的 (word, confidence)

    輸出：
    - final_word: 最終要送給翻譯器的詞
    - source: 結果來源，例如 AI / LOGIC / BOTH / CONFUSABLE

    這裡同時處理：
    - 同義詞合併
    - 混淆詞保留
    - AI 與規則信心分數加權
    - temporal smoothing，降低連續影格閃爍
    """
    def __init__(self):
        # ========================================
        # ===== 同義詞群組（語意相同的詞）=====
        # ========================================
        self.similar_groups = {
            "我": ["我_A", "我_B"],
            "謝謝": ["謝謝_A", "謝謝_B"],
            "跑": ["跑_A"],
            "公車": ["公車", "客運", "巴士"],
            "學校": ["學校", "教室", "上課"],
            "家": ["家", "回家", "住處"],
            "吃": ["吃", "用餐", "食物"],
            "喝": ["喝", "飲料", "水"],
            "買": ["買", "購買", "消費"],
            "想": ["想", "思考", "認為"],
            # ⭐ 請根據你的手語詞彙持續擴充
        }
        
        # ========================================
        # ===== 混淆詞群組（動作相似但語意不同）=====
        # ========================================
        self.confusable_groups = [
            ["先生", "謝謝_A"],
            ["好", "吃"],
            ["好", "棒", "讚"],
            ["不要", "不是", "沒有"],
            ["爸爸", "叔叔"],
            ["媽媽", "阿姨"],
            ["喜歡", "愛"],
            ["早安", "你好"],
            ["再見", "掰掰"],
            ["什麼", "為什麼"],
            ["誰", "哪裡"],
            ["多少", "幾個"],
            ["可以", "好的", "沒問題"],
            ["生氣", "討厭"],
            ["高興", "開心", "快樂"],
            ["累", "睏", "想睡"],
            ["冷", "涼"],
            ["熱", "暖"],
            ["大", "多"],
            ["小", "少"],
            # ⭐ 請根據你的訓練資料持續擴充
        ]
        
        # 建立反向查詢表
        self.word_to_group = {}
        for main_word, synonyms in self.similar_groups.items():
            for word in synonyms:
                self.word_to_group[word] = main_word
        
        self.word_to_confusable_group = {}
        for group in self.confusable_groups:
            for word in group:
                self.word_to_confusable_group[word] = group
        
        # 統計資料（可選）
        self.confusable_stats = {}
        self.last_confidence = 0.0
        self.smoother = TemporalSmoother(window_size=5, min_score=0.50)
    
    
    def are_similar(self, word_a, word_b):
        """判斷兩個詞是否為同義詞"""
        if word_a == word_b:
            return True
        group_a = self.word_to_group.get(word_a)
        group_b = self.word_to_group.get(word_b)
        return group_a and group_b and group_a == group_b
    
    
    def are_confusable(self, word_a, word_b):
        """判斷兩個詞是否為混淆詞"""
        group = self.word_to_confusable_group.get(word_a)
        if group and word_b in group:
            return True
        return False
    
    
    def get_confusable_words(self, word):
        """取得與某個詞動作相似的其他詞"""
        return self.word_to_confusable_group.get(word, [])
    
    
    def merge_similar_words(self, word_a, word_b):
        """將同義詞合併"""
        group = self.word_to_group.get(word_a) or self.word_to_group.get(word_b)
        if group:
            candidates = self.similar_groups[group]
            appeared = set([word_a, word_b])
            result = [w for w in candidates if w in appeared]
            return f"[{'/'.join(result)}]"
        else:
            return f"[{word_a}/{word_b}]"
    
    
    def merge_confusable_words(self, words):
        """將多個混淆詞打包"""
        unique_words = sorted(set(words))
        return f"[{'/'.join(unique_words)}]"
    
    
    def _finish(self, word, source, confidence):
        """
        統一收尾融合結果。

        所有融合分支最後都會走到這裡：
        1. 將 confidence 放入 TemporalSmoother。
        2. 更新 last_confidence，讓 main.py 可以顯示分數。
        3. 寫入 system.log，方便日後追查 AI/B 流衝突。
        """
        smoothed_word, smoothed_confidence = self.smoother.update(word, confidence)
        self.last_confidence = smoothed_confidence
        if not smoothed_word:
            return None, None
        logger.info(
            "fusion_result word=%s source=%s confidence=%.3f",
            smoothed_word,
            source,
            smoothed_confidence,
        )
        return smoothed_word, source

    def fuse_with_confidence(self, ai_result, logic_result):
        """
        融合 A流 和 B流 的結果
        
        參數:
            ai_result: (word, confidence) 或 None
            logic_result: (word, confidence) 或 None
        
        回傳:
            final_word: 最終決定的詞彙
            source: "AI", "LOGIC", "BOTH", "SYNONYM", "CONFUSABLE", "UNCERTAIN", None
        """
        
        # 情況 1：兩方都沒有結果，代表目前影格沒有可靠手語詞。
        if ai_result is None and logic_result is None:
            self.last_confidence = 0.0
            self.smoother.update(None, 0.0)
            return None, None
        
        # 情況 2：只有 B 流命中，直接使用規則結果，但仍會進 temporal smoothing。
        if ai_result is None:
            return self._finish(logic_result[0], "LOGIC", logic_result[1])
        
        # 情況 3：只有 AI 流命中，直接使用 AI 結果。
        if logic_result is None:
            return self._finish(ai_result[0], "AI", ai_result[1])
        
        # 情況 2：雙方都有結果
        ai_word, ai_conf = ai_result
        logic_word, logic_conf = logic_result
        
        # 子情況 4a：AI 與 B 流完全同意，給額外加分。
        if ai_word == logic_word:
            confidence = weighted_confidence(logic_conf, ai_conf)
            return self._finish(ai_word, "BOTH", min(1.0, confidence + 0.10))
        
        # 子情況 4b：動作相似但語意不同，保留成候選組合，避免翻譯器過早選錯。
        if self.are_confusable(ai_word, logic_word):
            merged = self.merge_confusable_words([ai_word, logic_word])
            # 記錄統計
            key = tuple(sorted([ai_word, logic_word]))
            self.confusable_stats[key] = self.confusable_stats.get(key, 0) + 1
            confidence = weighted_confidence(logic_conf, ai_conf, rule_weight=0.5, ai_weight=0.5)
            return self._finish(merged, "CONFUSABLE", confidence)
        
        # 子情況 4c：同義詞或 A/B 版本，合併後輸出。
        if self.are_similar(ai_word, logic_word):
            merged = self.merge_similar_words(ai_word, logic_word)
            confidence = weighted_confidence(logic_conf, ai_conf)
            return self._finish(merged, "SYNONYM", confidence)
        
        # 子情況 4d：完全不同時，依信心分數加權比較。
        LOGIC_WEIGHT = 1.2  # ⭐ 邏輯流較可靠，給予加成
        AI_WEIGHT = 1.0
        
        weighted_logic = logic_conf * LOGIC_WEIGHT
        weighted_ai = ai_conf * AI_WEIGHT
        
        # 信心度接近，打包成候選詞
        if abs(weighted_logic - weighted_ai) < 0.15:
            confidence = weighted_confidence(logic_conf, ai_conf, rule_weight=0.5, ai_weight=0.5)
            return self._finish(f"[{logic_word}/{ai_word}]", "UNCERTAIN", confidence)
        
        # 選信心度高的
        if weighted_logic > weighted_ai:
            return self._finish(logic_word, "LOGIC", logic_conf)
        else:
            return self._finish(ai_word, "AI", ai_conf)

    def fuse(self, ai_result, logic_result):
        """相容舊版呼叫：回傳 (word, source)，信心分數放在 last_confidence。"""
        return self.fuse_with_confidence(ai_result, logic_result)
