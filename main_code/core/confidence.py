import sys
sys.dont_write_bytecode = True

from dataclasses import dataclass
from typing import Deque, Iterable, Optional, Tuple
from collections import deque


@dataclass(frozen=True)
class RecognitionResult:
    """
    單一辨識結果的標準格式。

    目前主程式仍使用舊的 tuple 格式 (word, confidence)，
    這個 dataclass 保留給後續重構時逐步替換 tuple，讓 AI/B流/融合器格式一致。
    """
    word: str
    confidence: float
    source: str


def clamp_score(value: float) -> float:
    """把任何分數限制在 0.0 到 1.0，避免異常分數影響融合結果。"""
    return max(0.0, min(1.0, float(value)))


def weighted_confidence(
    rule_score: Optional[float],
    ai_score: Optional[float],
    rule_weight: float = 0.4,
    ai_weight: float = 0.6,
) -> float:
    """
    將規則流與 AI 流的信心分數加權合併。

    rule_score:
        B 流規則信心分數。
    ai_score:
        LSTM softmax 機率。
    rule_weight / ai_weight:
        兩個來源的權重。預設 AI 稍高，但 fusion.py 會依情境調整。
    """
    total = 0.0
    weight = 0.0
    if rule_score is not None:
        total += clamp_score(rule_score) * rule_weight
        weight += rule_weight
    if ai_score is not None:
        total += clamp_score(ai_score) * ai_weight
        weight += ai_weight
    return total / weight if weight else 0.0


class TemporalSmoother:
    """
    時間平滑器，用來降低畫面結果閃爍。

    手語辨識是逐影格執行，單一影格可能因光線、遮擋、MediaPipe 抖動而誤判。
    smoother 會保留最近幾幀結果，讓連續出現且信心足夠的詞才輸出。
    """

    def __init__(self, window_size: int = 5, min_score: float = 0.55):
        self.window: Deque[Tuple[str, float]] = deque(maxlen=window_size)
        self.min_score = min_score

    def update(self, word: Optional[str], confidence: float) -> Tuple[Optional[str], float]:
        """
        放入最新一幀的結果，回傳平滑後的結果。

        若 word 是 None，代表目前沒有可靠辨識結果，會清空視窗。
        若同一個詞在視窗中出現比例高、平均信心也高，才會輸出該詞。
        """
        if not word:
            self.window.clear()
            return None, 0.0

        self.window.append((word, clamp_score(confidence)))
        scores = {}
        counts = {}
        for item_word, item_score in self.window:
            scores[item_word] = scores.get(item_word, 0.0) + item_score
            counts[item_word] = counts.get(item_word, 0) + 1

        best_word = max(scores, key=scores.get)
        avg_score = scores[best_word] / counts[best_word]
        consistency = counts[best_word] / len(self.window)
        smoothed_score = clamp_score(avg_score * consistency)
        if smoothed_score < self.min_score:
            return None, smoothed_score
        return best_word, smoothed_score

    @staticmethod
    def average(scores: Iterable[float]) -> float:
        scores = [clamp_score(score) for score in scores]
        return sum(scores) / len(scores) if scores else 0.0
