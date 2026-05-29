from dataclasses import dataclass
from typing import Deque, Iterable, Optional, Tuple
from collections import deque


@dataclass(frozen=True)
class RecognitionResult:
    word: str
    confidence: float
    source: str


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def weighted_confidence(
    rule_score: Optional[float],
    ai_score: Optional[float],
    rule_weight: float = 0.4,
    ai_weight: float = 0.6,
) -> float:
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
    """Keep output stable by requiring repeated evidence across recent frames."""

    def __init__(self, window_size: int = 5, min_score: float = 0.55):
        self.window: Deque[Tuple[str, float]] = deque(maxlen=window_size)
        self.min_score = min_score

    def update(self, word: Optional[str], confidence: float) -> Tuple[Optional[str], float]:
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

