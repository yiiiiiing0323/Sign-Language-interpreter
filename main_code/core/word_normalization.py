import re
from itertools import product
from typing import Iterable, Sequence, Tuple


_VARIANT_SUFFIX = re.compile(r"(?:_[A-Za-z])+$")


def word_alternatives(word, strip_suffix: bool = True) -> Tuple[str, ...]:
    """Return normalized alternatives for labels such as 爸爸/父親 or 我_A.

    strip_suffix=False 保留 _A/_B/_N/_S 這類尾碼。大部分情境（顯示、同義詞比對、
    temporal smoothing）都應該把尾碼去掉當同一個詞；但複合詞比對是例外——
    有些尾碼（例如 _N vs _S）代表語意不同、要輸出不同複合詞，不能被合併掉。
    """
    if word is None:
        return ()

    text = str(word).strip()
    if not text:
        return ()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    alternatives = []
    for part in text.split("/"):
        part = part.strip()
        normalized = _VARIANT_SUFFIX.sub("", part) if strip_suffix else part
        if normalized and normalized not in alternatives:
            alternatives.append(normalized)
    return tuple(alternatives)


def canonical_word(word) -> str:
    """Return the preferred internal token while preserving explicit candidate groups."""
    alternatives = word_alternatives(word)
    if not alternatives:
        return ""
    return alternatives[0]


def normalize_output_word(word) -> str:
    """Normalize a recognizer output; bracketed values remain candidate groups."""
    alternatives = word_alternatives(word)
    if not alternatives:
        return ""
    text = str(word).strip()
    if text.startswith("[") and text.endswith("]") and len(alternatives) > 1:
        return format_candidates(alternatives)
    return alternatives[0]


def expanded_patterns(words: Sequence[str], strip_suffix: bool = True) -> Iterable[Tuple[str, ...]]:
    """Expand each token's alternatives for compound-phrase matching."""
    alternatives = [word_alternatives(word, strip_suffix=strip_suffix) for word in words]
    if not alternatives or any(not choices for choices in alternatives):
        return ()
    return product(*alternatives)


def format_candidates(words: Iterable[str]) -> str:
    normalized = []
    for word in words:
        for alternative in word_alternatives(word):
            if alternative not in normalized:
                normalized.append(alternative)
    if not normalized:
        return ""
    if len(normalized) == 1:
        return normalized[0]
    return f"[{'/'.join(normalized)}]"
