from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import pandas as pd

from core.word_normalization import expanded_patterns

@dataclass(frozen=True)
class CompoundMatch:
    output: str
    consumed: int
    pattern: Tuple[str, ...]

class CompoundPhraseResolver:
    def __init__(
        self,
        excel_path: str,
        sheet_name="工作表3",  # 預設直接設為工作表3
        source_col="Unnamed: 0",     # 預設改成你的欄位名稱（若是第一欄且無名字，可用 "Unnamed: 0"）
        output_col="中文",     # 預設改成中文欄位名稱
        alt_sep="|",
        token_sep="+",
    ):
        # rules_exact 保留 _A/_B/_N/_S 尾碼，用來分辨語意不同的變體
        # （例如 要_N+不要_N -> 要不要_N，要_S+不要_S -> 要不要_S）。
        # rules_canonical 是原本「去尾碼」的寬鬆比對，當精確比對找不到結果時才退回使用，
        # 相容不在乎尾碼、單純用不同詞組拼出同一個輸出的規則。
        self.rules_exact: Dict[Tuple[str, ...], str] = {}
        self.rules_canonical: Dict[Tuple[str, ...], str] = {}
        self.max_len = 2
        self.loaded_ok = False

        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            # 🛡️ 自動相容：如果使用者傳入的是數字索引，就用 iloc 方式處理；如果是字串，就用標籤方式處理
            use_iloc = isinstance(source_col, int) or isinstance(output_col, int)

            for _, row in df.iterrows():
                if use_iloc:
                    # 如果是數字
                    source_val = row.iloc[source_col] if source_col < len(row) else None
                    output_val = row.iloc[output_col] if output_col < len(row) else None
                else:
                    # 如果是字串欄位名稱 (🛡️ 防禦：找不到欄位就跳過)
                    source_val = row.get(source_col, None)
                    output_val = row.get(output_col, None)

                source = self._clean(source_val)
                output = self._clean(output_val)

                if not source or not output:
                    continue

                # 例如：媽媽+姊姊|媽媽+妹妹 -> 阿姨
                for alt in source.split(alt_sep):
                    tokens = [t.strip() for t in alt.replace("＋", "+").split(token_sep) if t.strip()]
                    if len(tokens) < 2:
                        continue

                    for key in expanded_patterns(tokens, strip_suffix=False):
                        if key in self.rules_exact and self.rules_exact[key] != output:
                            print(f"⚠️ [複合詞] 精確規則衝突：{key} 原本對應 {self.rules_exact[key]!r}，被 {output!r} 覆蓋")
                        self.rules_exact[key] = output
                    for key in expanded_patterns(tokens, strip_suffix=True):
                        self.rules_canonical[key] = output
                    self.max_len = max(self.max_len, len(tokens))

            self.loaded_ok = True
            print(
                f"[複合詞] 成功載入 {len(self.rules_exact)} 條精確規則、"
                f"{len(self.rules_canonical)} 條一般規則 (Max Len: {self.max_len})"
            )

        except Exception as e:
            raise RuntimeError(f"複合詞初始化失敗: {e}") from e

    @staticmethod
    def _clean(value) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        return text

    def resolve_tail(self, raw_words: Sequence[str]) -> Optional[CompoundMatch]:
        """
        raw_words 必須是保留 _A/_B/_N/_S 尾碼的原始詞（例如 main.py 的
        translator.raw_word_buffer），才能讓精確比對分辨語意不同的變體。
        """
        limit = min(self.max_len, len(raw_words))

        # 從最長開始比對，避免短規則蓋掉長規則
        for size in range(limit, 1, -1):
            original_pattern = tuple(raw_words[-size:])

            # 先試精確比對（保留尾碼），例如 要_N/要_S 要對應到不同輸出。
            for pattern in expanded_patterns(original_pattern, strip_suffix=False):
                if pattern in self.rules_exact:
                    return CompoundMatch(
                        output=self.rules_exact[pattern],
                        consumed=size,
                        pattern=original_pattern,
                    )

            # 精確比對沒有結果，才退回去尾碼的寬鬆比對。
            for pattern in expanded_patterns(original_pattern, strip_suffix=True):
                if pattern in self.rules_canonical:
                    return CompoundMatch(
                        output=self.rules_canonical[pattern],
                        consumed=size,
                        pattern=original_pattern,
                    )

        return None
