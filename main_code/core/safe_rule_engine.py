import sys
sys.dont_write_bytecode = True

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set


@dataclass
class RuleEvaluation:
    """
    單一規則的解析結果。

    matched:
        這條規則在目前影格是否成立。
    confidence:
        規則信心分數，範圍 0.0 到 1.0。一般比較式或布林特徵成立時為 1.0；
        複合規則會依 and/or/not 組合後得到整體分數。
    matched_terms / total_terms:
        用來除錯規則命中比例，例如 3 個條件中命中 2 個。
    unknown_names:
        Excel 規則中出現、但 A 流目前沒有提供的特徵名稱。
    error:
        語法錯誤或不被允許的表達式說明。
    """
    matched: bool
    confidence: float = 0.0
    matched_terms: int = 0
    total_terms: int = 0
    unknown_names: Set[str] = field(default_factory=set)
    error: str = ""


class SafeRuleEvaluator:
    """
    安全的 Excel 規則解析器。

    這個類別取代原本的 eval()，避免 Excel 規則欄位被寫入任意 Python 程式碼。
    允許的語法只有：
    - and / or / not
    - 小括號
    - True / False
    - 特徵名稱，例如 is_flat_HAND
    - 比較式，例如 dist_HAND_8_FACE_1 < 0.5

    不允許的語法包括：
    - function()
    - object.attribute
    - import / __import__
    - list / dict / subscript
    - 任意 Python 執行

    因此 Excel 規則仍然好寫，但不會變成可執行任意程式碼的入口。
    """

    _BOOL_REPLACEMENTS = (
        (re.compile(r"\bTRUE\b", re.IGNORECASE), "True"),
        (re.compile(r"\bFALSE\b", re.IGNORECASE), "False"),
    )

    def __init__(self, known_features: Optional[Iterable[str]] = None):
        self.known_features = set(known_features or [])

    def evaluate(self, expression: str, variables: Dict[str, Any]) -> RuleEvaluation:
        """
        解析並計算一條規則。

        expression:
            Excel 中的條件字串。
        variables:
            A 流目前影格輸出的 feature dict。

        回傳 RuleEvaluation，而不是只回傳 True/False，
        讓 B 流可以同時取得命中結果、信心分數、未知特徵與錯誤訊息。
        """
        if not expression:
            return RuleEvaluation(False, error="empty expression")

        normalized = self._normalize(expression)
        try:
            tree = ast.parse(normalized, mode="eval")
            return self._eval_node(tree.body, variables)
        except SyntaxError as exc:
            return RuleEvaluation(False, error=f"syntax error: {exc.msg}")
        except ValueError as exc:
            return RuleEvaluation(False, error=str(exc))
        except Exception as exc:
            return RuleEvaluation(False, error=f"rule evaluation failed: {exc}")

    def names_in(self, expression: str) -> Set[str]:
        try:
            tree = ast.parse(self._normalize(expression), mode="eval")
        except SyntaxError:
            return set()
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def _eval_node(self, node: ast.AST, variables: Dict[str, Any]) -> RuleEvaluation:
        # BoolOp 對應 and/or。這裡逐一解析每個子條件，再依運算子合成結果。
        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(value, variables) for value in node.values]
            if isinstance(node.op, ast.And):
                matched = all(value.matched for value in values)
                confidence = self._avg(value.confidence for value in values)
            elif isinstance(node.op, ast.Or):
                matched = any(value.matched for value in values)
                confidence = max((value.confidence for value in values), default=0.0)
            else:
                raise ValueError("unsupported boolean operator")
            return self._combine(matched, confidence, values)

        # not A 會反轉 A 的 matched，信心分數也做反向處理。
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self._eval_node(node.operand, variables)
            matched = not value.matched
            return RuleEvaluation(
                matched=matched,
                confidence=1.0 - value.confidence if value.total_terms else float(matched),
                matched_terms=1 if matched else 0,
                total_terms=max(1, value.total_terms),
                unknown_names=value.unknown_names,
                error=value.error,
            )

        # 比較式只允許左右兩側為「特徵名稱」或「常數」。
        # 例如 dist_HAND_8_FACE_1 < 0.5 是合法的，danger() < 1 不合法。
        if isinstance(node, ast.Compare):
            left = self._value(node.left, variables)
            result = True
            last_value = left
            unknown_names = self._unknown_names(node.left, variables)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._value(comparator, variables)
                unknown_names.update(self._unknown_names(comparator, variables))
                result = result and self._compare(last_value, right, op)
                last_value = right
            return RuleEvaluation(
                matched=bool(result),
                confidence=1.0 if result else 0.0,
                matched_terms=1 if result else 0,
                total_terms=1,
                unknown_names=unknown_names,
            )

        # 單獨的特徵名稱代表「這個布林特徵是否為真」。
        # 若 A 流沒有提供該名稱，會視為 False 並記到 unknown_names。
        if isinstance(node, ast.Name):
            value = bool(variables.get(node.id, False))
            unknown_names = {node.id} if node.id not in variables else set()
            return RuleEvaluation(
                matched=value,
                confidence=1.0 if value else 0.0,
                matched_terms=1 if value else 0,
                total_terms=1,
                unknown_names=unknown_names,
            )

        # 常數主要支援 True / False，也保留數字與字串常數給比較式使用。
        if isinstance(node, ast.Constant):
            value = bool(node.value)
            return RuleEvaluation(
                matched=value,
                confidence=1.0 if value else 0.0,
                matched_terms=1 if value else 0,
                total_terms=1,
            )

        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    def _value(self, node: ast.AST, variables: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Name):
            return variables.get(node.id, False)
        if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float, str)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._value(node.operand, variables)
            if isinstance(value, (int, float)):
                return -value
        raise ValueError(f"unsupported comparison value: {type(node).__name__}")

    def _unknown_names(self, node: ast.AST, variables: Dict[str, Any]) -> Set[str]:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id not in variables
        }

    @staticmethod
    def _compare(left: Any, right: Any, op: ast.cmpop) -> bool:
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        raise ValueError("unsupported comparison operator")

    @classmethod
    def _normalize(cls, expression: str) -> str:
        normalized = str(expression).strip()
        for pattern, replacement in cls._BOOL_REPLACEMENTS:
            normalized = pattern.sub(replacement, normalized)
        return normalized

    @staticmethod
    def _avg(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _combine(matched: bool, confidence: float, values: Iterable[RuleEvaluation]) -> RuleEvaluation:
        values = list(values)
        unknown_names: Set[str] = set()
        errors = []
        matched_terms = 0
        total_terms = 0
        for value in values:
            unknown_names.update(value.unknown_names)
            if value.error:
                errors.append(value.error)
            matched_terms += value.matched_terms
            total_terms += value.total_terms
        return RuleEvaluation(
            matched=matched,
            confidence=confidence,
            matched_terms=matched_terms,
            total_terms=total_terms,
            unknown_names=unknown_names,
            error="; ".join(errors),
        )
