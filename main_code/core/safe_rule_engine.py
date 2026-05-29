import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set


@dataclass
class RuleEvaluation:
    matched: bool
    confidence: float = 0.0
    matched_terms: int = 0
    total_terms: int = 0
    unknown_names: Set[str] = field(default_factory=set)
    error: str = ""


class SafeRuleEvaluator:
    """Evaluate Excel boolean rules without Python eval()."""

    _BOOL_REPLACEMENTS = (
        (re.compile(r"\bTRUE\b", re.IGNORECASE), "True"),
        (re.compile(r"\bFALSE\b", re.IGNORECASE), "False"),
    )

    def __init__(self, known_features: Optional[Iterable[str]] = None):
        self.known_features = set(known_features or [])

    def evaluate(self, expression: str, variables: Dict[str, Any]) -> RuleEvaluation:
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
