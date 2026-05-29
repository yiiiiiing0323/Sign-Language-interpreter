from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    dtype: type
    default: Any
    description: str = ""
    tensor_index: Optional[int] = None


class FeatureRegistry:
    """Central place to register feature names, defaults, and tensor ordering."""

    def __init__(self, tensor_dim: int = 218):
        self.tensor_dim = tensor_dim
        self._features: Dict[str, FeatureInfo] = {}
        self._tensor_order = [f"ai_dim_{i:03d}" for i in range(tensor_dim)]

    def register(
        self,
        name: str,
        dtype: type = bool,
        default: Any = False,
        description: str = "",
        tensor_index: Optional[int] = None,
    ) -> FeatureInfo:
        info = FeatureInfo(
            name=name,
            dtype=dtype,
            default=default,
            description=description,
            tensor_index=tensor_index,
        )
        self._features[name] = info
        return info

    def register_from_defaults(self, defaults: Dict[str, Any]) -> None:
        for name, default in defaults.items():
            self.register(
                name=name,
                dtype=self._infer_dtype(default),
                default=default,
                description="Imported from A-stream default feature set.",
            )

    def create_feature_dict(self, seed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        values = {name: info.default for name, info in self._features.items()}
        if seed:
            for name, value in seed.items():
                if name not in self._features:
                    self.register(
                        name=name,
                        dtype=self._infer_dtype(value),
                        default=value,
                        description="Auto-registered from runtime feature seed.",
                    )
                values[name] = value
        return values

    def ensure_feature(self, name: str) -> FeatureInfo:
        if name not in self._features:
            default = self.default_for_name(name)
            self.register(
                name=name,
                dtype=self._infer_dtype(default),
                default=default,
                description="Auto-registered from feature name convention.",
            )
        return self._features[name]

    def validate_feature_names(self, names: Iterable[str]) -> List[str]:
        return sorted({name for name in names if name not in self._features})

    def tensor_order(self) -> List[str]:
        return list(self._tensor_order)

    def empty_ai_tensor(self) -> List[float]:
        return [0.0] * self.tensor_dim

    @staticmethod
    def default_for_name(name: str) -> Any:
        if name.startswith("dist_"):
            return 99.0
        if name.startswith("vector_align_") and name not in {
            "vector_align_HAND_out",
            "vector_align_HAND_PALM_INWARD",
            "vector_align_HAND_PALM_UPWARD",
            "vector_align_HAND_PALM_OPPOSITE",
            "vector_align_HAND_8_LEFT_AXIS",
        }:
            return 0.0
        if name.startswith(
            (
                "is_",
                "detect_",
                "move_",
                "palm_",
                "palms_",
                "hands_",
                "tips_",
                "fingers_",
            )
        ):
            return False
        return 0.0

    @staticmethod
    def _infer_dtype(value: Any) -> type:
        if isinstance(value, bool):
            return bool
        if isinstance(value, int) and not isinstance(value, bool):
            return int
        if isinstance(value, float):
            return float
        return type(value)


FEATURE_REGISTRY = FeatureRegistry()
AI_TENSOR_DIM = FEATURE_REGISTRY.tensor_dim

