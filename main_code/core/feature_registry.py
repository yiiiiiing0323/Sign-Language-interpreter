import sys
sys.dont_write_bytecode = True

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class FeatureInfo:
    """
    單一 feature 的 metadata。

    name:
        A 流輸出的特徵名稱，也會被 Excel 規則引用。
    dtype:
        預期型別，例如 bool / float / int。
    default:
        沒有偵測到對應資料時的預設值。
    description:
        人類可讀的說明，方便未來產生文件或檢查 Excel 規則。
    tensor_index:
        若該 feature 也要進入 AI tensor，可在這裡指定固定位置。
    """
    name: str
    dtype: type
    default: Any
    description: str = ""
    tensor_index: Optional[int] = None


class FeatureRegistry:
    """
    特徵登錄中心。

    原本 A 流與 Excel 規則都直接手寫 feature 字串，容易發生：
    - 拼字錯誤
    - Excel 規則引用不存在的特徵
    - 預設值不一致
    - AI tensor 維度漂移

    這個 registry 提供集中管理位置：
    - 註冊 feature 名稱、型別、預設值
    - 建立每一幀的 feature dict
    - 提供空白 AI tensor
    - 未來可用來驗證 Excel 規則與產生文件
    """

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
        """手動註冊一個 feature；若名稱已存在，會以新的設定覆蓋。"""
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
        """從一批預設值批次註冊 feature，適合承接舊版 A 流的巨大預設 dict。"""
        for name, default in defaults.items():
            self.register(
                name=name,
                dtype=self._infer_dtype(default),
                default=default,
                description="Imported from A-stream default feature set.",
            )

    def create_feature_dict(self, seed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        建立目前影格要使用的 feature dict。

        seed 是 A 流既有的預設特徵表。registry 會先把它註冊起來，
        再回傳一份包含預設值的 dict。這樣舊程式可以維持原本寫法，
        新架構也能逐步接管 feature 管理。
        """
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
        """
        確保某個 feature 已存在。

        若 Excel 或未來模組先引用了 feature，但 registry 尚未註冊，
        會依命名慣例推測預設值，例如 dist_ 開頭預設 99.0，
        is_/detect_/move_ 開頭預設 False。
        """
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
        """產生一個全 0 的 AI tensor，用於沒有偵測到 pose/hand 時維持 LSTM 輸入維度。"""
        return [0.0] * self.tensor_dim

    @staticmethod
    def default_for_name(name: str) -> Any:
        """依 feature 命名慣例推測合理預設值。"""
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
