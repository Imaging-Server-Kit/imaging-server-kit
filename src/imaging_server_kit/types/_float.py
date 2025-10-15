from typing import Dict, Optional

from imaging_server_kit.types.data_layer import DataLayer


class Float(DataLayer):
    def __init__(
        self,
        data: Optional[float] = None,
        name="Float",
        description="Numeric parameter (floating point)",
        min: float = 0.0,
        max: float = 1000.0,
        step: float = 0.1,
        default: float = 0.0,
        auto_call: bool = False,
        meta: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
        )
        self.kind = "float"
        self.type = float
        self.min = min
        self.max = max
        self.step = step
        self.default = default
        self.auto_call = auto_call

    @classmethod
    def to_features(cls, data):
        return float(data)

    @classmethod
    def to_data(cls, features):
        return float(features)
