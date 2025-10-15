from typing import Dict, Optional

from imaging_server_kit.types.data_layer import DataLayer


class Integer(DataLayer):
    def __init__(
        self,
        data: Optional[int] = None,
        name="Int",
        description="Numeric parameter (integer)",
        min: int = 0,
        max: int = 1000,
        step: int = 1,
        default: int = 0,
        auto_call: bool = False,
        meta: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
        )
        self.kind = "int"
        self.type = int
        self.min = min
        self.max = max
        self.step = step
        self.default = default
        self.auto_call = auto_call

    @classmethod
    def to_features(cls, data):
        return int(data)

    @classmethod
    def to_data(cls, features):
        return int(features)
