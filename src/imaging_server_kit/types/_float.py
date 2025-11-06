from typing import Dict, Optional
import numpy as np

from imaging_server_kit.types.data_layer import DataLayer


class Float(DataLayer):
    """Data layer used to represent floating-point (decimal) values."""

    kind = "float"
    type = float

    def __init__(
        self,
        data: Optional[float] = None,
        name="Float",
        description="Numeric parameter (floating point)",
        min: float = float(np.finfo(np.float32).min),
        max: float = float(np.finfo(np.float32).max),
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
        self.min = min
        self.max = max
        self.step = step
        self.default = default
        self.auto_call = auto_call
        
        main = {
            "default": self.default,
            "ge": self.min,
            "le": self.max,
        }
        extra = {
            "auto_call": self.auto_call,
            "step": self.step,
        }
        self.constraints = [main, extra]
        
        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

    @classmethod
    def serialize(cls, data: Optional[float], client_origin: str):
        if data is not None:
            return float(data)

    @classmethod
    def deserialize(cls, serialized_data: Optional[float], client_origin: str):
        if serialized_data is not None:
            return float(serialized_data)
