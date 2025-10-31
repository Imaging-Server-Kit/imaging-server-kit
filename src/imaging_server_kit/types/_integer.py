from typing import Dict, Optional
import numpy as np

from imaging_server_kit.types.data_layer import DataLayer


class Integer(DataLayer):
    """Data layer used to represent integer values."""

    kind = "int"
    type = int

    def __init__(
        self,
        data: Optional[int] = None,
        name="Int",
        description="Numeric parameter (integer)",
        default: int = 0,
        auto_call: bool = False,
        min: int = int(np.iinfo(np.int16).min),
        max: int = int(np.iinfo(np.int16).max),
        step: int = 1,
        meta: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
        )
        self.default = default
        self.auto_call = auto_call
        self.min = min
        self.max = max
        self.step = step
        
        # Schema contributions
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
    def serialize(cls, data, client_origin):
        return int(data)

    @classmethod
    def deserialize(cls, serialized_data, client_origin):
        return int(serialized_data)
