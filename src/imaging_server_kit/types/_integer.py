from typing import Dict, Optional
import numpy as np

from imaging_server_kit.core.tiling import TileMeta
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
        tile_meta: Optional[TileMeta] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
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
    def serialize(cls, data: Optional[int], client_origin: str):
        if data is not None:
            return int(data)

    @classmethod
    def deserialize(cls, serialized_data: Optional[str], client_origin: str):
        if serialized_data is not None:
            return int(serialized_data)
