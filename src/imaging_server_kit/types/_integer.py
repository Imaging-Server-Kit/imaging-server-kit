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
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
            default=default,
            auto_call=auto_call,
            min=min,
            max=max,
            step=step,
            **kwargs,
        )