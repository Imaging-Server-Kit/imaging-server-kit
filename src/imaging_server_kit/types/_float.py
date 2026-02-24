from typing import Dict, Optional
import numpy as np

from imaging_server_kit.core.tiling import TileMeta
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
        required: bool = True,
        auto_call: bool = False,
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
            required=required,
            auto_call=auto_call,
            min=min,
            max=max,
            step=step,
            **kwargs,
        )