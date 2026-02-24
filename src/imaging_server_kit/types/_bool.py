from typing import Dict, Optional

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer


class Bool(DataLayer):
    """Data layer used to represent boolean values."""

    kind = "bool"
    type = bool

    def __init__(
        self,
        data: Optional[bool] = None,
        name="Bool",
        description="Boolean parameter",
        default: bool = False,
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
            **kwargs,
        )
