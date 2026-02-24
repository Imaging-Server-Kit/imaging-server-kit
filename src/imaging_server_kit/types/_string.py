from typing import Dict, Optional

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer


class String(DataLayer):
    """Data layer used to represent strings of text."""

    kind = "str"
    type = str

    def __init__(
        self,
        data: Optional[str] = None,
        name="String",
        description="String parameter",
        default: str = "",
        required: bool = True,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            data=data,
            meta=meta,
            tile_meta=tile_meta,
            description=description,
            default=default,
            required=required,
            **kwargs,
        )