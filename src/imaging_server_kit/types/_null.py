from typing import Any, Dict, Optional, Type
from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer


class Null(DataLayer):
    """
    Data layer used to represent None or the absence of data.
    """

    kind = "null"
    type = type(None)
    
    def __init__(
        self,
        data: Optional[Any] = None,
        name="None",
        description="Null (None) type",
        default=None,
        data_serializer: str = "default",
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
            data_serializer=data_serializer,
            **kwargs,
        )