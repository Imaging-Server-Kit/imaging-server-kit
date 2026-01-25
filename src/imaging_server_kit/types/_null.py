from typing import Any, Dict, Optional
from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer, DataSerializer


class NullDataSerializer(DataSerializer):
    def serialize(self, data, client_origin: str) -> Any:
        if data is not None:
            raise ValueError(f"Cannot serialize this object: {data}")
        return None

    def deserialize(self, serialized_data, client_origin: str) -> None:
        return None


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
        
        # Schema contributions
        main = {"default": self.default}
        extra = {}
        self.constraints = [main, extra]
        
        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

        self.data_serializer = NullDataSerializer()
