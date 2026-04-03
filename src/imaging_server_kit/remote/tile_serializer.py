from typing import Dict, Optional

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.types.layer import Layer


class TileMetaSerializer(Serializer):
    @staticmethod
    def serialize(layer: Optional[Layer], client_origin: str) -> Optional[Dict]:
        if layer is not None:
            if layer.tile_meta is not None:
                return layer.tile_meta.serialize()

    @staticmethod
    def deserialize(serialized_data: Optional[Dict], client_origin: str) -> TileMeta:
        if serialized_data is not None:
            return TileMeta(**serialized_data)
        else:
            return TileMeta()
