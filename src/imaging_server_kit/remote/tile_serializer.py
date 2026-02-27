from typing import Dict, Optional

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.remote.data_serializer import Serializer
from imaging_server_kit.types.data_layer import DataLayer


class TileMetaSerializer(Serializer):
    @staticmethod
    def serialize(
        layer: Optional[DataLayer], client_origin: str
    ) -> Optional[Dict]:
        if layer is not None:
            if layer.tile_meta is not None:
                return layer.tile_meta.serialize()

    @staticmethod
    def deserialize(
        serialized_data: Optional[Dict], client_origin: str
    ) -> TileMeta:
        if serialized_data is not None:
            return TileMeta(**serialized_data)
        else:
            return TileMeta()
