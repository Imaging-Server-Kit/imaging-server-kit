from typing import Any, Dict, Type

from imaging_server_kit.types import Layer, DATA_TYPES
from imaging_server_kit.remote.meta_serializer import MetaSerializer
from imaging_server_kit.remote.tile_serializer import TileMetaSerializer
from imaging_server_kit.remote.serializer import Serializer, DefaultDataSerializer
from imaging_server_kit.remote._image_serializer import ImageDataSerializer
from imaging_server_kit.remote._mask_serializer import MaskDataSerializer
from imaging_server_kit.remote._boxes_serializer import BoxesDataSerializer
from imaging_server_kit.remote._paths_serializer import PathsDataSerializer
from imaging_server_kit.remote._points_serializer import PointsDataSerializer
from imaging_server_kit.remote._tracks_serializer import TracksDataSerializer
from imaging_server_kit.remote._vectors_serializer import VectorsDataSerializer
from imaging_server_kit.remote._null_serializer import NullDataSerializer


LAYER_DATA_SERIALIZERS: Dict[str, Type[Serializer]] = {
    "image": ImageDataSerializer,
    "mask": MaskDataSerializer,
    "points": PointsDataSerializer,
    "boxes": BoxesDataSerializer,
    "paths": PathsDataSerializer,
    "tracks": TracksDataSerializer,
    "vectors": VectorsDataSerializer,
    "null": NullDataSerializer,
}


def find_layer_serializer(layer_kind: str) -> Serializer:
    serializer_cls = LAYER_DATA_SERIALIZERS.get(layer_kind, DefaultDataSerializer)
    
    return serializer_cls()


class LayerSerializer(Serializer):
    @staticmethod
    def serialize(layer: Layer, client_origin: str) -> Dict[str, Any]:
        """Serialize a layer."""

        data_serializer = find_layer_serializer(layer.kind)
        serialized_data = data_serializer.serialize(layer, client_origin)

        meta_serializer = MetaSerializer()
        serialized_meta = meta_serializer.serialize(layer, client_origin)

        tile_serializer = TileMetaSerializer()
        serialized_tile_meta = tile_serializer.serialize(layer, client_origin)

        return {
            "kind": layer.kind,
            "data": serialized_data,
            "name": layer.name,
            "meta": serialized_meta,
            "tile_meta": serialized_tile_meta,
        }

    @staticmethod
    def deserialize(serialized_layer: Dict[str, Any], client_origin: str) -> Layer:
        """Deserialize a layer."""
        kind = serialized_layer["kind"]
        name = serialized_layer["name"]
        encoded_data = serialized_layer["data"]
        encoded_meta = serialized_layer["meta"]
        encoded_tile_meta = serialized_layer["tile_meta"]

        cls: Type[Layer] = DATA_TYPES[kind]
        layer_serializer = find_layer_serializer(kind)
        data = layer_serializer.deserialize(encoded_data, client_origin)

        meta_serializer = MetaSerializer()
        meta = meta_serializer.deserialize(encoded_meta, client_origin)

        tile_serializer = TileMetaSerializer()
        tile_meta = tile_serializer.deserialize(encoded_tile_meta, client_origin)

        return cls(
            data=data,
            name=name,
            meta=meta,
            tile_meta=tile_meta,
        )
