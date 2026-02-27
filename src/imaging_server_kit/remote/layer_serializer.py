from typing import Any, Dict, Type

from imaging_server_kit.types import DataLayer, DATA_TYPES
from imaging_server_kit.remote.meta_serializer import DefalutMetaSerializer
from imaging_server_kit.remote.tile_serializer import TileMetaSerializer
from imaging_server_kit.remote.data_serializer import Serializer, DefaultDataSerializer
from imaging_server_kit.remote._image_serializer import ImageDataSerializer
from imaging_server_kit.remote._mask_serializer import MaskDataSerializer
from imaging_server_kit.remote._boxes_serializer import BoxesDataSerializer
from imaging_server_kit.remote._paths_serializer import PathsDataSerializer
from imaging_server_kit.remote._points_serializer import PointsDataSerializer
from imaging_server_kit.remote._tracks_serializer import TracksDataSerializer
from imaging_server_kit.remote._vectors_serializer import VectorsDataSerializer
from imaging_server_kit.remote._null_serializer import NullDataSerializer\


LAYER_DATA_SERIALIZERS: Dict[str, Dict[str, Type[Serializer]]] = {
    "image": {"default": ImageDataSerializer},
    "mask": {"default": MaskDataSerializer},
    "points": {"default": PointsDataSerializer},
    "boxes": {"default": BoxesDataSerializer},
    "paths": {"default": PathsDataSerializer},
    "tracks": {"default": TracksDataSerializer},
    "vectors": {"default": VectorsDataSerializer},
    "null": {"default": NullDataSerializer},
}


def find_layer_serializer(layer: DataLayer) -> Serializer:
    if layer.kind in LAYER_DATA_SERIALIZERS:
        lds = LAYER_DATA_SERIALIZERS[layer.kind]
        data_serializer_cls = lds.get(layer.data_serializer, DefaultDataSerializer)
    else:
        data_serializer_cls = DefaultDataSerializer

    return data_serializer_cls()


class LayerSerializer(Serializer):
    @staticmethod
    def serialize(layer: DataLayer, client_origin: str) -> Dict[str, Any]:
        """Serialize a layer."""

        data_serializer = find_layer_serializer(layer)
        serialized_data = data_serializer.serialize(layer, client_origin)

        meta_serializer = DefalutMetaSerializer()
        serialized_meta = meta_serializer.serialize(layer, client_origin)

        tile_serializer = TileMetaSerializer()
        serialized_tile_meta = tile_serializer.serialize(layer, client_origin)

        return {
            "kind": layer.kind,
            "data": serialized_data,
            "name": layer.name,
            "meta": serialized_meta,
            "tile_meta": serialized_tile_meta,
            "merger": layer.merger_type,
            "data_serializer": layer.data_serializer,
        }

    @staticmethod
    def deserialize(serialized_layer: Dict[str, Any], client_origin: str) -> DataLayer:
        """Deserialize a layer."""
        kind = serialized_layer["kind"]
        name = serialized_layer["name"]
        encoded_data = serialized_layer["data"]
        encoded_meta = serialized_layer["meta"]
        encoded_tile_meta = serialized_layer["tile_meta"]
        merger_type = serialized_layer["merger"]
        data_serializer = serialized_layer["data_serializer"]
        
        cls: Type[DataLayer] = DATA_TYPES[kind]
        layer_proto = cls(data_serializer=data_serializer)
        layer_serializer = find_layer_serializer(layer_proto)
        data = layer_serializer.deserialize(encoded_data, client_origin)
        
        meta_serializer = DefalutMetaSerializer()
        meta = meta_serializer.deserialize(encoded_meta, client_origin)
        
        tile_serializer = TileMetaSerializer()
        tile_meta = tile_serializer.deserialize(encoded_tile_meta, client_origin)
        
        return cls(
            data=data,
            name=name,
            meta=meta,
            tile_meta=tile_meta,
            merger=merger_type,
            data_serializer=data_serializer,
        )
