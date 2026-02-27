from typing import Dict, List

from imaging_server_kit.core import Results
from imaging_server_kit.remote.layer_serializer import LayerSerializer


class ResultsSerializer:
    @staticmethod
    def serialize(results: Results, client_origin: str) -> List[Dict]:
        """Serialize a layer stack to JSON-compatible representation."""
        layer_serializer = LayerSerializer()
        serialized_results = []
        for layer in results.layers:
            serialized_layer = layer_serializer.serialize(layer, client_origin)
            serialized_results.append(serialized_layer)
        return serialized_results

    @staticmethod
    def deserialize(serialized_results: List[Dict], client_origin: str) -> Results:
        layer_serializer = LayerSerializer()
        layers = []
        for serialized_layer in serialized_results:
            layer = layer_serializer.deserialize(serialized_layer, client_origin)
            layers.append(layer)
        return Results(layers=layers)
