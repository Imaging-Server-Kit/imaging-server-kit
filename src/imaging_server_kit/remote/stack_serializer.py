from typing import Dict, List

from imaging_server_kit.core import Stack
from imaging_server_kit.remote.layer_serializer import LayerSerializer


class StackSerializer:
    @staticmethod
    def serialize(stack: Stack) -> List[Dict]:
        """Serialize a layer stack to JSON-compatible representation."""
        layer_serializer = LayerSerializer()
        serialized_stack = []
        for layer in stack.layers:
            serialized_layer = layer_serializer.serialize(layer)
            serialized_stack.append(serialized_layer)
        return serialized_stack

    @staticmethod
    def deserialize(serialized_stack: List[Dict]) -> Stack:
        layer_serializer = LayerSerializer()
        layers = []
        for serialized_layer in serialized_stack:
            layer = layer_serializer.deserialize(serialized_layer)
            layers.append(layer)
        return Stack(layers=layers)
