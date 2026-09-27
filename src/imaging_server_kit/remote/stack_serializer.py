from typing import Dict, List

from imaging_server_kit.core import Stack
from imaging_server_kit.remote.layer_serializer import LayerSerializer


class StackSerializer:
    @staticmethod
    def serialize(stack: Stack) -> List[Dict]:
        """Serialize a layer stack to JSON-compatible representation."""
        return [LayerSerializer.serialize(layer) for layer in stack.layers]

    @staticmethod
    def deserialize(serialized_stack: List[Dict]) -> Stack:
        layers = [LayerSerializer.deserialize(l) for l in serialized_stack]
        return Stack(layers=layers)
