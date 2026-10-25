from abc import ABC, abstractmethod

from imaging_server_kit.types.layer import Layer


class Merger(ABC):
    """Merging strategy of a layer kind. `LayerMerger` creates one instance per merge run (first => last tile)."""

    @abstractmethod
    def merge(self, receiving_layer: Layer, incoming_layer: Layer) -> None: ...

    @abstractmethod
    def on_first_merge(self, receiving_layer: Layer, incoming_layer: Layer): ...

    @abstractmethod
    def on_last_merge(self, receiving_layer: Layer, incoming_layer: Layer): ...


class DefaultMerger(Merger):
    def merge(self, receiving_layer: Layer, incoming_layer: Layer) -> None:
        receiving_layer.data = incoming_layer.data
        receiving_layer.meta = incoming_layer.meta

    def on_first_merge(self, receiving_layer: Layer, incoming_layer: Layer):
        pass

    def on_last_merge(self, receiving_layer: Layer, incoming_layer: Layer):
        pass
