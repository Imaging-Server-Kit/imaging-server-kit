from abc import ABC, abstractmethod

from imaging_server_kit.types.layer import Layer


class Merger(ABC):
    @staticmethod
    @abstractmethod
    def merge(receiving_layer: Layer, incoming_layer: Layer) -> None: ...

    @staticmethod
    @abstractmethod
    def on_first_merge(receiving_layer: Layer, incoming_layer: Layer): ...

    @staticmethod
    @abstractmethod
    def on_last_merge(receiving_layer: Layer, incoming_layer: Layer): ...


class DefaultMerger(Merger):
    @staticmethod
    def merge(receiving_layer: Layer, incoming_layer: Layer) -> None:
        receiving_layer.data = incoming_layer.data
        receiving_layer.meta = incoming_layer.meta

    @staticmethod
    def on_first_merge(receiving_layer: Layer, incoming_layer: Layer):
        pass

    @staticmethod
    def on_last_merge(receiving_layer: Layer, incoming_layer: Layer):
        pass
