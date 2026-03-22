from abc import ABC, abstractmethod

from imaging_server_kit.types.data_layer import DataLayer


class Merger(ABC):
    @staticmethod
    @abstractmethod
    def merge(receiving_layer: DataLayer, incoming_layer: DataLayer) -> None: ...

    @staticmethod
    @abstractmethod
    def on_first_merge(receiving_layer: DataLayer, incoming_layer: DataLayer): ...

    @staticmethod
    @abstractmethod
    def on_last_merge(receiving_layer: DataLayer, incoming_layer: DataLayer): ...


class DefaultMerger(Merger):
    @staticmethod
    def merge(receiving_layer: DataLayer, incoming_layer: DataLayer) -> None:
        receiving_layer.data = incoming_layer.data
        receiving_layer.meta = incoming_layer.meta

    @staticmethod
    def on_first_merge(receiving_layer: DataLayer, incoming_layer: DataLayer):
        pass

    @staticmethod
    def on_last_merge(receiving_layer: DataLayer, incoming_layer: DataLayer):
        pass
