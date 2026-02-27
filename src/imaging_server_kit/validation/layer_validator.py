from typing import Dict,Type
from imaging_server_kit.types import DataLayer
from imaging_server_kit.validation.validator import Validator, DefaultValidator
from imaging_server_kit.validation._boxes_validator import BoxesValidator
from imaging_server_kit.validation._image_validator import ImageValidator
from imaging_server_kit.validation._mask_validator import MaskValidator
from imaging_server_kit.validation._points_validator import PointsValidator
from imaging_server_kit.validation._vectors_validator import VectorsValidator


LAYER_VALIDATORS: Dict[str, Dict[str, Type[Validator]]] = {
    "image": {"default": ImageValidator},
    "mask": {"default": MaskValidator},
    "points": {"default": PointsValidator},
    "boxes": {"default": BoxesValidator},
    "vectors": {"default": VectorsValidator},
}


def find_layer_validator(layer: DataLayer) -> Validator:
    if layer.kind in LAYER_VALIDATORS:
        lv = LAYER_VALIDATORS[layer.kind]
        validator_cls = lv.get(layer.validator, DefaultValidator)
    else:
        validator_cls = DefaultValidator

    return validator_cls()


class LayerValidator:
    @classmethod
    def validate(cls, v, layer: DataLayer) -> None:
        """Validate a layer's internal attributes."""
        if layer.data is None:
            return
        
        validator = find_layer_validator(layer)
        validator.validate(layer)
