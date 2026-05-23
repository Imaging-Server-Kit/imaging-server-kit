from typing import Dict, Type
from imaging_server_kit.types import Layer
from imaging_server_kit.validation.validator import Validator, DefaultValidator
from imaging_server_kit.validation._boxes_validator import BoxesValidator
from imaging_server_kit.validation._image_validator import ImageValidator
from imaging_server_kit.validation._mask_validator import MaskValidator
from imaging_server_kit.validation._points_validator import PointsValidator
from imaging_server_kit.validation._vectors_validator import VectorsValidator


LAYER_VALIDATORS: Dict[str, Type[Validator]] = {
    "image": ImageValidator,
    "mask": MaskValidator,
    "points": PointsValidator,
    "boxes": BoxesValidator,
    "vectors": VectorsValidator,
}


def find_layer_validator(layer: Layer) -> Validator:
    validator_cls = LAYER_VALIDATORS.get(layer.kind, DefaultValidator)
    
    return validator_cls()


class LayerValidator:
    @classmethod
    def validate(cls, v, layer: Layer) -> None:
        """Validate a layer's internal attributes."""
        if layer.data is None:
            return

        validator = find_layer_validator(layer)
        validator.validate(layer)
