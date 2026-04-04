
from typing import Optional

import numpy as np

from imaging_server_kit.validation.layer_validator import Validator
from imaging_server_kit.types._image import Image


class ImageValidator(Validator):
    @staticmethod
    def validate(image: Optional[Image]) -> None:
        data = image.data        
        meta = image.meta
        
        assert isinstance(
            data, np.ndarray
        ), f"Image data ({type(data)}) is not a Numpy array"

        if not all(data.shape):
            raise ValueError("Image array has an invalid shape: ", data.shape)

        if len(data.shape) not in meta["dimensionality"]:
            raise ValueError("Image array has the wrong dimensionality.")

        # TODO: if we have a RGB movie, this will wrongly fail:
        if meta["rgb"] is True:
            if len(data.shape) not in [3, 4]:
                raise ValueError("Image should be RGB.")
            if data.shape[-1] != 3:
                raise ValueError("Image should be RGB.")