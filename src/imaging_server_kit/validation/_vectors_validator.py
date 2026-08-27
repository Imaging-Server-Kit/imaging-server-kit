from typing import Optional

import numpy as np

from imaging_server_kit.validation.layer_validator import Validator
from imaging_server_kit.types._vectors import Vectors


class VectorsValidator(Validator):
    @staticmethod
    def validate(vectors: Optional[Vectors]) -> None:
        if vectors is None:
            return

        data = vectors.data

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Vectors data ({type(data)}) is not a Numpy array")

        if (len(data.shape) != 3) or (data.shape[1] != 2):
            raise ValueError("Vectors data should have shape (N, 2, D)")
