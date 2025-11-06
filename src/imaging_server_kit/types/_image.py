from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.data_layer import DataLayer


def _get_tile_slices(tile_info: Dict):
    tile_info = tile_info["tile_params"]
    ndim = tile_info["ndim"]
    tile_sizes = [tile_info[f"tile_size_{idx}"] for idx in range(ndim)]
    tile_positions = [tile_info[f"pos_{idx}"] for idx in range(ndim)]
    tile_max_positions = [
        tile_pos + tile_size
        for (tile_pos, tile_size) in zip(tile_positions, tile_sizes)
    ]
    slices = tuple(
        slice(pos, max_pos) for pos, max_pos in zip(tile_positions, tile_max_positions)
    )
    return slices


class Image(DataLayer):
    """Data layer used to represent images and image-like data.

    Parameters
    ----------
    data: Numpy arrays.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    rgb: Set to True for RGB images.
    """

    kind = "image"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Image",
        description="Input image (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        required: bool = True,  # When set to True, triggers a parameter validation error if image is None
        rgb: bool = False,
        meta: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
        )
        self.dimensionality = (
            dimensionality if dimensionality is not None else np.arange(6).tolist()
        )
        self.rgb = rgb
        self.required = required

        # Schema contributions
        main = {}
        if not self.required:
            self.default = None
            main["default"] = self.default
        extra = {
            "dimensionality": self.dimensionality,
            "rgb": self.rgb,
            "required": self.required,
        }
        self.constraints = [main, extra]

        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

    def pixel_domain(self):
        if self.data is None:
            return
        if self.rgb:
            return (self.data.shape[0], self.data.shape[1])
        else:
            return self.data.shape

    def get_tile(self, tile_info: Dict) -> Tuple[Optional[np.ndarray], Dict]:
        tile_slices = _get_tile_slices(tile_info)
        if self.data is not None:
            tile_data = self.data[tile_slices]
        else:
            tile_data = None
        return tile_data, self.meta

    def merge_tile(self, image_tile: np.ndarray, tile_info: Dict):
        tile_slices = _get_tile_slices(tile_info)
        # TODO: implement a linear blending strategy instead of this.
        if self.data is not None:
            self.data[tile_slices] = image_tile
        # For this, we'd need a mask representing how many tiles are already overlapping at a given pixel,
        # so that we can compute a running average.

    @classmethod
    def serialize(cls, data: Optional[np.ndarray], client_origin: str):
        if data is not None:
            return encode_contents(data.astype(np.float32))

    @classmethod
    def deserialize(cls, serialized_data: Optional[Union[np.ndarray, str]], client_origin:str):
        if serialized_data is None:
            return None
        if isinstance(serialized_data, str):
            serialized_data = decode_contents(serialized_data)
        return serialized_data.astype(float)

    @classmethod
    def _get_initial_data(cls, pixel_domain: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros(pixel_domain, dtype=np.float32)

    @classmethod
    def validate_data(cls, data, meta, constraints):
        main, extra = constraints
        if extra["required"] is False:
            return

        assert isinstance(
            data, np.ndarray
        ), f"Image data ({type(data)}) is not a Numpy array"

        if not all(data.shape):
            raise ValueError("Image array has an invalid shape: ", data.shape)

        if len(data.shape) not in extra["dimensionality"]:
            raise ValueError("Image array has the wrong dimensionality.")

        if extra["rgb"] is True:
            if len(data.shape) != 3:
                raise ValueError("Image should be RGB(A).")
            if data.shape[2] not in [3, 4]:
                raise ValueError("Image should be RGB(A).")
