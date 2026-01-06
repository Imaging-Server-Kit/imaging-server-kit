from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.data_layer import DataLayer
from imaging_server_kit.core.tiling import TileMeta


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
        tile_meta: Optional[TileMeta] = None,
    ):
        self.rgb = rgb
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
        )
        self.dimensionality = (
            dimensionality if dimensionality is not None else np.arange(6).tolist()
        )
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

    @property
    def _pixel_domain(self) -> Optional[Tuple]:
        if self.data is None:
            return
        if self.rgb:
            return (self.data.shape[0], self.data.shape[1])
        else:
            return self.data.shape

    def get_tile(self, tile_meta: TileMeta) -> Optional[Image]:
        if (tile_meta.coords_max > np.asarray(self.pixel_domain)).any():
            print("Could not get an image tile from that tile meta.")
            return
        
        tile_data = self.data[tile_meta.slices] if self.data is not None else None
        return Image(
            data=tile_data,
            name=self.name,
            meta=self.meta,
            tile_meta=tile_meta,
        )

    def merge_tile(self, image_tile: Image) -> None:
        if (self.data is not None) and (image_tile.tile_meta is not None):
            self.data[image_tile.tile_meta.slices] = (
                self.data[image_tile.tile_meta.slices]
                + image_tile.data / image_tile.tile_meta.overlap_count_map
            )
        else:
            raise RuntimeError("Invalid attempt to merge an image tile.")

    @classmethod
    def serialize(cls, data: Optional[np.ndarray], client_origin: str):
        if data is not None:
            return encode_contents(data.astype(np.float32))

    @classmethod
    def deserialize(
        cls, serialized_data: Optional[Union[np.ndarray, str]], client_origin: str
    ):
        if serialized_data is None:
            return None
        if isinstance(serialized_data, str):
            serialized_data = decode_contents(serialized_data)
        return serialized_data.astype(float)

    @classmethod
    def _get_initial_data(
        cls, pixel_domain: Optional[Union[Tuple, List]]
    ) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros(pixel_domain, dtype=np.float32)

    @classmethod
    def initialize(cls, pixel_domain: Union[Tuple, List]) -> Image:
        return cls(data=cls._get_initial_data(pixel_domain))
    
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
