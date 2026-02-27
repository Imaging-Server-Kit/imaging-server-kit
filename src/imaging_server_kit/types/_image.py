from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np

from imaging_server_kit.types.data_layer import DataLayer, DefaultMerger, Merger
from imaging_server_kit.core.tiling import TileMeta


class ImageOverrideMerger(DefaultMerger):
    """Merge images using and `override` strategy: last tile overrides existing data in overlapping regions."""

    def merge(self, src_layer: Image, dst_layer: Image) -> None:
        if (
            (dst_layer.data is None)
            or (dst_layer.tile_meta is None)
            or (dst_layer.pixel_domain is None)
        ):
            return

        if (
            (src_layer.data is None)
            or (src_layer.tile_meta is None)
            or (src_layer.pixel_domain is None)
        ):
            src_layer.data = dst_layer.initialize([1] * dst_layer.ndim)
            src_layer.meta = dst_layer.meta

        if (src_layer.pixel_domain is None) or (src_layer.tile_meta is None):
            return  # This should never happen (just there for type hints)

        _slices = dst_layer.tile_meta.slices
        if _slices is not None:
            _stack = np.stack([src_layer.pixel_domain, dst_layer.pixel_domain])
            _pixel_domain = np.max(_stack, axis=0).tolist()
            # If the incoming tile extends the pixel domain, we create a new Image,
            # write src_layer.data into it, then merge the tile
            if _pixel_domain != src_layer.pixel_domain:
                new_data = dst_layer.initialize(_pixel_domain)
                new_data[src_layer.tile_meta.slices] = src_layer.data
            else:
                new_data = src_layer.data

            # New tile overrides existing data
            new_data[_slices] = dst_layer.data  # type: ignore
            src_layer.data = new_data
            src_layer.meta = dst_layer.meta

    def first_tile_hook(self, src_layer: Image, dst_layer: Image):
        src_layer.meta = dst_layer.meta


class ImageTileOverlapMerger(DefaultMerger):
    """Merge images while averaging image intensities in overlapping regions."""

    def merge(self, src_layer: Image, dst_layer: Image) -> None:
        if (
            (dst_layer.data is None)
            or (dst_layer.tile_meta is None)
            or (dst_layer.pixel_domain is None)
        ):
            return

        if (
            (src_layer.data is None)
            or (src_layer.tile_meta is None)
            or (src_layer.pixel_domain is None)
        ):
            src_layer.data = dst_layer.initialize([1] * dst_layer.ndim)
            src_layer.meta = dst_layer.meta

        if (src_layer.pixel_domain is None) or (src_layer.tile_meta is None):
            return  # This should never happen (just there for type hints)

        _overlap_count_map = dst_layer.tile_meta.overlap_count_map

        _slices = dst_layer.tile_meta.slices
        if (_slices is not None) and (_overlap_count_map is not None):
            _stack = np.stack([src_layer.pixel_domain, dst_layer.pixel_domain])
            _pixel_domain = np.max(_stack, axis=0).tolist()  # (x, y)
            # If the incoming tile extends the pixel domain, we create a new Image,
            # write src_layer.data into it, then merge the tile
            if _pixel_domain != src_layer.pixel_domain:
                new_data = dst_layer.initialize(_pixel_domain)
                new_data[src_layer.tile_meta.slices] = src_layer.data
            else:
                new_data = src_layer.data
            # We `add` the incoming image data to merge it cleanly
            # Add fake dims (fixes the RGB case)
            _overlap_count_map_dims_matched = _overlap_count_map.reshape(
                _overlap_count_map.shape
                + (1,) * (dst_layer.data.ndim - _overlap_count_map.ndim)
            )
            new_data[_slices] = new_data[_slices] + dst_layer.data / _overlap_count_map_dims_matched  # type: ignore
            src_layer.data = new_data
            src_layer.meta = dst_layer.meta

    def first_tile_hook(self, src_layer: Image, dst_layer: Image):
        # Re-initialize image data on first tile to avoid accumulating data indefinitely on multiple runs
        src_layer.data = dst_layer.initialize([1] * dst_layer.ndim)
        src_layer.meta = dst_layer.meta


class Image(DataLayer):
    """Data layer used to represent images and image-like data.

    Parameters
    ----------
    data: Numpy arrays.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    rgb: Set to True for RGB images.
    """

    kind = "image"
    mergers: Dict[str, Type[Merger]] = {
        "default": ImageTileOverlapMerger,
        "override": ImageOverrideMerger,
    }

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Image",
        description="Input image (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        merger: str = "default",
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        rgb: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
            description=description,
            dimensionality=dimensionality,
            merger=merger,
            rgb=rgb,
            **kwargs,
        )

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        if self._data is None:
            return
        if self.meta is None:
            return
        if self.meta["rgb"] is True:
            return (self._data.shape[0], self._data.shape[1])
        else:
            return self._data.shape

    def get_tile(self, tile_meta: TileMeta) -> Image:
        if (
            (self.data is None)
            or (tile_meta.coords_max is None)
            or (self.pixel_domain is None)
        ):
            _data = None
        else:
            _data = self.data[tile_meta.slices]

        return Image(
            data=_data,
            name=self.name,
            meta=self.meta,
            tile_meta=tile_meta,
        )

    @staticmethod
    def _get_initial_data(
        pixel_domain: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros(pixel_domain, dtype=np.float32)

    def initialize(self, domain: List[int]) -> Optional[np.ndarray]:
        if self.meta["rgb"] is True:
            domain.extend([3])
        return self._get_initial_data(domain)