from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core.domain import Domain


class Image(Layer):
    """Data layer used to represent images and image-like data.

    Parameters
    ----------
    data: Numpy arrays.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    rgb: Set to True for RGB images.
    channel_axis: Optional index of the channel axis.
      - The channel axis does not affect the `bounds`, `ndim`, and `domain` attributes.
      - The channel axis is set to `2` if rgb is True and there is no time axis.
      - tile_size along the channel axis defaults to the length of this axis.
    """

    kind = "image"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Image",
        description="Input image (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        rgb: bool = False,
        channel_axis: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            name=name,
            description=description,
            dimensionality=dimensionality,
            rgb=rgb,
            channel_axis=channel_axis,
            **kwargs,
        )

    @property
    def channel_axis(self):
        if self.meta["rgb"] is True:
            if self.data is not None:
                if self.data.ndim == 4:
                    return 3
                else:
                    return 2
            else:
                return 2

        if self.meta["channel_axis"] is not None:
            return self.meta["channel_axis"]

    @property
    def _bounds(self) -> Optional[Tuple]:
        """Data bounds in local coordinates."""
        if self._data is None:
            return

        if self.meta is None:
            return

        if self.channel_axis is not None:
            shape = list(self._data.shape)
            shape.pop(self.channel_axis)
            bounds_min = tuple([0] * len(shape))
            bounds_max = tuple(shape)
        else:
            bounds_min = tuple([0] * len(self._data.shape))
            bounds_max = tuple(self._data.shape)

        return (bounds_min, bounds_max)

    def select(self, domain: Domain) -> Image:
        """Select data in a given domain."""
        if (self.data is None) or (domain.size is None):
            _data = None
        elif (domain.coords_max > np.asarray(self.coords_max)).any():
            # TODO: What happens if we query a domain bigger than the image?
            _data = None
        else:
            # Get the slice indices
            cmin_rounded = [
                math.floor(v - p) for v, p in zip(domain.coords_min, self.position)
            ]
            cmax_rounded = [
                math.ceil(v - p) for v, p in zip(domain.coords_max, self.position)
            ]

            slices = tuple(
                [slice(cmin, cmax) for cmin, cmax in zip(cmin_rounded, cmax_rounded)]
            )

            # Account for the channel_axis
            if self.channel_axis:
                slices_with_channel = (
                    slices[: self.channel_axis]
                    + (slice(None),)
                    + slices[self.channel_axis :]
                )
            else:
                slices_with_channel = slices

            try:
                _data = self.data[slices_with_channel]
            except:
                raise RuntimeError(
                    "Data re-initialization in the provided domain failed. Did you pass a domain range outside of the object's domain?"
                )

        image_selection = Image(
            data=_data,
            name=self.name,
            meta=self.meta,
            tile_meta=self.tile_meta,
            position=domain.coords_min,
        )
        
        image_selection.position = domain.coords_min
        
        return image_selection

    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            if domain.size is not None:
                return np.zeros(domain.size, dtype=np.float32)

    def _reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        # Get the slice indices
        cmin_rounded = [
            math.floor(max(vmin, pmin) - pmin)
            for vmin, pmin in zip(domain.coords_min, self.coords_min)
        ]
        cmax_rounded = [
            math.ceil(min(vmax, pmax) - max(vmin, pmin))
            for vmin, vmax, pmin, pmax in zip(
                domain.coords_min, domain.coords_max, self.coords_min, self.coords_max
            )
        ]

        slices = tuple(
            [slice(cmin, cmax) for cmin, cmax in zip(cmin_rounded, cmax_rounded)]
        )

        # Account for the channel_axis
        if self.channel_axis is not None:
            slices_with_channel = (
                slices[: self.channel_axis]
                + (slice(None),)
                + slices[self.channel_axis :]
            )
        else:
            slices_with_channel = slices

        try:
            new_data = self.data.copy()
            new_data[slices_with_channel] = 0
            self.data = new_data
        except:
            raise RuntimeError(
                "Data re-initialization in the provided domain failed. Did you pass a domain range outside of the object's domain?"
            )
