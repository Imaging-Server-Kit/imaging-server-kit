from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core.domain import Domain
from imaging_server_kit.types.common import copy_meta, domain_slices


class Image(Layer):
    """Data layer for images: 2D or 3D arrays, optionally multichannel or RGB.

    Parameters
    ----------
    data : numpy.ndarray, optional
        The image array.
    name : str, default="Image"
        Name of the layer.
    description : str, default="Input image"
        Description of the layer, displayed on the algorithm documentation page.
    dimensionality : list of int, optional
        Accepted numbers of dimensions, for example `[2, 3]`. By default, any number
        of dimensions is accepted.
    rgb : bool, default=False
        Set to `True` for RGB images.
    channel_axis : int, optional
        Index of the channel axis, for multichannel images. The channel axis is not
        counted as a spatial dimension (it doesn't affect `ndim` and `extent`), and
        it is never split into tiles. With `rgb=True`, the last axis is always the
        channel axis.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).

    Examples
    --------
    >>> image = sk.Image(np.zeros((100, 200)), name="Blurred", colormap="viridis")
    """

    kind = "image"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Image",
        description="Input image",
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

    def _summary(self) -> str:
        summary = super()._summary()
        if self.data is None:
            return summary
        if self.meta.get("rgb"):
            summary += ", rgb"
        elif self.meta.get("channel_axis") is not None:
            summary += f", channel_axis={self.meta['channel_axis']}"
        return summary

    @property
    def channel_axis(self) -> Optional[int]:
        if self.meta:
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
        """Select the part of the layer inside a domain.

        Parameters
        ----------
        domain : Domain
            The region to select, in global pixel coordinates.

        Returns
        -------
        Image
            A new layer with the selected data, positioned in global coordinates.
        """
        _meta = copy_meta(self)

        if (self.data is None) or (domain.size is None):
            return Image(
                data=None,
                name=self.name,
                meta=_meta,
                tile_meta=self.tile_meta.copy(),
            )

        selection = domain_slices(self, domain)
        if selection is None:
            # No intersection
            _data = None
            position = domain.coords_min
        else:
            slices_with_channel, position = selection
            _data = self.data[slices_with_channel]

        image_selection = Image(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta.copy(),
        )

        image_selection.position = position

        return image_selection

    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            if domain.size is not None:
                return np.zeros(domain.size, dtype=np.float32)

    def _reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        if self.data is None:
            return

        selection = domain_slices(self, domain)
        if selection is None:
            # No intersection
            return

        slices_with_channel, _ = selection

        new_data = self.data.copy()
        new_data[slices_with_channel] = 0
        self.data = new_data
