from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from imaging_server_kit.core.domain import Domain
from imaging_server_kit.types.layer import Layer
from imaging_server_kit.types.common import copy_meta, domain_slices


class _ArrayLayer(Layer):
    """Base class of layers holding an array of pixels: Image and Mask.

    The spatial extent is the shape of the data, without the channel axis (if any).
    """

    # Dtype of the zero-valued data created by `_zeros_in()`
    _zeros_dtype = np.float32

    @property
    def channel_axis(self) -> Optional[int]:
        if self.meta:
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

    def select(self, domain: Domain) -> _ArrayLayer:
        """Select the part of the layer inside a domain.

        Parameters
        ----------
        domain : Domain
            The region to select, in global pixel coordinates.

        Returns
        -------
        Layer
            A new layer with the selected data, positioned in global coordinates.
        """
        cls = type(self)
        _meta = copy_meta(self)

        if (self.data is None) or (domain.size is None):
            return cls(
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

        layer_selection = cls(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta.copy(),
        )

        layer_selection.position = position

        return layer_selection

    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            if domain.size is not None:
                return np.zeros(domain.size, dtype=self._zeros_dtype)

    def _reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        if self.data is None:
            return

        selection = domain_slices(self, domain)
        if selection is None:
            # No intersection
            return

        slices_with_channel, _ = selection

        self._erase(slices_with_channel)

    def _erase(self, slices_with_channel: Tuple[slice, ...]) -> None:
        """Set the data to zero in the given slices (on a copy of the data)."""
        new_data = self.data.copy()
        new_data[slices_with_channel] = 0
        self.data = new_data
