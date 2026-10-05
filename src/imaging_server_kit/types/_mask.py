from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import geojson
import numpy as np
import pandas as pd
import rasterio.features
import shapely
from geojson import Feature
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core.domain import Domain
from imaging_server_kit.types.common import copy_meta, domain_slices

# Largest mask (in pixels) for which the number of labels is shown in `repr`
MAX_SIZE_COUNT_LABELS = 2**24

# Label dtypes accepted by `rasterio.features.shapes`
_RASTERIO_INT_DTYPES = (np.uint8, np.int16, np.uint16, np.int32)


def _offset_transform(offset: Sequence[float]) -> Affine:
    """Affine transform from (row, col) pixel indices to (x, y) coordinates, shifted by `offset` (row, col)."""
    return Affine.translation(float(offset[1]), float(offset[0]))


def _to_geojson(geom: BaseGeometry):
    return geojson.loads(shapely.to_geojson(geom))


def _label_shapes(
    segmentation_mask: np.ndarray, offset: Sequence[float]
) -> Iterator[Tuple[BaseGeometry, int]]:
    """Yield a (polygon, label) pair for each 4-connected component of each non-zero label.

    Polygons follow pixel edges (pixel (r, c) covers [c, c+1] x [r, r+1]) and include holes as interior rings.
    """
    if segmentation_mask.ndim != 2:
        raise ValueError(
            f"Only 2D masks can be converted to features (got {segmentation_mask.ndim}D)."
        )

    if segmentation_mask.dtype == bool:
        segmentation_mask = segmentation_mask.astype(np.uint8)
    elif segmentation_mask.dtype not in _RASTERIO_INT_DTYPES:
        if segmentation_mask.size and (
            segmentation_mask.min() < np.iinfo(np.int32).min
            or segmentation_mask.max() > np.iinfo(np.int32).max
        ):
            raise ValueError("Mask values do not fit in int32.")
        segmentation_mask = segmentation_mask.astype(np.int32)

    for geometry, label in rasterio.features.shapes(
        segmentation_mask,
        mask=segmentation_mask > 0,
        connectivity=4,
        transform=_offset_transform(offset),
    ):
        yield shape(geometry), int(label)


def _rasterize(
    features: List[Feature],
    image_shape: Tuple,
    value_key: str,
    offset: Sequence[float],
) -> np.ndarray:
    """Burn the `value_key` property of each feature into a mask (pixels whose center falls inside the geometry)."""
    shapes = [(f["geometry"], int(f["properties"][value_key])) for f in features]
    if len(shapes) == 0:
        return np.zeros(image_shape, dtype=np.int32)

    return rasterio.features.rasterize(
        shapes,
        out_shape=image_shape,
        transform=_offset_transform(offset),
        fill=0,
        dtype=np.int32,
    )


def mask2features(
    segmentation_mask: np.ndarray, offset: Sequence[float] = (0, 0)
) -> List[Feature]:
    """Convert a semantic segmentation mask to GeoJSON features.

    One feature is created per 4-connected component of each class. Holes are kept as interior rings.

    Parameters
    ----------
    segmentation_mask: 2D mask with the background set to zero and pixels assigned to a class set to an int value.
    offset: (row, col) offset added to the feature coordinates.

    Returns
    -------
    A list of Polygon features with properties `Detection ID` (component index, starting at 1) and `Class` (pixel value).
    """
    features = []
    for detection_id, (geom, label) in enumerate(
        _label_shapes(segmentation_mask, offset), start=1
    ):
        features.append(
            Feature(
                geometry=_to_geojson(geom),
                properties={"Detection ID": detection_id, "Class": label},
            )
        )
    return features


def features2mask(
    features: List[Feature], image_shape: Tuple, offset: Sequence[float] = (0, 0)
) -> np.ndarray:
    """Convert GeoJSON features to a semantic segmentation mask (inverse of `mask2features`).

    Parameters
    ----------
    features: Polygon or MultiPolygon features with a `Class` property.
    image_shape: Shape of the output mask.
    offset: (row, col) offset of the feature coordinates, as passed to `mask2features`.

    Returns
    -------
    An int32 mask where pixels inside each feature are set to its `Class`.
    """
    return _rasterize(features, image_shape, "Class", offset)


def instance_mask2features(
    segmentation_mask: np.ndarray, offset: Sequence[float] = (0, 0)
) -> List[Feature]:
    """Convert an instance segmentation mask to GeoJSON features.

    One feature is created per label. Holes are kept as interior rings, and disconnected parts
    of the same label are grouped in a MultiPolygon.

    Parameters
    ----------
    segmentation_mask: 2D mask with the background set to zero and pixels assigned to an object instance set to an int value.
    offset: (row, col) offset added to the feature coordinates.

    Returns
    -------
    A list of (Multi)Polygon features with properties `Detection ID` (the label) and `Class` (always 1), sorted by label.
    """
    parts: Dict[int, List[BaseGeometry]] = defaultdict(list)
    for geom, label in _label_shapes(segmentation_mask, offset):
        parts[label].append(geom)

    features = []
    for label in sorted(parts):
        geoms = parts[label]
        geom = geoms[0] if len(geoms) == 1 else shapely.union_all(geoms)
        features.append(
            Feature(
                geometry=_to_geojson(geom),
                properties={"Detection ID": label, "Class": 1},
            )
        )
    return features


def features2instance_mask(
    features: List[Feature], image_shape: Tuple, offset: Sequence[float] = (0, 0)
) -> np.ndarray:
    """Convert GeoJSON features to an instance segmentation mask (inverse of `instance_mask2features`).

    Parameters
    ----------
    features: Polygon or MultiPolygon features with a `Detection ID` property.
    image_shape: Shape of the output mask.
    offset: (row, col) offset of the feature coordinates, as passed to `instance_mask2features`.

    Returns
    -------
    An int32 mask where pixels inside each feature are set to its `Detection ID`.
    """
    return _rasterize(features, image_shape, "Detection ID", offset)


class Mask(Layer):
    """Data layer for segmentation masks.

    Masks are label images, where integer values encode either object classes (e.g.
    for pixel classification) or object instances.

    Parameters
    ----------
    data : numpy.ndarray, optional
        The label image. Boolean arrays are accepted for binary masks.
    name : str, default="Mask"
        Name of the layer.
    description : str, default="Segmentation mask"
        Description of the layer, displayed on the algorithm documentation page.
    dimensionality : list of int, optional
        Accepted numbers of dimensions, for example `[2, 3]`. By default, any number
        of dimensions is accepted.
    channel_axis : int, optional
        Index of the channel axis, if any. The channel axis is not counted as a
        spatial dimension, and it is never split into tiles.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties. Use `merger="instances"` for
        instance segmentation masks, so that labels stay unique when the mask is
        assembled from tiles.

    Examples
    --------
    >>> mask = sk.Mask(labels, name="Nuclei", merger="instances")
    """

    kind = "mask"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name: str = "Mask",
        description: str = "Segmentation mask",
        dimensionality: Optional[List[int]] = None,
        channel_axis: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            data=data,
            dimensionality=dimensionality,
            channel_axis=channel_axis,
            **kwargs,
        )

    def _summary(self) -> str:
        data = self.data
        if data is None:
            return "empty"
        summary = f"{data.dtype} {data.shape}"
        # Counting labels requires a full pass over the data, so we skip it for large masks
        if data.size <= MAX_SIZE_COUNT_LABELS:
            summary += f", {self.n_objects} labels"
        return summary

    @property
    def channel_axis(self) -> Optional[int]:
        if self.meta:
            if self.meta["channel_axis"] is not None:
                return self.meta["channel_axis"]
    
    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            # pd.unique (hash-based) is several times faster than np.unique (sort-based)
            labels = pd.unique(self.data.ravel())
            n_labels = len(labels) - int(0 in labels)
            return n_labels

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

    def select(self, domain: Domain) -> Mask:
        """Select the part of the layer inside a domain.

        Parameters
        ----------
        domain : Domain
            The region to select, in global pixel coordinates.

        Returns
        -------
        Mask
            A new layer with the selected data, positioned in global coordinates.
        """
        _meta = copy_meta(self)

        if (self.data is None) or (domain.size is None):
            return Mask(
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

        mask_selection = Mask(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta.copy(),
        )

        mask_selection.position = position

        return mask_selection

    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            if domain.size is not None:
                return np.zeros(domain.size, dtype=np.uint32)

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
