"""Conversion between segmentation masks and GeoJSON features (used by the QuPath bridge).

This module does not depend on QuBaLab, so it can be used and tested without QuPath.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Sequence, Tuple

import geojson
import numpy as np
import rasterio.features
import shapely
from geojson import Feature
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

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
