from __future__ import annotations

import math
from typing import List, Optional, Tuple

import imantics
import numpy as np
from geojson import Feature, Polygon
from skimage.draw import polygon2mask

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core.domain import Domain


from skimage.measure import regionprops


def mask2features(segmentation_mask: np.ndarray) -> List[Feature]:
    """
    Args:
        segmentation_mask: Segmentation mask with the background set to zero and the pixels assigned to a class set to an int value

    Returns:
        A list containing the contours of each object as a geojson.Feature
    """
    features = []
    
    if segmentation_mask.dtype == "bool":
        segmentation_mask = segmentation_mask.astype(int)
        
    for prop in regionprops(segmentation_mask):
        pixel_class = prop.label
        minr, minc, maxr, maxc = prop.bbox
        local_mask = prop.image

        polygons = imantics.Mask(local_mask).polygons()
        for detection_id, contour in enumerate(polygons.points, start=1):
            coords = np.array(contour)
            if coords.shape[0] < 3:
                continue

            coords = coords + np.array([minc, minr])

            coords = np.vstack([coords, coords[0]])

            try:
                geom = Polygon(coordinates=[coords.tolist()], validate=True)
            except ValueError:
                print("⚠️ Invalid polygon (ignoring it).")
                continue

            feature = Feature(
                geometry=geom,
                properties={"Detection ID": detection_id, "Class": int(pixel_class)},
            )
            features.append(feature)

    return features


def features2mask(features: List[Feature], image_shape: Tuple) -> np.ndarray:
    segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
    for feature in features:
        feature_coordinates = np.array(feature["geometry"]["coordinates"])
        feature_coordinates = feature_coordinates[0, :, :]  # Remove an extra dimension
        feature_coordinates = feature_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask(image_shape, feature_coordinates)
        feature_properites = feature["properties"]
        feature_class = feature_properites["Class"]
        segmentation_mask[feature_mask] = feature_class
    return segmentation_mask


def instance_mask2features(segmentation_mask: np.ndarray) -> List[Feature]:
    """
    Args:
        segmentation_mask: Segmentation mask with the background set to zero and the pixels assigned to an object instance set to an int value

    Returns:
        A list containing the contours of each object as a geojson.Feature
    """
    features = []

    for prop in regionprops(segmentation_mask):
        detection_id = prop.label
        minr, minc, maxr, maxc = prop.bbox
        local_mask = prop.image

        polygons = imantics.Mask(local_mask).polygons()
        for contour in polygons.points:
            coords = np.array(contour)
            if coords.shape[0] < 3:
                # Only 3 points, let's skip it.
                continue

            # Offset back into full-image coordinates
            coords = coords + np.array([minc, minr])

            coords = np.vstack([coords, coords[0]])  # Close the polygon for QuPath

            try:
                geom = Polygon(coordinates=[coords.tolist()], validate=True)
            except ValueError:
                print("⚠️ Invalid polygon (ignoring it).")
                continue

            feature = Feature(
                geometry=geom,
                properties={"Detection ID": int(detection_id), "Class": 1},
            )
            features.append(feature)

    return features


def features2instance_mask(features: List[Feature], image_shape: Tuple) -> np.ndarray:
    segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
    for feature in features:
        feature_coordinates = np.array(feature["geometry"]["coordinates"])
        feature_coordinates = feature_coordinates[0, :, :]  # Remove an extra dimension
        feature_coordinates = feature_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask(image_shape, feature_coordinates)
        feature_properites = feature["properties"]
        feature_id = feature_properites["Detection ID"]
        segmentation_mask[feature_mask] = feature_id
    return segmentation_mask


def mask2features_3d(segmentation_mask: np.ndarray) -> List[Feature]:
    features = []
    for z_idx, mask_2d in enumerate(segmentation_mask):
        features_2d = mask2features(mask_2d)
        for feature_2d in features_2d:
            feature_2d.properties["z_idx"] = z_idx
            features.append(feature_2d)
    return features


def features2mask_3d(features: List[Feature], image_shape: Tuple) -> np.ndarray:
    segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
    _, ry, rx = image_shape
    for feature in features:
        feature_xy_coordinates = np.array(feature["geometry"]["coordinates"])
        feature_xy_coordinates = feature_xy_coordinates[0, :, :]  # Remove an extra dimension
        feature_xy_coordinates = feature_xy_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask((ry, rx), feature_xy_coordinates)
        feature_z_idx = feature["properties"]["z_idx"]
        feature_properites = feature["properties"]
        feature_id = feature_properites["Class"]
        segmentation_mask[feature_z_idx][feature_mask] = feature_id
    return segmentation_mask


def instance_mask2features_3d(segmentation_mask: np.ndarray) -> List[Feature]:
    features = []
    for z_idx, mask_2d in enumerate(segmentation_mask):
        features_2d = instance_mask2features(mask_2d)
        for feature_2d in features_2d:
            feature_2d.properties["z_idx"] = z_idx
            features.append(feature_2d)
    return features


def features2instance_mask_3d(features: List[Feature], image_shape: Tuple) -> np.ndarray:
    segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
    _, ry, rx = image_shape
    for feature in features:
        feature_xy_coordinates = np.array(feature["geometry"]["coordinates"])
        # Remove an extra dimension
        feature_xy_coordinates = feature_xy_coordinates[0, :, :]
        feature_xy_coordinates = feature_xy_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask((ry, rx), feature_xy_coordinates)
        feature_z_idx = feature["properties"]["z_idx"]
        feature_properites = feature["properties"]
        feature_id = feature_properites["Detection ID"]
        segmentation_mask[feature_z_idx][feature_mask] = feature_id
    return segmentation_mask


class Mask(Layer):
    """Data layer used to represent segmentation masks: label images where integer values encode either object classes or object instances.

    Parameters
    ----------
    data: Numpy arrays, integer type. Integers can represent object classes (e.g. pixel classification) or object instances.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    channel_axis: Optional index of the channel axis.
      - The channel axis does not affect the `bounds`, `ndim`, and `domain` attributes.
      - The channel axis is set to `2` if rgb is True and there is no time axis.
      - tile_size along the channel axis defaults to the length of this axis.
    """

    kind = "mask"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name: str = "Mask",
        description: str = "Segmentation mask (2D, 3D)",
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

    def select(self, domain: Domain) -> Mask:
        """Select data in a given domain."""
        if (self.data is None) or (domain.size is None):
            return Mask(
                data=None,
                name=self.name,
                meta=self.meta.copy() if self.meta is not None else self.meta,
                tile_meta=self.tile_meta.copy(),
            )

        # Get the slice indices
        cmin = [
            max([domain_cmin, this_cmin])
            for domain_cmin, this_cmin in zip(domain.coords_min, self.extent.coords_min)
        ]

        cmax = [
            min([domain_cmax, this_cmax])
            for domain_cmax, this_cmax in zip(domain.coords_max, self.extent.coords_max)
        ]

        csize = np.asarray([c1 - c0 for c1, c0 in zip(cmax, cmin)])

        if np.any(csize <= 0):
            # No intersection
            _data = None
        else:
            cmin_rounded = [math.floor(x) for x in cmin]

            slices = []
            for cmin_i, size_i, shape_i, this_cmin in zip(
                cmin_rounded,
                csize,
                self.shape,
                self.extent.coords_min,
            ):
                # Make sure not to overflow..
                size_i = min(size_i, shape_i)
                s0 = int(cmin_i - this_cmin)
                s1 = s0 + int(size_i)
                slices.append(slice(s0, s1))
            slices = tuple(slices)

            # Account for the channel_axis
            if self.channel_axis:
                slices_with_channel = (
                    slices[: self.channel_axis]
                    + (slice(None),)
                    + slices[self.channel_axis :]
                )
            else:
                slices_with_channel = slices

            # Select the data via indexing
            _data = self.data[slices_with_channel]

        # Create a new layer
        _meta = self.meta.copy() if self.meta is not None else self.meta

        mask_selection = Mask(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta.copy(),
        )

        mask_selection.position = cmin_rounded

        return mask_selection

    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            if domain.size is not None:
                return np.zeros(domain.size, dtype=np.uint16)

    def _reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        # Get the slice indices
        cmin = [
            max([domain_cmin, this_cmin])
            for domain_cmin, this_cmin in zip(domain.coords_min, self.extent.coords_min)
        ]

        cmax = [
            min([domain_cmax, this_cmax])
            for domain_cmax, this_cmax in zip(domain.coords_max, self.extent.coords_max)
        ]

        csize = np.asarray([c1 - c0 for c1, c0 in zip(cmax, cmin)])

        if np.any(csize <= 0):
            # No intersection
            return

        cmin_rounded = [math.floor(x) for x in cmin]

        slices = []
        for cmin_i, size_i, shape_i, this_cmin in zip(
            cmin_rounded,
            csize,
            self.shape,
            self.extent.coords_min,
        ):
            # Make sure not to overflow..
            size_i = min(size_i, shape_i)
            s0 = int(cmin_i - this_cmin)
            s1 = s0 + int(size_i)
            slices.append(slice(s0, s1))
        slices = tuple(slices)

        # Account for the channel_axis
        if self.channel_axis:
            slices_with_channel = (
                slices[: self.channel_axis]
                + (slice(None),)
                + slices[self.channel_axis :]
            )
        else:
            slices_with_channel = slices

        new_data = self.data.copy()
        new_data[slices_with_channel] = 0
        self.data = new_data
