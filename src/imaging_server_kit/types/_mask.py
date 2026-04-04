from __future__ import annotations

from typing import List, Optional, Tuple

import imantics
import numpy as np
from geojson import Feature, Polygon
from skimage.draw import polygon2mask

from imaging_server_kit.core.tiling import Domain
from imaging_server_kit.types.layer import Layer
from imaging_server_kit.types.common import safe_index_slice


def mask2features(segmentation_mask: np.ndarray) -> List[Feature]:
    """
    Args:
        segm_mask: Segmentation mask with the background pixels set to zero and the pixels assigned to a segmented
        class set to an int value

    Returns:
        A list containing the contours of each object as a geojson.Feature
    """
    features = []
    indices = np.unique(segmentation_mask)
    indices = indices[indices != 0]  # remove background

    if indices.size == 0:
        return features

    for pixel_class in indices:
        mask = segmentation_mask == int(pixel_class)
        polygons = imantics.Mask(mask).polygons()
        for detection_id, contour in enumerate(polygons.points, start=1):
            coords = np.array(contour)
            coords = np.vstack([coords, coords[0]])  # Close the polygon for QuPath
            try:
                geom = Polygon(coordinates=[coords.tolist()], validate=True)
                feature = Feature(
                    geometry=geom,
                    properties={
                        "Detection ID": detection_id,
                        "Class": int(
                            pixel_class
                        ),  # the int() casting solves a bug with json serialization
                    },
                )
                features.append(feature)
            except ValueError:
                print("Invalid polygon geometry.")

    return features


def features2mask(features, image_shape):
    segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
    for feature in features:
        feature_coordinates = np.array(feature["geometry"]["coordinates"])
        feature_coordinates = feature_coordinates[0, :, :]  # Remove an extra dimension
        feature_coordinates = feature_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask(image_shape, feature_coordinates)
        feature_properites = feature.get("properties")
        feature_class = feature_properites.get("Class")
        segmentation_mask[feature_mask] = feature_class
    return segmentation_mask


def instance_mask2features(segmentation_mask: np.ndarray) -> List[Feature]:
    """
    Args:
        segm_mask: Segmentation mask with the background pixels set to zero and the pixels assigned to a segmented
         object instance set to an int value

    Returns:
        A list containing the contours of each object as a geojson.Feature
    """
    features = []
    indices = np.unique(segmentation_mask)
    indices = indices[indices != 0]  # remove background

    if indices.size == 0:
        return features

    for detection_id in indices:
        mask = segmentation_mask == int(detection_id)
        polygons = imantics.Mask(mask).polygons()
        for contour in polygons.points:
            coords = np.array(contour)
            coords = np.vstack([coords, coords[0]])  # Close the polygon for QuPath
            try:
                geom = Polygon(coordinates=[coords.tolist()], validate=True)
                feature = Feature(
                    geometry=geom,
                    properties={"Detection ID": int(detection_id), "Class": 1},
                )
                features.append(feature)
            except ValueError:
                print("Invalid polygon geometry.")

    return features


def features2instance_mask(features, image_shape):
    segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
    for feature in features:
        feature_coordinates = np.array(feature["geometry"]["coordinates"])
        feature_coordinates = feature_coordinates[0, :, :]  # Remove an extra dimension
        feature_coordinates = feature_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask(image_shape, feature_coordinates)
        feature_properites = feature.get("properties")
        feature_id = feature_properites.get("Detection ID")
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


def features2mask_3d(features, image_shape):
    segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
    _, ry, rx = image_shape
    for feature in features:
        feature_xy_coordinates = np.array(feature["geometry"]["coordinates"])
        feature_coordinates = feature_coordinates[0, :, :]  # Remove an extra dimension
        feature_xy_coordinates = feature_xy_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask((ry, rx), feature_xy_coordinates)
        feature_z_idx = feature["properties"]["z_idx"]
        feature_properites = feature.get("properties")
        feature_id = feature_properites.get("Class")
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


def features2instance_mask_3d(features, image_shape):
    segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
    _, ry, rx = image_shape
    for feature in features:
        feature_xy_coordinates = np.array(feature["geometry"]["coordinates"])
        # Remove an extra dimension
        feature_xy_coordinates = feature_xy_coordinates[0, :, :]
        feature_xy_coordinates = feature_xy_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask((ry, rx), feature_xy_coordinates)
        feature_z_idx = feature["properties"]["z_idx"]
        feature_properites = feature.get("properties")
        feature_id = feature_properites.get("Detection ID")
        segmentation_mask[feature_z_idx][feature_mask] = feature_id
    return segmentation_mask


class Mask(Layer):
    """Data layer used to represent segmentation masks.

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
    def channel_axis(self):
        if self.meta["channel_axis"] is not None:
            return self.meta["channel_axis"]

    @property
    def bounds(self) -> Optional[Tuple]:
        """Data bounds in local coordinates."""
        if self._data is None:
            return

        if self.meta is None:
            return

        if self.channel_axis is not None:
            shape = list(self._data.shape)
            shape.pop(self.channel_axis)
            return tuple(shape)
        else:
            return self._data.shape

    def select(self, domain: Domain) -> Mask:
        """Select data in a given domain."""
        if (
            (self.data is None)
            or (domain.coords_max is None)
            or (self.coords_max is None)
        ):
            _data = None
        elif (domain.coords_max > np.asarray(self.coords_max)).any():
            _data = None
        else:
            domain_local = domain.copy()
            domain_local.coords_min = tuple(
                np.array(domain_local.coords_min) - np.array(self.coords_min)
            )
            slices_int = tuple(safe_index_slice(s) for s in domain_local.slices)

            # Account for the channel_axis
            if self.channel_axis is not None:
                slices_int_with_channel = (
                    slices_int[: self.channel_axis]
                    + (slice(None),)
                    + slices_int[self.channel_axis :]
                )
            else:
                slices_int_with_channel = slices_int

            try:
                _data = self.data[slices_int_with_channel]
            except:
                raise RuntimeError(
                    "Data re-initialization in the provided domain failed. Did you pass a domain range outside of the object's domain?"
                )

        return Mask(
            data=_data,
            name=self.name,
            meta=self.meta,
            tile_meta=self.tile_meta,
            domain=domain,
        )

    def zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            return np.zeros(domain.size, dtype=np.uint16)

    def reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        if not isinstance(domain, Domain):
            return

        domain_local = domain.copy()
        domain_local.coords_min = tuple(
            np.array(domain_local.coords_min) - np.array(self.coords_min)
        )
        slices_int = tuple(safe_index_slice(s) for s in domain_local.slices)

        # Account for the channel_axis
        if self.channel_axis is not None:
            slices_int_with_channel = (
                slices_int[: self.channel_axis]
                + (slice(None),)
                + slices_int[self.channel_axis :]
            )
        else:
            slices_int_with_channel = slices_int

        try:
            self.data[slices_int_with_channel] = 0
        except:
            raise RuntimeError(
                "Data re-initialization in the provided domain failed. Did you pass a domain range outside of the object's domain?"
            )
