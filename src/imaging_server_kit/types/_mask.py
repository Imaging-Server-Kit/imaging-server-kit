from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union

import imantics
import numpy as np
from geojson import Feature, Polygon
from skimage.draw import polygon2mask

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer


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
        feature_xy_coordinates = feature_xy_coordinates[
            0, :, :
        ]  # Remove an extra dimension
        feature_xy_coordinates = feature_xy_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask((ry, rx), feature_xy_coordinates)
        feature_z_idx = feature["properties"]["z_idx"]
        feature_properites = feature.get("properties")
        feature_id = feature_properites.get("Detection ID")
        segmentation_mask[feature_z_idx][feature_mask] = feature_id
    return segmentation_mask


class Mask(DataLayer):
    """Data layer used to represent segmentation masks.

    Parameters
    ----------
    data: Numpy arrays, integer type. Integers can represent object classes (e.g. pixel classification) or object instances.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    """

    kind = "mask"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name: str = "Mask",
        description: str = "Segmentation mask (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
            dimensionality=dimensionality,
            **kwargs,
        )

    @property
    def data_bounds(self) -> Optional[Tuple]:
        if isinstance(self.data, np.ndarray):
            return self.data.shape

    def select(self, tile_meta: TileMeta) -> Mask:
        if (
            (self.data is None)
            or (tile_meta.coords_max is None)
            or (self.bounds is None)
        ):
            _data = None
        elif (tile_meta.coords_max > np.asarray(self.bounds)).any():
            _data = None
        else:
            _data = self.data[tile_meta.slices]
        return Mask(
            data=_data,
            name=self.name,
            meta=self.meta,
            tile_meta=tile_meta,
        )

    @staticmethod
    def initialize_data(
        bounds: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if bounds is not None:
            return np.zeros(np.array(bounds).astype(np.uint16), dtype=np.uint16)

    def initialize(self, bounds: List[int]) -> Optional[np.ndarray]:
        return self.initialize_data(bounds)
