from typing import Dict, List, Optional, Tuple, Union

import imantics
import numpy as np
from geojson import Feature, Polygon
from skimage.draw import polygon2mask

from imaging_server_kit.core.encoding import decode_contents, encode_contents
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
        feature_xy_coordinates = feature_xy_coordinates[0, :, :]  # Remove an extra dimension
        feature_xy_coordinates = feature_xy_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask((ry, rx), feature_xy_coordinates)
        feature_z_idx = feature["properties"]["z_idx"]
        feature_properites = feature.get("properties")
        feature_id = feature_properites.get("Detection ID")
        segmentation_mask[feature_z_idx][feature_mask] = feature_id
    return segmentation_mask


def _get_slices(mask: np.ndarray, tile_info: Dict):
    tile_info = tile_info["tile_params"]
    ndim = tile_info["ndim"]
    if ndim != mask.ndim:
        raise ValueError(f"ndim from tile info ({ndim}) is incompatible with data shape ({mask.shape})")

    tile_sizes = [tile_info[f"tile_size_{idx}"] for idx in range(ndim)]
    tile_positions = [tile_info[f"pos_{idx}"] for idx in range(ndim)]
    tile_max_positions = [
        tile_pos + tile_size
        for (tile_pos, tile_size) in zip(tile_positions, tile_sizes)
    ]
    slices = [
        slice(pos, max_pos) for pos, max_pos in zip(tile_positions, tile_max_positions)
    ]
    return tuple(slices)


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
        required: bool = True,
        meta: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
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
            "required": self.required,
        }
        self.constraints = [main, extra]

        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

    def pixel_domain(self):
        if self.data is None:
            return
        return self.data.shape

    def get_tile(self, tile_info: Dict) -> Tuple[Optional[np.ndarray], Dict]:
        if self.data is not None:
            tile_silces = _get_slices(self.data, tile_info)
            tile_data = self.data[tile_silces]
        else:
            tile_data = None
        return tile_data, self.meta

    def merge_tile(self, mask_tile: np.ndarray, tile_info: Dict) -> None:
        if self.data is not None:
            tile_slices = _get_slices(self.data, tile_info)
            self.data[tile_slices] = mask_tile

    @classmethod
    def serialize(cls, mask: Optional[np.ndarray], client_origin: str) -> Optional[Union[List[Feature], str]]:
        if mask is None:
            return None
        if client_origin == "Python/Napari":
            features = encode_contents(mask.astype(np.uint16))
        elif client_origin == "Java/QuPath":
            features = mask2features(mask)
        else:
            raise ValueError(f"Unrecognized client origin: {client_origin}")
        return features

    @classmethod
    def deserialize(cls, serialized_data: Optional[Union[np.ndarray, str]], client_origin: str) -> Optional[np.ndarray]:
        if serialized_data is None:
            return None
        if isinstance(serialized_data, str):
            if client_origin == "Python/Napari":
                data = decode_contents(serialized_data).astype(int)
            else:
                raise ValueError(f"Unrecognized client origin: {client_origin}")
        return data

    @classmethod
    def _get_initial_data(cls, pixel_domain: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros(pixel_domain, dtype=np.uint16)

    @classmethod
    def validate_data(cls, data, meta, constraints):
        main, extra = constraints
        if extra["required"] is False:
            return

        assert isinstance(
            data, np.ndarray
        ), f"Mask data ({type(data)}) is not a Numpy array"

        if not all(data.shape):
            raise ValueError("Image array has an invalid shape: ", data.shape)

        if len(data.shape) not in extra["dimensionality"]:
            raise ValueError("Image array has the wrong dimensionality.")
