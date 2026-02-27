from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union

import imantics
import numpy as np
from geojson import Feature, Polygon
from skimage.draw import polygon2mask
import networkx as nx
from skimage.util import map_array

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer, Merger, DefaultMerger


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


def unique_positive(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels[labels > 0])


class MaskOverrideMerger(DefaultMerger):
    """Merge two masks using an `override` strategy: incoming data overrides existnig data."""

    def merge(self, src_layer: Mask, dst_layer: Mask) -> None:
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
            src_layer.data = src_layer.initialize([1] * dst_layer.ndim)
            src_layer.meta = dst_layer.meta

        if (
            (src_layer.pixel_domain is None)
            or (src_layer.tile_meta is None)
            or (src_layer.data is None)
        ):
            return  # This should never happen (just there for type hints)

        # Simple "Override" strategy; could be improved with pixel-wise majority voting between overlapping tiles
        _slices = dst_layer.tile_meta.slices
        _stack = np.stack([src_layer.pixel_domain, dst_layer.pixel_domain])
        _pixel_domain = np.max(_stack, axis=0).tolist()

        if _pixel_domain != src_layer.pixel_domain:
            new_data = dst_layer.initialize(_pixel_domain)
            new_data[src_layer.tile_meta.slices] = src_layer.data
        else:
            new_data = src_layer.data

        # `Override` strategy:
        new_data[_slices] = dst_layer.data
        src_layer.data = new_data
        src_layer.meta = dst_layer.meta

    def first_tile_hook(self, src_layer: Mask, dst_layer: Mask):
        src_layer.meta = dst_layer.meta


class MaskTileOverrideMerger(MaskOverrideMerger):
    """Merge two masks using an `override` strategy: incoming data overrides existnig data."""

    def first_tile_hook(self, src_layer: Mask, dst_layer: Mask):
        src_layer.data = Mask._get_initial_data(src_layer.pixel_domain)
        src_layer.meta = dst_layer.meta


class InstanceTileTracker:
    def __init__(self) -> None:
        self.initialize()

    def initialize(self):
        self.N = 0  # Current number of objects
        self.G = nx.Graph()

    def add_N_to_tile(self, labels: np.ndarray) -> np.ndarray:
        if labels.sum() > 0:
            labels[labels != 0] = labels[labels != 0] + self.N
            self.N = labels.max()
        return labels

    def add_node(self, lab):
        self.G.add_node(lab)

    def add_edge(self, a, b):
        self.G.add_edge(a, b)

    def build_mapping(self):
        self.G.add_nodes_from(range(1, self.N + 1))
        mapping = {}
        for comp_id, comp in enumerate(nx.connected_components(self.G), start=1):
            for n in comp:
                mapping[int(n)] = comp_id
        self._mapping = mapping

    def resolve(self, arr):
        if not hasattr(self, "_mapping"):
            self.build_mapping()
        input_vals = np.array(list(self._mapping.keys()), dtype=np.int64)
        output_vals = np.array(list(self._mapping.values()), dtype=np.int64)
        return map_array(arr, out=arr, input_vals=input_vals, output_vals=output_vals)


class InstanceMaskTileMerger(DefaultMerger):
    def __init__(self, min_intersecting_px: int = 1) -> None:
        self.min_intersecting_px = min_intersecting_px
        self.tile_tracker = InstanceTileTracker()

    def merge(self, src_layer: Mask, dst_layer: Mask) -> None:
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
            src_layer.data = dst_layer.initialize(tuple([1] * dst_layer.ndim))
            src_layer.meta = dst_layer.meta

        if (
            (src_layer.pixel_domain is None)
            or (src_layer.tile_meta is None)
            or (src_layer.data is None)
        ):
            return  # This should never happen (just there for type hints)

        _slices = dst_layer.tile_meta.slices
        _stack = np.stack([src_layer.pixel_domain, dst_layer.pixel_domain])
        _pixel_domain = np.max(_stack, axis=0).tolist()

        if _pixel_domain != src_layer.pixel_domain:
            new_data = dst_layer.initialize(_pixel_domain)
            new_data[src_layer.tile_meta.slices] = src_layer.data
        else:
            new_data = src_layer.data

        src_layer.data = new_data  # Extend the source layer data

        src_tile = src_layer.get_tile(dst_layer.tile_meta)
        if src_tile.data is None:
            raise ValueError(f"Could not get a mask tile where it was requested.")

        dst_arr = self.tile_tracker.add_N_to_tile(dst_layer.data)

        for new_label in unique_positive(dst_arr):
            self.tile_tracker.add_node(new_label)

        border_mask = dst_layer.tile_meta.overlap_border_mask
        if border_mask is not None:
            for dst_lab in unique_positive(dst_arr[border_mask]):
                filt = np.logical_and(border_mask, dst_arr == dst_lab)
                src_tile_filt = src_tile.data[filt]
                for src_lab in unique_positive(src_tile_filt):
                    n_intersecting_px = (src_tile_filt == src_lab).sum()
                    if n_intersecting_px > self.min_intersecting_px:
                        self.tile_tracker.add_edge(src_lab, dst_lab)

        new_data[_slices] = dst_arr
        src_layer.data = new_data
        src_layer.meta = dst_layer.meta

    def first_tile_hook(self, src_layer: Mask, dst_layer: Mask):
        # Important: just calling .initialize() would not work
        src_layer.data = dst_layer.initialize([1] * dst_layer.ndim)
        self.tile_tracker = InstanceTileTracker()  

    def last_tile_hook(self, src_layer: Mask, dst_layer: Mask):
        src_layer.data = self.tile_tracker.resolve(src_layer.data)
        self.tile_tracker = InstanceTileTracker()


class Mask(DataLayer):
    """Data layer used to represent segmentation masks.

    Parameters
    ----------
    data: Numpy arrays, integer type. Integers can represent object classes (e.g. pixel classification) or object instances.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    """

    kind = "mask"
    mergers: Dict[str, Type[Merger]] = {
        "default": MaskTileOverrideMerger,
        "instances": InstanceMaskTileMerger,
        "override": MaskOverrideMerger,
    }

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name: str = "Mask",
        description: str = "Segmentation mask (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        merger: str = "default",
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
            merger=merger,
            **kwargs,
        )

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        if isinstance(self.data, np.ndarray):
            return self.data.shape

    def get_tile(self, tile_meta: TileMeta) -> Mask:
        if (
            (self.data is None)
            or (tile_meta.coords_max is None)
            or (self.pixel_domain is None)
        ):
            _data = None
        elif (tile_meta.coords_max > np.asarray(self.pixel_domain)).any():
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
    def _get_initial_data(
        pixel_domain: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if pixel_domain is not None:
            return np.zeros(np.array(pixel_domain).astype(np.uint16), dtype=np.uint16)
    
    def initialize(self, domain: List[int]) -> Optional[np.ndarray]:
        return self._get_initial_data(domain)

    @staticmethod
    def validate_data(data, meta):
        assert isinstance(
            data, np.ndarray
        ), f"Mask data ({type(data)}) is not a Numpy array"

        if not all(data.shape):
            raise ValueError("Image array has an invalid shape: ", data.shape)

        if len(data.shape) not in meta["dimensionality"]:
            raise ValueError("Image array has the wrong dimensionality.")
