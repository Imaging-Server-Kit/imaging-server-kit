from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from geojson import Feature, Point

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.common import extract_meta_tile, merge_meta_tile
from imaging_server_kit.types.data_layer import DataLayer


def _preprocess_tile_info(points: np.ndarray, points_meta: Dict, tile_info: Dict):
    tile_info = tile_info["tile_params"]
    ndim = tile_info["ndim"]
    if ndim != points.shape[1]:
        raise ValueError(f"ndim from tile info ({ndim}) is incompatible with data shape ({points.shape})")

    tile_sizes = [tile_info[f"tile_size_{idx}"] for idx in range(ndim)]
    tile_positions = [tile_info[f"pos_{idx}"] for idx in range(ndim)]
    tile_max_positions = [
        tile_pos + tile_size
        for (tile_pos, tile_size) in zip(tile_positions, tile_sizes)
    ]

    n_objects = len(points)

    if n_objects:
        # Coordinates of the points
        points_coords = points[:, :ndim]  # shape (N, ndim)

        # Mask of point coordinates in the tile
        points_in_tile = (points_coords >= tile_positions) & (
            points_coords < tile_max_positions
        )

        # All coordinates must be in the tile bounds
        tile_filter = points_in_tile.all(axis=1)  # (N,)

        # Select points in the tile
        points_tile = points[tile_filter]

        # Select meta of points in the tile
        points_meta_tile = extract_meta_tile(points_meta, n_objects, tile_filter)
    else:
        points_tile = points
        points_meta_tile = points_meta
        tile_filter = np.full_like(points, fill_value=True, dtype=np.bool_)

    return ndim, points_tile, points_meta_tile, tile_filter, tile_positions


def decode_point_features(features: List[Feature]) -> np.ndarray:
    if len(features):
        points = np.array([feature["geometry"]["coordinates"] for feature in features])
        points = points[:, 0, :]  # Remove an extra dimension
        points = points[:, ::-1]  # Invert XY
        return points.astype(float)
    else:
        return np.asarray(features)


def encode_point_features(points: np.ndarray) -> List[Feature]:
    point_features = []
    point_coords = np.asarray(points)[:, ::-1]  # Invert XY
    for detection_id, point in enumerate(point_coords):
        try:
            geom = Point(coordinates=[np.asarray(point).tolist()])
            point_features.append(Feature(geometry=geom, properties={"Detection ID": detection_id}))
        except:
            print("Invalid point geometry.")
    return point_features


class Points(DataLayer):
    """Data layer used to represent points.

    Parameters
    ----------
    data: A Numpy array of shape (N, D) representing point coordinates, where D is the dimensionality (2, 3..).
    """

    kind = "points"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Points",
        description="Input points (2D, 3D)",
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
        self.required = False

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

        # TODO: Implement object-specific properties, like max_objects or min_point_distance (could be validated).

    def pixel_domain(self):
        if self.data is None:
            return
        return np.max(self.data, axis=0)

    def get_tile(self, tile_info: Dict) -> Tuple[np.ndarray, Dict]:
        ndim, points_tile, points_meta_tile, tile_filter, tile_positions = (
            _preprocess_tile_info(self.data, self.meta, tile_info)
        )

        # Offset the points in the tile by the tile position
        points_tile[:, :ndim] = points_tile[:, :ndim] - tile_positions

        return (points_tile, points_meta_tile)

    def merge_tile(self, points_tile: np.ndarray, tile_info: Dict) -> None:
        points_tile_meta = tile_info
        ndim, old_points_tile, old_points_meta_tile, tile_filter, tile_positions = (
            _preprocess_tile_info(self.data, points_tile_meta, tile_info)
        )

        # Offset the tile data by the tile positions
        points_tile[:, :ndim] = points_tile[:, :ndim] + tile_positions

        n_objects = len(self.data)

        if n_objects:
            # Remove the points from the points data that are in the tile
            points_clean = self.data[~tile_filter]

            # Merge the tile data with the cleaned points data
            merged_points = np.vstack((points_clean, points_tile))

            # Do the same for the meta
            merged_points_meta = merge_meta_tile(
                self.meta, points_tile_meta, n_objects, tile_filter
            )
        else:
            merged_points = points_tile
            merged_points_meta = points_tile_meta

        self.data = merged_points
        self.meta = merged_points_meta

    @classmethod
    def serialize(cls, points: Optional[np.ndarray], client_origin: str) -> Optional[Union[str, List[Feature]]]:
        if points is None:
            return None
        if client_origin == "Python/Napari":
            point_features = encode_contents(points.astype(np.float32))
        elif client_origin == "Java/QuPath":
            point_features = encode_point_features(points)
        else:
            raise ValueError(f"Unrecognized client origin: {client_origin}")
        return point_features

    @classmethod
    def deserialize(cls, serialized_points: Optional[Union[np.ndarray, str]], client_origin: str) -> Optional[np.ndarray]:
        if serialized_points is None:
            return None
        if isinstance(serialized_points, str):
            if client_origin == "Python/Napari":
                points = decode_contents(serialized_points).astype(float)
            else:
                raise ValueError(f"Unrecognized client origin: {client_origin}")
        else:
            points = serialized_points
        return points

    @classmethod
    def _get_initial_data(cls, pixel_domain: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros((0, len(pixel_domain)), dtype=np.float32)

    @classmethod
    def validate_data(cls, data, meta, constraints):
        main, extra = constraints
        if extra["required"] is False:
            return

        assert isinstance(
            data, np.ndarray
        ), f"Points data ({type(data)}) is not a Numpy array"
        assert len(data.shape) == 2, "Points data should have shape (N, D)"
