from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from geojson import Feature, Point

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.common import extract_meta_tile
from imaging_server_kit.types.data_layer import DataLayer, ObjectMerger, DataSerializer


def _get_tile(points: Points, tile_meta: TileMeta):
    # Mask of point coordinates in the tile
    points_in_tile = (points.data >= tile_meta.coords_min) & (
        points.data < tile_meta.coords_max
    )

    # All coordinates must be in the tile bounds
    tile_filter = points_in_tile.all(axis=1)  # (N,)

    # Select points in the tile
    points_data_tile = points.data[tile_filter]

    # Select meta of points in the tile
    points_meta_tile = extract_meta_tile(points.meta, points.n_objects, tile_filter)

    return points_data_tile, points_meta_tile, tile_filter


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
            point_features.append(
                Feature(geometry=geom, properties={"Detection ID": detection_id})
            )
        except:
            print("Invalid point geometry.")
    return point_features


class PointsDataSerializer(DataSerializer):
    def serialize(self, points: Optional[np.ndarray], client_origin: str) -> Optional[Union[str, List[Feature]]]:
        if points is None:
            return None
        if client_origin == "Python/Napari":
            point_features = encode_contents(points.astype(np.float32))
        elif client_origin == "Java/QuPath":
            point_features = encode_point_features(points)
        else:
            raise ValueError(f"Unrecognized client origin: {client_origin}")
        return point_features
    
    def deserialize(self, serialized_points: Optional[Union[str, List[Feature]]], client_origin: str) -> Optional[np.ndarray]:
        if serialized_points is None:
            return None
        if isinstance(serialized_points, str):
            points = decode_contents(serialized_points).astype(float)
        else:
            # Points are a List[Features]
            points = decode_point_features(serialized_points)
        return points


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
        tile_meta: Optional[TileMeta] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
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

        self.merger = ObjectMerger()
        
        self.data_serializer = PointsDataSerializer()

    @property
    def data_global_coords(self) -> Optional[np.ndarray]:
        """Data in global coordinate reference instead of local to the tile."""
        if (self.data is not None) and (self.tile_meta is not None):
            if self.tile_meta.coords_min is not None:
                return self.data + self.tile_meta.coords_min

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        if self.data is not None:
            if self.n_objects > 0:
                return tuple(np.max(self.data, axis=0))

    def get_tile(self, tile_meta: TileMeta) -> Points:
        if self.data is None:
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self._get_initial_data(self.pixel_domain)
            _meta = self.meta
        else:
            points_tile_data, points_tile_meta, _ = _get_tile(self, tile_meta)
            if points_tile_data is not None:
                points_tile_data = points_tile_data - tile_meta.coords_min
            _data = points_tile_data
            _meta = points_tile_meta
        return Points(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=tile_meta,
        )

    @classmethod
    def _get_initial_data(
        cls, pixel_domain: Optional[Union[Tuple, List]]
    ) -> Optional[np.ndarray]:
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
