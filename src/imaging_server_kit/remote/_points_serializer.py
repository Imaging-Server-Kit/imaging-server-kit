import warnings
from typing import Any, List, Optional

from geojson import Feature, Point
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import decode_coords, encode_coords
import numpy as np

from imaging_server_kit.types._points import Points


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
        except Exception:
            warnings.warn("Invalid point geometry.", stacklevel=2)
    return point_features


class PointsDataSerializer(Serializer):
    @staticmethod
    def serialize(points: Optional[Points]) -> Any:
        if points is None:
            return

        if points.data is None:
            return

        return encode_coords(points.data)

    @staticmethod
    def deserialize(serialized_points: Any) -> Optional[np.ndarray]:
        return decode_coords(serialized_points)
