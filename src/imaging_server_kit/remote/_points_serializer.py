from typing import List, Optional, Union

from geojson import Feature, Point
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import decode_contents, encode_contents
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
            print("⚠️ Invalid point geometry.")
    return point_features


class PointsDataSerializer(Serializer):
    @staticmethod
    def serialize(points: Optional[Points]) -> Optional[Union[str, List[Feature]]]:
        if points is None:
            return

        if points.data is None:
            return

        return encode_contents(points.data.astype(np.float32))

    @staticmethod
    def deserialize(serialized_points: Optional[str]) -> Optional[np.ndarray]:
        if isinstance(serialized_points, str):
            return decode_contents(serialized_points).astype(float)
