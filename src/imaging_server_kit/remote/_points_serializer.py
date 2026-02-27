from typing import List, Optional, Union

from geojson import Feature, Point
from imaging_server_kit.remote.data_serializer import Serializer
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
        except:
            print("Invalid point geometry.")
    return point_features


class PointsDataSerializer(Serializer):
    @staticmethod
    def serialize(points: Optional[Points], client_origin: str) -> Optional[Union[str, List[Feature]]]:
        if points is None:
            return
        
        if points.data is None:
            return
        
        if client_origin == "Python/Napari":
            point_features = encode_contents(points.data.astype(np.float32))
        elif client_origin == "Java/QuPath":
            point_features = encode_point_features(points.data)
        else:
            raise ValueError(f"Unrecognized client origin: {client_origin}")
        
        return point_features

    @staticmethod
    def deserialize(
        serialized_points: Optional[Union[str, List[Feature]]], client_origin: str
    ) -> Optional[np.ndarray]:
        if serialized_points is None:
            return
        
        if isinstance(serialized_points, str):
            points = decode_contents(serialized_points).astype(float)
        else:
            points = decode_point_features(serialized_points)
        
        return points