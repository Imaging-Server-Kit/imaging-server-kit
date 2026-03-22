from typing import List, Optional

from geojson import Feature, Polygon
import numpy as np
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.types._boxes import Boxes


class BoxesDataSerializer(Serializer):
    @staticmethod
    def serialize(boxes: Optional[Boxes], client_origin: str) -> Optional[List[Feature]]:
        if boxes is None:
            return
        
        if boxes.data is None:
            return
        
        features = []
        for i, box in enumerate(boxes.data):
            coords = np.array(box)[:, ::-1]  # Invert XY
            coords = coords.tolist()
            coords.append(coords[0])  # Close the Polygon
            try:
                geom = Polygon(coordinates=[coords], validate=True)
                features.append(Feature(geometry=geom, properties={"Detection ID": i}))
            except ValueError:
                print(
                    "Invalid box polygon geometry. Expected an array of shape (N, 4, D) representing the corners of the box."
                )
        
        return features

    @staticmethod
    def deserialize(serialized_boxes: Optional[List[Feature]], client_origin: str) -> Optional[np.ndarray]:
        if serialized_boxes is None:
            return None
        boxes = np.array(
            [feature["geometry"]["coordinates"] for feature in serialized_boxes]
        )
        boxes = np.array(
            [box[0] for box in boxes]
        )  # We get back a shape of (N, 1, 5, 2) - so we remove dim. 1
        if len(boxes) > 0:
            boxes = boxes[:, :-1]  # Remove the last element
            boxes = boxes[:, :, ::-1]  # Invert XY
        return boxes
