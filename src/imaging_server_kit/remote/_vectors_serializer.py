from typing import List, Optional
from geojson import Feature, LineString

import numpy as np

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.types._vectors import Vectors


class VectorsDataSerializer(Serializer):
    @staticmethod
    def serialize(vectors: Optional[Vectors], client_origin: str) -> Optional[List[Feature]]:
        if vectors is None:
            return None
        
        if vectors.data is None:
            return
        
        serialized_vectors = []
        vectors_data = vectors.data[:, :, ::-1]  # Invert XY
        for i, vector in enumerate(vectors_data):
            point_start = list(vector[0])
            point_end = list(vector[0] + vector[1])
            coords = [point_start, point_end]
            try:
                geom = LineString(coordinates=coords)
                serialized_vectors.append(
                    Feature(geometry=geom, properties={"Detection ID": i})
                )
            except ValueError:
                print("Invalid line string geometry.")
        
        return serialized_vectors

    @staticmethod
    def deserialize(serialized_vectors: Optional[List[Feature]], client_origin: str) -> Optional[np.ndarray]:
        if serialized_vectors is None:
            return None

        vectors_arr = np.array(
            [feature["geometry"]["coordinates"] for feature in serialized_vectors]
        )
        
        vector_coords = vectors_arr[:, 0]
        displacements = vectors_arr[:, 1] - vector_coords
        vectors = np.stack((vector_coords, displacements))
        vectors = np.rollaxis(vectors, 1)
        vectors = vectors[:, :, ::-1]  # Invert XY
        
        return vectors