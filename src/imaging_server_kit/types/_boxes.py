from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from geojson import Feature, Polygon

from imaging_server_kit.types.common import extract_meta_tile, merge_meta_tile
from imaging_server_kit.types.data_layer import DataLayer


def _preprocess_tile_info(boxes: np.ndarray, boxes_meta: Dict, tile_info: Dict):
    tile_info = tile_info["tile_params"]
    ndim = tile_info["ndim"]

    tile_sizes = [tile_info[f"tile_size_{idx}"] for idx in range(ndim)]
    tile_positions = [tile_info[f"pos_{idx}"] for idx in range(ndim)]
    tile_max_positions = [
        tile_pos + tile_size
        for (tile_pos, tile_size) in zip(tile_positions, tile_sizes)
    ]

    n_objects = len(boxes)

    if n_objects:
        # Box coordinates
        boxes_coords = np.asarray(boxes)[:, :, :ndim]

        # Mask of box coordinates in the tile
        coords_in_tile = (boxes_coords >= tile_positions) & (
            boxes_coords < tile_max_positions
        )

        # All coordinates must be in the tile bounds
        tile_filter = coords_in_tile.reshape((len(coords_in_tile), -1)).all(axis=1)

        # Select boxes in the tile
        boxes_tile = np.asarray(boxes)[~tile_filter]

        # Select meta of boxes in the tile
        boxes_meta_tile = extract_meta_tile(boxes_meta, n_objects, tile_filter)
    else:
        boxes_tile = boxes
        boxes_meta_tile = boxes_meta
        tile_filter = np.full_like(boxes, fill_value=True, dtype=np.bool_)

    return ndim, boxes_tile, boxes_meta_tile, tile_filter, tile_positions


class Boxes(DataLayer):
    """Data layer used to represent 2D boxes (rectangular bounding boxes).

    Parameters
    ----------
    data: A Numpy array of shape (N, 4, 2). data[:, 0, :], data[:, 1, :].. represent the coordinates of the four box corners.
    """

    kind = "boxes"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Boxes",
        description="Input boxes shapes (2D, 3D)",
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

        # TODO: Implement object-specific properties, like max_objects or min_box_area (could be validated).

    def pixel_domain(self):
        if self.data is None:
            return
        return np.max(np.asarray(self.data), axis=(0, 1))

    def get_tile(self, tile_info: Dict) -> Tuple[np.ndarray, Dict]:
        ndim, boxes_tile, boxes_meta_tile, _, tile_positions = _preprocess_tile_info(
            self.data, self.meta, tile_info
        )
        # Offset the boxes by the tile position
        boxes_tile[:, :, :ndim] = boxes_tile[:, :, :ndim] - tile_positions
        return boxes_tile, boxes_meta_tile

    def merge_tile(self, boxes_tile: np.ndarray, tile_info: Dict):
        boxes_tile_meta = tile_info
        ndim, old_boxes_tile, old_boxes_meta_tile, tile_filter, tile_positions = (
            _preprocess_tile_info(self.data, boxes_tile_meta, tile_info)
        )

        # Offset the tile data by the tile positions
        boxes_tile[:, :, :ndim] = boxes_tile[:, :, :ndim] + tile_positions

        n_objects = len(self.data)

        if n_objects:
            # Remove the boxes from the boxes data that are in the tile
            boxes_clean = self.data[~tile_filter]

            # Merge the tile data with the cleaned boxes data
            merged_boxes = np.vstack((boxes_clean, boxes_tile))

            # Do the same for the meta
            merged_boxes_meta = merge_meta_tile(
                self.meta, boxes_tile_meta, n_objects, tile_filter
            )
        else:
            merged_boxes = boxes_tile
            merged_boxes_meta = boxes_tile_meta

        self.data = merged_boxes
        self.meta = merged_boxes_meta

    @classmethod
    def serialize(cls, boxes: Optional[np.ndarray], client_origin: str) -> Optional[List[Feature]]:
        if boxes is None:
            return None
        
        features = []
        for i, box in enumerate(boxes):
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

    @classmethod
    def deserialize(cls, serialized_data: Optional[List[Dict[str, Any]]], client_origin: str) -> Optional[np.ndarray]:
        if serialized_data is None:
            return None
        boxes = np.array(
            [feature["geometry"]["coordinates"] for feature in serialized_data]
        )
        boxes = np.array(
            [box[0] for box in boxes]
        )  # We get back a shape of (N, 1, 5, 2) - so we remove dim. 1
        if len(boxes) > 0:
            boxes = boxes[:, :-1]  # Remove the last element
            boxes = boxes[:, :, ::-1]  # Invert XY
        return boxes

    @classmethod
    def _get_initial_data(cls, pixel_domain: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros((0, 4, len(pixel_domain)), dtype=np.float32)

    @classmethod
    def validate_data(cls, data, meta, constraints):
        main, extra = constraints
        if extra["required"] is False:
            return

        assert isinstance(
            data, np.ndarray
        ), f"Boxes data ({type(data)}) is not a Numpy array"
        
        assert len(data.shape) == 3, "Boxes data should have shape (N, 4, D)"
