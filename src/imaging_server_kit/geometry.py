from typing import List
from skimage.draw import polygon2mask
from geojson import Feature, Polygon, Point, LineString
import numpy as np
from imantics import Mask


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
    indices = indices[indices != 0] # remove background

    if indices.size == 0:
        return features
    
    for detection_id in indices:
        mask = segmentation_mask == int(detection_id)
        polygons = Mask(mask).polygons()
        for contour in polygons.points:
            coords = np.array(contour)
            coords = np.vstack([coords, coords[0]])  # Close the polygon for QuPath
            geom = Polygon(coordinates=[coords.tolist()])
            feature = Feature(
                geometry=geom, 
                properties={
                    "Detection ID": int(detection_id), 
                    "Class": 1
                }
            )
            features.append(feature)

    return features


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
    indices = indices[indices != 0] # remove background

    if indices.size == 0:
        return features
    
    for pixel_class in indices:
        mask = segmentation_mask == int(pixel_class)
        polygons = Mask(mask).polygons()
        for detection_id, contour in enumerate(polygons.points, start=1):
            coords = np.array(contour)
            coords = np.vstack([coords, coords[0]])  # Close the polygon for QuPath
            geom = Polygon(coordinates=[coords.tolist()])
            feature = Feature(
                geometry=geom, 
                properties={
                    "Detection ID": detection_id, 
                    "Class": pixel_class
                }
            )
            features.append(feature)

    return features


def features2mask(features, image_shape):
    segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
    for feature in features:
        feature_coordinates = np.array(feature["geometry"]["coordinates"])
        feature_coordinates = np.squeeze(feature_coordinates)
        feature_coordinates = feature_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask(image_shape, feature_coordinates)
        feature_properites = feature.get("properties")
        feature_class = feature_properites.get("Class")
        segmentation_mask[feature_mask] = feature_class
    return segmentation_mask


def features2instance_mask(features, image_shape):
    segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
    for feature in features:
        feature_coordinates = np.array(feature["geometry"]["coordinates"])
        feature_coordinates = np.squeeze(feature_coordinates)
        feature_coordinates = feature_coordinates[:, ::-1]  # Invert XY
        feature_mask = polygon2mask(image_shape, feature_coordinates)
        feature_properites = feature.get("properties")
        feature_id = feature_properites.get("Detection ID")
        segmentation_mask[feature_mask] = feature_id
    return segmentation_mask


def boxes2features(boxes: np.ndarray) -> List[Feature]:
    """
    Convert an array of bounding boxes to a list of geojson-like features.

    Args:
        boxes (np.ndarray): A numpy array of bounding boxes, where each bounding box
                            is represented by a list of coordinates.

    Returns:
        List[Feature]: A list of geojson-like Feature objects, each containing a Polygon
                       geometry and a property "Detection ID" indicating the index of the box.
    """
    features = []
    for i, box in enumerate(boxes):
        coords = np.array(box)[:, ::-1]  # Invert XY
        coords = coords.tolist()
        coords.append(
            coords[0]
        )  # Add the first element at the end to close the Polygon
        geom = Polygon(coordinates=[coords])
        features.append(Feature(geometry=geom, properties={"Detection ID": i}))
    return features


def features2boxes(features):
    boxes = np.array([feature["geometry"]["coordinates"] for feature in features])
    boxes = np.squeeze(boxes)
    boxes = boxes[:, :-1] # Remove the last element
    boxes = boxes[:, :, ::-1]  # Invert XY
    return boxes


def points2features(points: np.ndarray) -> List[Feature]:
    """
    Convert an array of points into a list of GeoJSON-like features.

    Each point is converted into a GeoJSON Point geometry and wrapped in a Feature
    with a property "Detection ID" that corresponds to the index of the point in the input array.

    Args:
        points (np.ndarray): A numpy array of points where each point is represented by its coordinates.

    Returns:
        List[Feature]: A list of features where each feature contains a Point geometry and properties.
    """
    features = []
    point_coords = np.array(points)[:, ::-1]  # Invert XY
    for i, point in enumerate(point_coords):
        geom = Point(coordinates=[np.array(point).tolist()])
        features.append(Feature(geometry=geom, properties={"Detection ID": i}))
    return features


def features2points(features):
    points = np.array([feature["geometry"]["coordinates"] for feature in features])
    points = np.squeeze(points)
    points = points[:, ::-1]  # Invert XY
    return points


def vectors2features(vectors: np.ndarray) -> List[Feature]:
    """
    Convert an array of vectors to a list of GeoJSON-like Feature objects.

    Each vector is inverted (XY to YX) and converted to a LineString geometry.
    The resulting features include a "Detection ID" property corresponding to the index of the vector.

    Args:
        vectors (np.ndarray): A numpy array of vectors.

    Returns:
        List[Feature]: A list of Feature objects with LineString geometries and "Detection ID" properties.
    """
    features = []
    vectors = vectors[:, :, ::-1]  # Invert XY
    for i, vector in enumerate(vectors):
        point_start = list(vector[0])
        point_end = list(vector[0] + vector[1])
        coords = [point_start, point_end]
        geom = LineString(coordinates=coords)
        features.append(Feature(geometry=geom, properties={"Detection ID": i}))
    return features


def features2vectors(features):
    vectors_arr = np.array(
        [feature["geometry"]["coordinates"] for feature in features]
    )
    origins = vectors_arr[:, 0]
    displacements = vectors_arr[:, 1] - origins
    vectors = np.stack((origins, displacements))
    vectors = np.rollaxis(vectors, 1)
    print(f"{vectors.shape=}")
    vectors = vectors[:, :, ::-1]  # Invert XY
    return vectors