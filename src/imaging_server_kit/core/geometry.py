# """
# Geometry featurization module for the Imaging Server Kit.
# """

# from typing import List
# from skimage.draw import polygon2mask
# from geojson import Feature, Polygon
# import numpy as np
# import imantics


# def mask2features(segmentation_mask: np.ndarray) -> List[Feature]:
#     """
#     Args:
#         segm_mask: Segmentation mask with the background pixels set to zero and the pixels assigned to a segmented 
#         class set to an int value

#     Returns:
#         A list containing the contours of each object as a geojson.Feature
#     """
#     features = []
#     indices = np.unique(segmentation_mask)
#     indices = indices[indices != 0] # remove background

#     if indices.size == 0:
#         return features
    
#     for pixel_class in indices:
#         mask = segmentation_mask == int(pixel_class)
#         polygons = imantics.Mask(mask).polygons()
#         for detection_id, contour in enumerate(polygons.points, start=1):
#             coords = np.array(contour)
#             coords = np.vstack([coords, coords[0]])  # Close the polygon for QuPath
#             try:
#                 geom = Polygon(coordinates=[coords.tolist()], validate=True)
#                 feature = Feature(
#                     geometry=geom, 
#                     properties={
#                         "Detection ID": detection_id, 
#                         "Class": int(pixel_class),  # the int() casting solves a bug with json serialization
#                     }
#                 )
#                 features.append(feature)
#             except ValueError:
#                 print("Invalid polygon geometry.")

#     return features


# def features2mask(features, image_shape):
#     segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
#     for feature in features:
#         feature_coordinates = np.array(feature["geometry"]["coordinates"])
#         feature_coordinates = feature_coordinates[0, :, :]  # Remove an extra dimension
#         feature_coordinates = feature_coordinates[:, ::-1]  # Invert XY
#         feature_mask = polygon2mask(image_shape, feature_coordinates)
#         feature_properites = feature.get("properties")
#         feature_class = feature_properites.get("Class")
#         segmentation_mask[feature_mask] = feature_class
#     return segmentation_mask


# def instance_mask2features(segmentation_mask: np.ndarray) -> List[Feature]:
#     """
#     Args:
#         segm_mask: Segmentation mask with the background pixels set to zero and the pixels assigned to a segmented
#          object instance set to an int value

#     Returns:
#         A list containing the contours of each object as a geojson.Feature
#     """
#     features = []
#     indices = np.unique(segmentation_mask)
#     indices = indices[indices != 0] # remove background

#     if indices.size == 0:
#         return features
    
#     for detection_id in indices:
#         mask = segmentation_mask == int(detection_id)
#         polygons = imantics.Mask(mask).polygons()
#         for contour in polygons.points:
#             coords = np.array(contour)
#             coords = np.vstack([coords, coords[0]])  # Close the polygon for QuPath
#             try:
#                 geom = Polygon(coordinates=[coords.tolist()], validate=True)
#                 feature = Feature(
#                     geometry=geom, 
#                     properties={
#                         "Detection ID": int(detection_id), 
#                         "Class": 1
#                     }
#                 )
#                 features.append(feature)
#             except ValueError:
#                 print("Invalid polygon geometry.")

#     return features


# def features2instance_mask(features, image_shape):
#     segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
#     for feature in features:
#         feature_coordinates = np.array(feature["geometry"]["coordinates"])
#         feature_coordinates = feature_coordinates[0, :, :]  # Remove an extra dimension
#         feature_coordinates = feature_coordinates[:, ::-1]  # Invert XY
#         feature_mask = polygon2mask(image_shape, feature_coordinates)
#         feature_properites = feature.get("properties")
#         feature_id = feature_properites.get("Detection ID")
#         segmentation_mask[feature_mask] = feature_id
#     return segmentation_mask


# def mask2features_3d(segmentation_mask: np.ndarray) -> List[Feature]:
#     features = []
#     for z_idx, mask_2d in enumerate(segmentation_mask):
#         features_2d = mask2features(mask_2d)
#         for feature_2d in features_2d:
#             feature_2d.properties["z_idx"] = z_idx
#             features.append(feature_2d)
#     return features


# def features2mask_3d(features, image_shape):
#     segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
#     _, ry, rx = image_shape
#     for feature in features:
#         feature_xy_coordinates = np.array(feature["geometry"]["coordinates"])
#         feature_coordinates = feature_coordinates[0, :, :]  # Remove an extra dimension
#         feature_xy_coordinates = feature_xy_coordinates[:, ::-1]  # Invert XY
#         feature_mask = polygon2mask((ry, rx), feature_xy_coordinates)
#         feature_z_idx = feature["properties"]["z_idx"]
#         feature_properites = feature.get("properties")
#         feature_id = feature_properites.get("Class")
#         segmentation_mask[feature_z_idx][feature_mask] = feature_id
#     return segmentation_mask


# def instance_mask2features_3d(segmentation_mask: np.ndarray) -> List[Feature]:
#     features = []
#     for z_idx, mask_2d in enumerate(segmentation_mask):
#         features_2d = instance_mask2features(mask_2d)
#         for feature_2d in features_2d:
#             feature_2d.properties["z_idx"] = z_idx
#             features.append(feature_2d)
#     return features


# def features2instance_mask_3d(features, image_shape):
#     segmentation_mask = np.zeros(image_shape, dtype=np.uint16)
#     _, ry, rx = image_shape
#     for feature in features:
#         feature_xy_coordinates = np.array(feature["geometry"]["coordinates"])
#         feature_coordinates = feature_coordinates[0, :, :]  # Remove an extra dimension
#         feature_xy_coordinates = feature_xy_coordinates[:, ::-1]  # Invert XY
#         feature_mask = polygon2mask((ry, rx), feature_xy_coordinates)
#         feature_z_idx = feature["properties"]["z_idx"]
#         feature_properites = feature.get("properties")
#         feature_id = feature_properites.get("Detection ID")
#         segmentation_mask[feature_z_idx][feature_mask] = feature_id
#     return segmentation_mask