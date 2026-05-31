"""Common image processing operations implemented as Imaging Server Kit Algorithms."""

from pathlib import Path

import numpy as np
import skimage.data
import scipy.ndimage as ndi
from skimage.filters import gaussian, sobel, median, threshold_otsu
from skimage.morphology import (
    disk,
    label,
    remove_small_objects,
    isotropic_erosion,
    isotropic_dilation,
    isotropic_opening,
    isotropic_closing,
)
from skimage.transform import rescale
from skimage.exposure import rescale_intensity
from skimage.util import crop, img_as_float

import imaging_server_kit as sk


## Process ##


# - Gaussian
@sk.algorithm(
    name="Process - Gaussian filter",
    description="Apply a Gaussian filter (blur) to an image.",
    project_url="https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian",
    tags=["Process", "Scikit-image"],
    parameters={
        "image": sk.Image(),
        "sigma": sk.Float(
            name="Sigma",
            min=0,
            max=100,
            step=0.1,
            default=1,
        ),
        "mode": sk.Choice(
            name="Mode",
            items=["reflect", "constant", "nearest", "mirror", "wrap"],
            default="nearest",
        ),
        "preserve_range": sk.Bool(name="Preserve range", default=True),
    },
    samples=[{"image": skimage.data.camera()}],
)
def gaussian_algo(image, sigma, mode, preserve_range):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    filtered = gaussian(image, sigma=sigma, mode=mode, preserve_range=preserve_range)

    return sk.Image(
        filtered,
        name="Filtered image (Gaussian)",
        contrast_limits=[filtered.min(), filtered.max()],
    )


# - Median
@sk.algorithm(
    name="Process - Median filter",
    description="Apply a median filter to an image.",
    project_url="https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.median",
    tags=["Process", "Scikit-image"],
    parameters={
        "image": sk.Image(),
        "size": sk.Integer(name="Size", min=1, max=100, default=2),
    },
    samples=[{"image": skimage.data.camera()}],
)
def median_algo(image, size):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    filtered = median(image, footprint=disk(size))

    return sk.Image(
        filtered,
        name="Filtered image (Median)",
        contrast_limits=[filtered.min(), filtered.max()],
    )


# - Variance
@sk.algorithm(
    name="Process - Variance filter",
    description="Apply a variance filter to an image.",
    tags=["Process", "Scikit-image"],
    parameters={
        "image": sk.Image(),
        "size": sk.Integer(name="Size", min=1, max=100, default=2),
    },
    samples=[{"image": skimage.data.camera()}],
)
def variance_algo(image, size):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    _img = image.astype(float)

    mean = ndi.uniform_filter(_img, size=size)
    mean_sq = ndi.uniform_filter(_img**2, size=size)

    filtered = mean_sq - mean**2

    return sk.Image(
        filtered,
        name="Filtered image (Variance)",
        contrast_limits=[filtered.min(), filtered.max()],
    )


# - Sobel
@sk.algorithm(
    name="Process - Sobel filter",
    description="Apply a Sobel filter to an image.",
    project_url="https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.sobel",
    tags=["Process", "Scikit-image"],
    samples=[{"image": skimage.data.camera()}],
)
def sobel_algo(image):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    filtered = sobel(image)

    return sk.Image(
        filtered,
        name="Filtered image (Sobel)",
        contrast_limits=[filtered.min(), filtered.max()],
    )


# - Invert image
@sk.algorithm(
    name="Process - Invert image",
    description="Invert the image intensity.",
    tags=["Process"],
    samples=[{"image": skimage.data.camera()}],
)
def invert_algo(image):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    inverted = -image

    return sk.Image(
        inverted,
        name="Inverted image",
        contrast_limits=[inverted.min(), inverted.max()],
    )


# - Rescale intensity
@sk.algorithm(
    parameters={
        "image": sk.Image(),
        "min_val": sk.Float(name="Min", default=0.0, min=-1e9, max=1e9),
        "max_val": sk.Float(name="Max", default=1.0, min=-1e9, max=1e9),
    },
    name="Process - Rescale intensity",
    description="Rescale the image intensity.",
    tags=["Process"],
    samples=[{"image": skimage.data.camera()}],
)
def rescale_intensity_algo(image, min_val, max_val):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if max_val < min_val:
        return sk.Notification(
            "Max value must be larger than min value", level="warning"
        )

    normalized = rescale_intensity(image, out_range=(min_val, max_val))

    return sk.Image(
        normalized, name="Normalized image", contrast_limits=[min_val, max_val]
    )


# - Blobness
@sk.algorithm(
    parameters={
        "image": sk.Image(),
        "min_sigma": sk.Float(
            name="Min sigma", min=0.1, max=1000.0, default=1.0, step=0.1
        ),
        "max_sigma": sk.Float(
            name="Max sigma", min=0.1, max=1000.0, default=10.0, step=0.1
        ),
        "num_sigma": sk.Integer(name="Num sigma", min=1, max=1000, default=10),
    },
    name="Process - Blobness",
    description="Apply a blobness filter to an image.",
    tags=["Process"],
    samples=[{"image": skimage.data.camera()}],
)
def blobness(image, min_sigma, max_sigma, num_sigma):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    image = img_as_float(image)

    sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    image_cube = np.empty(image.shape + (len(sigma_list),), dtype=np.float32)
    for i, s in enumerate(sigma_list):
        image_cube[..., i] = -ndi.gaussian_laplace(image, s) * np.mean(s) ** 2

    blobness = np.max(image_cube, axis=-1)

    return sk.Image(blobness, name="Filtered image (Blobness)")


## Utils ##

# - Rescale
@sk.algorithm(
    parameters={
        "image": sk.Image(),
        "scale": sk.Float(name="Scale", default=0.5, min=0.1, max=16.0),
        "order": sk.Integer(name="Interpolation order", default=1, min=0, max=3),
    },
    name="Utils - Rescale image",
    description="Rescale the image.",
    tags=["Utils"],
    samples=[{"image": skimage.data.camera()}],
)
def rescale_algo(image, scale, order):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    rescaled = rescale(
        image, scale=scale, order=order, anti_aliasing=True, preserve_range=True
    )

    return sk.Image(rescaled, name="Rescaled image")


# - Crop
@sk.algorithm(
    parameters={
        "image": sk.Image(),
        "crop_width": sk.Integer(
            name="Crop border width",
            description="Number of values to remove from the edges of each axis.",
            default=0,
            min=0,
            max=10_000,
        ),
    },
    name="Utils - Centered crop",
    description="Extract a crop from the center of the image.",
    project_url="https://scikit-image.org/docs/0.25.x/api/skimage.util.html#skimage.util.crop",
    tags=["Utils"],
    samples=[{"image": skimage.data.camera()}],
)
def crop_algo(image, crop_width):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    cropped = crop(image, crop_width=crop_width)

    return sk.Image(cropped, name="Cropped image")


## Detect ##


# - Blob detector
@sk.algorithm(
    name="Detect - Blobs",
    description="Blob detection algorithm implemented with a Laplacian of Gaussian (LoG) filter.",
    project_url="https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log",
    tags=["Scikit-image"],
    parameters={
        "image": sk.Image(description="Input image (2D, 3D).", dimensionality=[2, 3]),
        "min_sigma": sk.Integer(
            name="Min sigma",
            description="Minimum standard deviation of the Gaussian kernel, in pixels.",
            default=5,
            min=1,
            max=1000,
            step=1,
        ),
        "max_sigma": sk.Integer(
            name="Max sigma",
            description="Maximum standard deviation of the Gaussian kernel, in pixels.",
            default=10,
            min=1,
            max=1000,
            step=1,
        ),
        "num_sigma": sk.Integer(
            name="Num sigma",
            description="Number of intermediate sigma values to compute between the min_sigma and max_sigma.",
            default=10,
            min=1,
            max=1000,
            step=1,
        ),
        "threshold": sk.Float(
            name="Threshold",
            description="Lower bound for scale space maxima.",
            default=0.1,
            min=0.01,
            max=1.0,
            step=0.01,
        ),
    },
    samples=[
        {"image": Path(__file__).parent / "sample_images" / "blobs.tif"},
    ],
)
def blob_detector_algo(image, min_sigma, max_sigma, num_sigma, threshold):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    image = img_as_float(image)

    stack = skimage.feature.blob_log(
        image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )

    points = stack[:, : image.ndim]
    sigmas = stack[:, image.ndim]

    points_params = {
        "opacity": 0.7,
        "face_color": "sigma",
        "features": {"sigma": sigmas},
    }

    n_points = len(points)

    if n_points:
        return sk.Points(points, name="Detections", meta=points_params)
    else:
        return sk.Notification("No points detected.")


## Segmentation ##


# - Threshold (manual)
@sk.algorithm(
    name="Segmentation - Intensity threshold",
    description="Segment an image based on an intensity threshold.",
    tags=["Segmentation"],
    parameters={
        "image": sk.Image(),
        "threshold": sk.Float(
            name="Threshold (rel.)",
            description="Intensity threshold, relative to the image min() and max() values.",
            default=0.5,
            min=0,
            max=1,
            step=0.01,
        ),
        "dark_background": sk.Bool(name="Dark background", default=True),
    },
    samples=[
        {
            "image": skimage.data.coins(),
            "threshold": 0.45,
        },
        {
            "image": skimage.data.camera(),
            "dark_background": False,
        },
    ],
)
def threshold_manual(image, threshold, dark_background):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    thresh_rel = threshold * (image.max() - image.min())

    if dark_background:
        mask = image > thresh_rel
    else:
        mask = image <= thresh_rel

    return sk.Mask(mask, name="Binarized image")


# - Threshold Otsu
@sk.algorithm(
    name="Segmentation - Otsu's threshold",
    description="Otsu's automatic thresholding algorithm.",
    tags=["Segmentation", "Scikit-image"],
    samples=[{"image": skimage.data.camera()}],
)
def otsu_threshold(image):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    mask = image > threshold_otsu(image)

    return sk.Mask(mask, name="Binary mask (auto)")


## Mask ##


# - Label
@sk.algorithm(
    name="Mask - Label",
    description="Connected components labelling.",
    tags=["Mask", "Scikit-image"],
)
def label_algo(mask):
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    labelled = label(mask)

    return sk.Mask(labelled, name="Labelled mask", merger="instances")


# - Remove small objects
@sk.algorithm(
    parameters={
        "mask": sk.Mask(),
        "min_size": sk.Integer(
            name="Min size",
            description="Minimum size of the remaining objects.",
            default=10,
            min=1,
            max=100_000,
        ),
    },
    name="Mask - Remove small objects",
    description="Remove objects of size below a predetermined threshold.",
    tags=["Mask", "Scikit-image"],
)
def remove_small_objects_algo(mask, min_size):
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    new_mask = remove_small_objects(mask, min_size)

    return sk.Mask(new_mask, name="Mask (small objects removed)")


# - Keep biggest objects
def keep_biggest_objects(labelled: np.ndarray, n=1) -> np.ndarray:
    """Remove all but the N biggest objects in a labelled array."""
    uniques, counts = np.unique(labelled, return_counts=1)

    # Ignore the background if it's there
    if uniques[0] == 0:
        uniques = uniques[1:]
        counts = counts[1:]

    # Sort unique values by counts (descending), then extract the N unique values corresponding to the biggest objects
    biggest_labels = uniques[np.argsort(counts)[::-1][:n]]

    biggest_objects_filt = np.isin(labelled, biggest_labels)

    biggest_objects_mask = labelled.copy()
    biggest_objects_mask[~biggest_objects_filt] = 0

    return biggest_objects_mask


@sk.algorithm(
    parameters={
        "mask": sk.Mask(),
        "n_objects": sk.Integer(
            name="N",
            description="Number of objects to keep, based on their size.",
            default=1,
            min=1,
            max=100_000,
        ),
    },
    name="Mask - Keep biggest objects",
    description="Keep a predetermined number of objects based on their size.",
    tags=["Mask"],
)
def keep_biggest_objects_algo(mask, n_objects):
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    new_mask = keep_biggest_objects(mask, n_objects)

    return sk.Mask(new_mask, name=f"Mask (N={n_objects})")


# - Fill holes
@sk.algorithm(
    name="Mask - Fill holes",
    description="Fill holes in a binary mask.",
    tags=["Mask"],
)
def fill_holes_algo(mask):
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    labelled = ndi.binary_fill_holes(mask)

    return sk.Mask(labelled, name="Filled mask")


# - Erode / Dilate / Open / Close
@sk.algorithm(
    parameters={
        "mask": sk.Mask(),
        "operation": sk.Choice(
            name="Operation",
            items=["erosion", "dilation", "opening", "closing"],
            default="erosion",
        ),
        "radius": sk.Integer(
            name="N",
            description="Number of steps for the morphological operation.",
            default=1,
            min=1,
            max=100_000,
        ),
    },
    name="Mask - Morphological operators",
    description="Binary morphological operators (erode, dilate, open, close).",
    tags=["Mask"],
)
def morphological_operators(mask, operation, radius):
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    mappings = {
        "erosion": (isotropic_erosion, "eroded"),
        "dilation": (isotropic_dilation, "dilated"),
        "opening": (isotropic_opening, "opened"),
        "closing": (isotropic_closing, "closed"),
    }

    func, operated = mappings[operation]

    new_mask = func(mask, radius=radius)

    return sk.Mask(new_mask, name=f"Mask ({operated})")


## Math operations ##


# - Between two images
@sk.algorithm(
    parameters={
        "image1": sk.Image(name="Image 1"),
        "image2": sk.Image(name="Image 2"),
        "operation": sk.Choice(
            name="Operation",
            items=["subtract", "add", "divide", "multiply"],
            default="subtract",
        ),
    },
    name="Math - Image-to-Image",
    description="Simple mathematical operations between two images.",
    tags=["Math"],
)
def math_operators_images(image1, image2, operation):
    if (image1 is None) or (image2 is None):
        return sk.Notification("Two images are required!", level="warning")

    mappings = {
        "subtract": (np.subtract, "subtracted"),
        "add": (np.add, "added"),
        "divide": (np.divide, "divided"),
        "multiply": (np.multiply, "multiplied"),
    }

    func, operated = mappings[operation]

    new_image = func(image1, image2)

    return sk.Image(new_image, name=f"Image ({operated})")


# - With constant values
@sk.algorithm(
    parameters={
        "image": sk.Image(name="Image 1"),
        "value": sk.Float(name="Value", min=-1e9, max=1e9),
        "operation": sk.Choice(
            name="Operation",
            items=["subtract", "add", "divide", "multiply"],
            default="subtract",
        ),
    },
    name="Math - Image Math",
    description="Simple mathematical operations on an image.",
    tags=["Math"],
)
def math_operators_algo(image, value, operation):
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    mappings = {
        "subtract": (np.subtract, "subtracted"),
        "add": (np.add, "added"),
        "divide": (np.divide, "divided"),
        "multiply": (np.multiply, "multiplied"),
    }

    func, operated = mappings[operation]

    new_image = func(image, value)

    return sk.Image(new_image, name=f"Image ({operated})")


# - Between two masks
@sk.algorithm(
    parameters={
        "mask1": sk.Mask(name="Mask 1"),
        "mask2": sk.Mask(name="Mask 2"),
        "operation": sk.Choice(
            name="Operation", items=["AND", "OR", "XOR", "NOT"], default="AND"
        ),
    },
    name="Math - Mask-to-Mask",
    description="Simple logical operations between two masks.",
    tags=["Math"],
)
def logial_operators_masks(mask1, mask2, operation):
    if (mask1 is None) or (mask2 is None):
        return sk.Notification("Two masks are required!", level="warning")

    mappings = {
        "AND": (np.logical_and, "AND"),
        "OR": (np.logical_or, "OR"),
        "XOR": (np.logical_xor, "XOR"),
        "NOT": (np.logical_not, "NOT"),
    }

    func, operated = mappings[operation]

    new_mask = func(mask1, mask2)

    return sk.Mask(new_mask, name=f"Mask ({operated})")


# - Between an image and a mask (set to value)
@sk.algorithm(
    parameters={
        "image": sk.Image(),
        "mask": sk.Mask(),
        "value": sk.Float(name="Set value", default=0, min=-1e9, max=1e9),
    },
    name="Math - Image masking",
    description="Set the value of an image inside a mask.",
    tags=["Math"],
)
def image_masking_algo(image, mask, value):
    if image is None:
        return sk.Notification("An image is required!", level="warning")
    
    if mask is None:
        return sk.Notification("A mask is required!", level="warning")

    new_image = image.copy()
    
    new_image[mask.astype(bool)] = value

    return sk.Image(new_image, name=f"Image (masked)")


## Special ##

# # Special - Translate an image (OK to debug, but not super useful)
# @sk.algorithm(
#     parameters={
#         "image": sk.Image(),
#         "position": sk.Integer(name="Position", min=-100_000, max=100_000, default=0, step=5),
#     },
#     name="Special - Translate image",
#     description="Translate an image.",
#     tags=["Special"],
#     samples=[{"image": skimage.data.camera()}],
# )
# def translate_image(image, position):
#     if image is None:
#         return sk.Notification("An image is required!", level="warning")

#     position = tuple([position]*image.ndim)
    
#     return sk.Image(image, name="Translated", position=position, colormap="viridis")
