"""Common image processing operations implemented as Imaging Server Kit Algorithms."""

from pathlib import Path

import numpy as np
import skimage.data
import scipy.ndimage as ndi
from skimage.filters import gaussian, sobel, median, threshold_otsu
from skimage.morphology import (
    ball,
    disk,
    label,
    isotropic_erosion,
    isotropic_dilation,
    isotropic_opening,
    isotropic_closing,
)
from skimage.transform import rescale
from skimage.exposure import rescale_intensity
from skimage.util import crop, img_as_float

import imaging_server_kit as sk


## Helpers ##


def _first_axis_param() -> sk.Bool:
    return sk.Bool(
        name="Across first axis",
        description="Process each slice along axis 0 independently (e.g. the frames of a time series).",
        default=False,
    )


def _first_axis_unavailable(array, first_axis, rgb=False):
    """A warning if `first_axis` is set but `array` has no leading axis (the RGB channel axis does not count)."""
    if first_axis and array.ndim - int(rgb) < 3:
        return sk.Notification(
            "Applying along the first axis requires a leading axis (e.g. time).",
            level="warning",
        )


def _apply_first_axis(func, image, first_axis):
    """Apply `func` to `image`, or to each slice along axis 0 if `first_axis` is set."""
    if not first_axis:
        return func(image)
    return np.stack([func(frame) for frame in image])


## Process ##


# - Gaussian
@sk.algorithm(
    name="Process - Gaussian filter",
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
        "rgb": sk.Bool(name="RGB image", default=False),
        "first_axis": _first_axis_param(),
    },
    samples=[{"image": skimage.data.camera()}],
)
def gaussian_algo(image, sigma, mode, preserve_range, rgb, first_axis=False):
    """Apply a Gaussian filter (blur) to an image."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if warning := _first_axis_unavailable(image, first_axis, rgb):
        return warning

    def _gaussian(img):
        return gaussian(
            img,
            sigma=sigma,
            mode=mode,
            preserve_range=preserve_range,
            channel_axis=-1 if rgb else None,
        )

    filtered = _apply_first_axis(_gaussian, image, first_axis)

    return sk.Image(
        filtered,
        name="Filtered image (Gaussian)",
        contrast_limits=[filtered.min(), filtered.max()],
        rgb=rgb,
    )


# - Median
@sk.algorithm(
    name="Process - Median filter",
    project_url="https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.median",
    tags=["Process", "Scikit-image"],
    parameters={
        "image": sk.Image(),
        "size": sk.Integer(name="Size", min=1, max=100, default=2),
        "first_axis": _first_axis_param(),
    },
    samples=[{"image": skimage.data.camera()}],
)
def median_algo(image, size, first_axis=False):
    """Apply a median filter to an image."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if warning := _first_axis_unavailable(image, first_axis):
        return warning

    def _median(img):
        footprint = disk(size) if img.ndim == 2 else ball(size)
        return median(img, footprint=footprint)

    filtered = _apply_first_axis(_median, image, first_axis)

    return sk.Image(
        filtered,
        name="Filtered image (Median)",
        contrast_limits=[filtered.min(), filtered.max()],
    )


# - Variance
@sk.algorithm(
    name="Process - Variance filter",
    tags=["Process"],
    parameters={
        "image": sk.Image(),
        "size": sk.Integer(name="Size", min=1, max=100, default=2),
        "first_axis": _first_axis_param(),
    },
    samples=[{"image": skimage.data.camera()}],
)
def variance_algo(image, size, first_axis=False):
    """Apply a variance filter to an image."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if warning := _first_axis_unavailable(image, first_axis):
        return warning

    def _variance(img):
        _img = img.astype(float)
        mean = ndi.uniform_filter(_img, size=size)
        mean_sq = ndi.uniform_filter(_img**2, size=size)
        return mean_sq - mean**2

    filtered = _apply_first_axis(_variance, image, first_axis)

    return sk.Image(
        filtered,
        name="Filtered image (Variance)",
        contrast_limits=[filtered.min(), filtered.max()],
    )


# - Sobel
@sk.algorithm(
    name="Process - Sobel filter",
    project_url="https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.sobel",
    tags=["Process", "Scikit-image"],
    parameters={
        "image": sk.Image(),
        "first_axis": _first_axis_param(),
    },
    samples=[{"image": skimage.data.camera()}],
)
def sobel_algo(image, first_axis=False):
    """Apply a Sobel filter to an image."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if warning := _first_axis_unavailable(image, first_axis):
        return warning

    filtered = _apply_first_axis(sobel, image, first_axis)

    return sk.Image(
        filtered,
        name="Filtered image (Sobel)",
        contrast_limits=[filtered.min(), filtered.max()],
    )


# - Invert image
@sk.algorithm(
    name="Process - Invert image",
    tags=["Process"],
    samples=[{"image": skimage.data.camera()}],
)
def invert_algo(image):
    """Invert the image intensity."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if image.dtype == bool:
        inverted = np.logical_not(image)
    else:
        # Mirror the intensities within [min, max] (no overflow for unsigned integers)
        inverted = image.max() - (image - image.min())

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
        "first_axis": _first_axis_param(),
    },
    name="Process - Rescale intensity",
    tags=["Process"],
    samples=[{"image": skimage.data.camera()}],
)
def rescale_intensity_algo(image, min_val, max_val, first_axis=False):
    """Rescale the image intensity."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if max_val < min_val:
        return sk.Notification(
            "Max value must be larger than min value", level="warning"
        )

    if warning := _first_axis_unavailable(image, first_axis):
        return warning

    normalized = _apply_first_axis(
        lambda img: rescale_intensity(img, out_range=(min_val, max_val)),
        image,
        first_axis,
    )

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
        "first_axis": _first_axis_param(),
    },
    name="Process - Blobness",
    tags=["Process"],
    samples=[{"image": skimage.data.camera()}],
)
def blobness(image, min_sigma, max_sigma, num_sigma, first_axis=False):
    """Apply a blobness filter to an image."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if warning := _first_axis_unavailable(image, first_axis):
        return warning

    image = img_as_float(image)

    sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    def _blobness(img):
        image_cube = np.empty(img.shape + (len(sigma_list),), dtype=np.float32)
        for i, s in enumerate(sigma_list):
            image_cube[..., i] = -ndi.gaussian_laplace(img, s) * s**2
        return np.max(image_cube, axis=-1)

    blobness = _apply_first_axis(_blobness, image, first_axis)

    return sk.Image(blobness, name="Filtered image (Blobness)")


## Utils ##


# - Rescale
@sk.algorithm(
    parameters={
        "image": sk.Image(),
        "scale": sk.Float(name="Scale", default=0.5, min=0.1, max=16.0),
        "order": sk.Integer(name="Interpolation order", default=1, min=0, max=3),
        "rgb": sk.Bool(name="RGB image", default=False),
        "first_axis": _first_axis_param(),
    },
    name="Utils - Rescale image",
    tags=["Utils"],
    samples=[{"image": skimage.data.camera()}],
)
def rescale_algo(image, scale, order, rgb, first_axis=False):
    """Rescale the image."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if warning := _first_axis_unavailable(image, first_axis, rgb):
        return warning

    def _rescale(img):
        return rescale(
            img,
            scale=scale,
            order=order,
            anti_aliasing=True,
            preserve_range=True,
            channel_axis=-1 if rgb else None,
        )

    rescaled = _apply_first_axis(_rescale, image, first_axis)

    return sk.Image(rescaled, name="Rescaled image", rgb=rgb)


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
        "rgb": sk.Bool(name="RGB image", default=False),
        "first_axis": _first_axis_param(),
    },
    name="Utils - Centered crop",
    project_url="https://scikit-image.org/docs/0.25.x/api/skimage.util.html#skimage.util.crop",
    tags=["Utils"],
    samples=[{"image": skimage.data.camera()}],
)
def crop_algo(image, crop_width, rgb, first_axis=False):
    """Extract a crop from the center of the image."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if warning := _first_axis_unavailable(image, first_axis, rgb):
        return warning

    def _crop(img):
        # For RGB images, the channel axis is left untouched
        n_spatial = img.ndim - 1 if rgb else img.ndim
        widths = [(crop_width, crop_width)] * n_spatial + [(0, 0)] * (
            img.ndim - n_spatial
        )
        return crop(img, crop_width=widths)

    cropped = _apply_first_axis(_crop, image, first_axis)

    return sk.Image(cropped, name="Cropped image", rgb=rgb)


## Detect ##


# - Blob detector
@sk.algorithm(
    name="Detect - Blobs",
    project_url="https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log",
    tags=["Detect", "Scikit-image"],
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
        "first_axis": _first_axis_param(),
    },
    samples=[
        {"image": Path(__file__).parent / "sample_images" / "blobs.tif"},
    ],
)
def blob_detector_algo(
    image, min_sigma, max_sigma, num_sigma, threshold, first_axis=False
):
    """Blob detection algorithm implemented with a Laplacian of Gaussian (LoG) filter."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if warning := _first_axis_unavailable(image, first_axis):
        return warning

    image = img_as_float(image)

    def _detect(img):
        return skimage.feature.blob_log(
            img,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
        )

    if first_axis:
        # Detect in each frame, then prepend the frame index to the coordinates
        stack = np.concatenate(
            [
                np.column_stack([np.full(len(blobs), idx), blobs])
                for idx, blobs in enumerate(_detect(frame) for frame in image)
            ]
        )
    else:
        stack = _detect(image)

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
    """Segment an image based on an intensity threshold."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    thresh = image.min() + threshold * (image.max() - image.min())

    if dark_background:
        mask = image > thresh
    else:
        mask = image <= thresh

    return sk.Mask(mask, name="Binarized image")


# - Threshold Otsu
@sk.algorithm(
    name="Segmentation - Otsu's threshold",
    tags=["Segmentation", "Scikit-image"],
    parameters={
        "image": sk.Image(),
        "first_axis": _first_axis_param(),
    },
    samples=[{"image": skimage.data.camera()}],
)
def otsu_threshold(image, first_axis=False):
    """Otsu's automatic thresholding algorithm."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if warning := _first_axis_unavailable(image, first_axis):
        return warning

    mask = _apply_first_axis(lambda img: img > threshold_otsu(img), image, first_axis)

    return sk.Mask(mask, name="Binary mask (auto)")


## Mask ##


# - Label
@sk.algorithm(
    name="Mask - Label",
    tags=["Mask", "Scikit-image"],
    parameters={
        "mask": sk.Mask(),
        "first_axis": _first_axis_param(),
    },
    tileable=True,
)
def label_algo(mask, first_axis=False):
    """Connected components labelling."""
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    if warning := _first_axis_unavailable(mask, first_axis):
        return warning

    labelled = _apply_first_axis(label, mask, first_axis)

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
        "mask_is_binary": sk.Bool(name="Mask is binary", default=False),
        "first_axis": _first_axis_param(),
    },
    name="Mask - Remove small objects",
    tags=["Mask", "Scikit-image"],
)
def remove_small_objects_algo(mask, min_size, mask_is_binary, first_axis=False):
    """Remove objects of size below a predetermined threshold."""
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    if warning := _first_axis_unavailable(mask, first_axis):
        return warning

    mask_dtype = mask.dtype

    def _remove_small_objects(m):
        if mask_is_binary:
            # We first label the mask
            m = label(m)

        # Set the labels of objects with fewer than `min_size` pixels to zero
        labels = m.astype(np.intp)
        counts = np.bincount(labels.ravel())
        new_m = m.copy()
        new_m[counts[labels] < min_size] = 0

        if mask_is_binary:
            # Back to a binary mask, with original dtype
            new_m = (new_m > 0).astype(mask_dtype)

        return new_m

    new_mask = _apply_first_axis(_remove_small_objects, mask, first_axis)
    merger = "default" if mask_is_binary else "instances"

    return sk.Mask(new_mask, name="Mask (small objects removed)", merger=merger)


# - Keep biggest objects
def keep_biggest_objects(mask: np.ndarray, n=1, mask_is_binary=False) -> np.ndarray:
    """Remove all but the N biggest objects in a labelled array."""
    if mask_is_binary:
        mask_dtype = mask.dtype
        mask = label(mask)
    
    uniques, counts = np.unique(mask, return_counts=True)

    # Ignore the background if it's there
    if uniques[0] == 0:
        uniques = uniques[1:]
        counts = counts[1:]

    # Sort unique values by counts (descending), then extract the N unique values corresponding to the biggest objects
    biggest_labels = uniques[np.argsort(counts)[::-1][:n]]

    biggest_objects_filt = np.isin(mask, biggest_labels)

    biggest_objects_mask = mask.copy()
    biggest_objects_mask[~biggest_objects_filt] = 0

    if mask_is_binary:
        # Back to a binary mask, with original dtype
        biggest_objects_mask = (biggest_objects_mask > 0).astype(mask_dtype)

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
        "mask_is_binary": sk.Bool(name="Mask is binary", default=False),
        "first_axis": _first_axis_param(),
    },
    name="Mask - Keep biggest objects",
    tags=["Mask"],
)
def keep_biggest_objects_algo(mask, n_objects, mask_is_binary, first_axis=False):
    """Keep a predetermined number of objects based on their size."""
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    if warning := _first_axis_unavailable(mask, first_axis):
        return warning

    new_mask = _apply_first_axis(
        lambda m: keep_biggest_objects(m, n_objects, mask_is_binary), mask, first_axis
    )

    return sk.Mask(new_mask, name=f"Mask (N={n_objects})")


# - Fill holes
@sk.algorithm(
    parameters={
        "mask": sk.Mask(),
        "first_axis": _first_axis_param(),
    },
    name="Mask - Fill holes",
    tags=["Mask"],
)
def fill_holes_algo(mask, first_axis=False):
    """Fill holes in a binary mask."""
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    if warning := _first_axis_unavailable(mask, first_axis):
        return warning

    labelled = _apply_first_axis(ndi.binary_fill_holes, mask, first_axis)

    return sk.Mask(labelled, name="Filled mask")


# - Invert mask
@sk.algorithm(
    name="Mask - Invert",
    tags=["Mask"],
    tileable=True,
)
def invert_mask_algo(mask):
    """Invert a binary mask."""
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    inverted = np.logical_not(mask).astype(mask.dtype)

    return sk.Mask(inverted, name="Inverted mask")


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
        "first_axis": _first_axis_param(),
    },
    name="Mask - Morphological operators",
    tags=["Mask"],
)
def morphological_operators(mask, operation, radius, first_axis=False):
    """Binary morphological operators (erode, dilate, open, close)."""
    if mask is None:
        return sk.Notification("A binary mask is required!", level="warning")

    mappings = {
        "erosion": (isotropic_erosion, "eroded"),
        "dilation": (isotropic_dilation, "dilated"),
        "opening": (isotropic_opening, "opened"),
        "closing": (isotropic_closing, "closed"),
    }

    func, operated = mappings[operation]

    if warning := _first_axis_unavailable(mask, first_axis):
        return warning

    new_mask = _apply_first_axis(lambda m: func(m, radius=radius), mask, first_axis)

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
    tags=["Math"],
    tileable=True,
)
def math_operators_images(image1, image2, operation):
    """Simple mathematical operations between two images."""
    if (image1 is None) or (image2 is None):
        return sk.Notification("Two images are required!", level="warning")

    mappings = {
        "subtract": (np.subtract, "subtracted"),
        "add": (np.add, "added"),
        "divide": (np.divide, "divided"),
        "multiply": (np.multiply, "multiplied"),
    }

    func, operated = mappings[operation]

    # Avoid integer overflow (e.g. negative values after subtracting uint8 images)
    new_image = func(image1.astype(float), image2.astype(float))

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
    tags=["Math"],
    tileable=True,
)
def math_operators_algo(image, value, operation):
    """Simple mathematical operations on an image."""
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
            name="Operation", items=["AND", "OR", "XOR"], default="AND"
        ),
    },
    name="Math - Mask-to-Mask",
    tags=["Math"],
    tileable=True,
)
def logical_operators_masks(mask1, mask2, operation):
    """Simple logical operations between two masks."""
    if (mask1 is None) or (mask2 is None):
        return sk.Notification("Two masks are required!", level="warning")

    mappings = {
        "AND": (np.logical_and, "AND"),
        "OR": (np.logical_or, "OR"),
        "XOR": (np.logical_xor, "XOR"),
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
        "rgb": sk.Bool(name="RGB image", default=False),
    },
    name="Math - Image masking",
    tags=["Math"],
    tileable=True,
)
def image_masking_algo(image, mask, value, rgb):
    """Set the value of an image inside a mask."""
    if image is None:
        return sk.Notification("An image is required!", level="warning")

    if mask is None:
        return sk.Notification("A mask is required!", level="warning")

    new_image = image.copy()

    # For RGB images, all channels are set to the value
    new_image[mask.astype(bool)] = value

    return sk.Image(new_image, name="Image (masked)", rgb=rgb)
