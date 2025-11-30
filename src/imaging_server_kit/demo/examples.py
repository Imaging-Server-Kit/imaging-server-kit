"""Algorithm examples for the Imaging Server Kit package demo."""

import time
from pathlib import Path

import numpy as np
import skimage.data
from skimage.filters import gaussian, sobel, threshold_li, threshold_otsu
from skimage.restoration import denoise_nl_means
from skimage.segmentation import slic
from skimage.util import img_as_float

import imaging_server_kit as sk


## Intensity threshold
@sk.algorithm(
    name="Intensity threshold",
    description="Segment an image based on an intensity threshold.",
    tags=["Segmentation", "Demo"],
    parameters={
        "image": sk.Image(),
        "threshold": sk.Float(
            name="Threshold (rel.)",
            description="Intensity threshold, relative to the image min() and max() values.",
            default=0.5,
            min=0,
            max=1,
            step=0.1,
            auto_call=True,
        ),
        "dark_background": sk.Bool(
            name="Dark background", default=True, auto_call=True
        ),
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
def threshold_algo(image, threshold, dark_background):
    thresh_rel = threshold * (image.max() - image.min())
    if dark_background:
        mask = image > thresh_rel
    else:
        mask = image <= thresh_rel
    return sk.Mask(mask, name="Binarized image")


## Automatic threshold
@sk.algorithm(
    name="Automatic threshold",
    description="Implementation of an automatic threshold algorithm.",
    tags=["Segmentation", "Scikit-image", "Demo"],
    parameters={
        "image": sk.Image(),
        "method": sk.Choice(
            name="Method",
            default="Otsu",
            description="Auto-threshold method.",
            items=["Otsu", "Li"],
            auto_call=True,
        ),
    },
    samples=[
        {
            "image": skimage.data.coins(),
        },
        {
            "image": skimage.data.camera(),
        },
    ],
)
def auto_threshold(image, method):
    if method == "Otsu":
        mask = image > threshold_otsu(image)
    elif method == "Li":
        mask = image > threshold_li(image)
    return sk.Mask(mask, name="Binary mask (auto)")


## Gaussian filter
@sk.algorithm(
    name="Gaussian filter",
    description="Apply a Gaussian filter (blur) to an image.",
    project_url="https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian",
    tags=["Filtering", "Scikit-image", "Demo"],
    parameters={
        "image": sk.Image(),
        "sigma": sk.Float(
            name="Sigma",
            min=0,
            max=10,
            step=2,
            default=4,
            auto_call=True,
        ),
        "mode": sk.Choice(
            name="Mode",
            items=["reflect", "constant", "nearest", "mirror", "wrap"],
            auto_call=True,
            default="nearest",
        ),
        "preserve_range": sk.Bool(
            name="Preserve range",
            description="Whether to keep the original range of values.",
            default=False,
        ),
    },
    samples=[
        {
            "image": skimage.data.camera(),
            "sigma": 2.0,
        },
    ],
)
def gaussian_algo(image, sigma, mode, preserve_range):
    filtered = gaussian(image, sigma=sigma, mode=mode, preserve_range=preserve_range)
    return sk.Image(
        filtered,
        name=f"Filtered image (Gaussian)",
        meta={"contrast_limits": [filtered.min(), filtered.max()]},
    )


## Sobel filter
@sk.algorithm(
    name="Sobel filter",
    description="Apply a Sobel filter to an image.",
    project_url="https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.sobel",
    tags=["Filtering", "Scikit-image", "Demo"],
    samples=[
        {
            "image": skimage.data.camera(),
        },
    ],
)
def sobel_algo(image):
    filtered = sobel(image)
    return sk.Image(
        filtered,
        name=f"Edges (Sobel)",
        meta={"contrast_limits": [filtered.min(), filtered.max()]},
    )


## Fibonacci sphere
@sk.algorithm(
    name="Fibonacci sphere",
    description="Evenly distributes points on a sphere using the Fibonacci sphere algorithm.",
    parameters={
        "N": sk.Integer(name="N Points", min=1, default=100, auto_call=True),
        "r": sk.Float(name="Radius", min=1, default=20, step=2, auto_call=True),
        "color": sk.Choice(
            name="Color",
            default="white",
            items=["white", "red", "green", "blue"],
            auto_call=True,
        ),
        "size": sk.Integer(name="Size", default=1, min=1, max=10, auto_call=True),
    },
)
def fibonacci_sphere(N, r, color, size):
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(N, dtype=np.float64)

    z = 1.0 - 2.0 * (i + 0.5) / N
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * i

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    points = np.column_stack((x, y, z)) * r

    return sk.Points(
        points,
        meta={"size": size, "border_color": "transparent", "face_color": color},
    )


## Blob detector
@sk.algorithm(
    name="Blob detector",
    description="Blob detection algorithm implemented with a Laplacian of Gaussian (LoG) filter.",
    project_url="https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log",
    tags=["Scikit-image", "Demo"],
    parameters={
        "image": sk.Image(description="Input image (2D, 3D).", dimensionality=[2, 3]),
        "min_sigma": sk.Integer(
            name="Min sigma",
            description="Minimum standard deviation of the Gaussian kernel, in pixels.",
            default=5,
            min=1,
            max=100,
            step=1,
            auto_call=True,
        ),
        "max_sigma": sk.Integer(
            name="Max sigma",
            description="Maximum standard deviation of the Gaussian kernel, in pixels.",
            default=10,
            min=1,
            max=100,
            step=1,
            auto_call=True,
        ),
        "num_sigma": sk.Integer(
            name="Num sigma",
            description="Number of intermediate sigma values to compute between the min_sigma and max_sigma.",
            default=10,
            min=1,
            max=100,
            step=1,
            auto_call=True,
        ),
        "threshold": sk.Float(
            name="Threshold",
            description="Lower bound for scale space maxima.",
            default=0.1,
            min=0.01,
            max=1.0,
            step=0.01,
            auto_call=True,
        ),
        "invert_image": sk.Bool(
            name="Dark blobs",
            description="Whether to invert the image before computing the LoG filter.",
            default=False,
        ),
        "time_dim": sk.Bool(
            name="Frame by frame",
            description="Only applicable to 3D images. If set, the first dimension is considered time and the LoG is computed independently for every frame.",
            default=True,
        ),
    },
    samples=[
        {"image": Path(__file__).parent / "sample_images" / "blobs.tif"},
        {"image": Path(__file__).parent / "sample_images" / "tracks.tif"},
    ],
)
def blob_detector_algo(
    image: np.ndarray,
    max_sigma: int,
    num_sigma: int,
    threshold: float,
    invert_image: bool,
    time_dim: bool,
    min_sigma: int,
):
    if invert_image:
        image = -image

    image = img_as_float(image)

    if (image.ndim == 3) & time_dim:
        # Handle a time-series
        points = np.empty((0, 3))
        sigmas = []
        for frame_id, frame in enumerate(image):
            frame_results = skimage.feature.blob_log(
                frame,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                threshold=threshold,
            )
            frame_points = frame_results[:, :2]  # Shape (N, 2)
            frame_sigmas = list(frame_results[:, 2])  # Shape (N,)
            sigmas.extend(frame_sigmas)
            frame_points = np.hstack(
                (np.array([frame_id] * len(frame_points))[..., None], frame_points)
            )  # Shape (N, 3)
            points = np.vstack((points, frame_points))
        sigmas = np.array(sigmas)
    else:
        results = skimage.feature.blob_log(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
        )
        points = results[:, :image.ndim]
        sigmas = results[:, image.ndim]

    points_params = {
        "opacity": 0.7,
        "face_color": "sigma",
        "features": {"sigma": sigmas},  # numpy array representing the point size
    }

    n_points = len(points)
    if n_points:
        return (
            sk.Points(points, name="Detections", meta=points_params),
            f"Points detected: {n_points}",
        )
    else:
        return sk.Notification("No points detected.")


## Non-local means denoising
@sk.algorithm(
    name="Non-local Means Denoising",
    description="Non-local means denoising, implementation from Scikit-image.",
    project_url="https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_nl_means",
    tags=["Filtering", "Scikit-image", "Demo"],
    parameters={
        "image": sk.Image(
            name="Image",
            description="Input image to be denoised.",
            dimensionality=[2, 3],
        ),
        "patch_size": sk.Integer(
            name="Patch size",
            description="Size of patches used for denoising.",
            default=7,
            step=2,
            min=3,
            max=30,
        ),
        "patch_distance": sk.Integer(
            name="Patch distance",
            description="Maximal distance in pixels where to search patches used for denoising.",
            default=11,
            step=2,
            min=3,
            max=30,
        ),
        "h": sk.Float(
            name="Gray cut-off",
            description="Cut-off distance (in gray levels), relative to image intensity range. The higher h, the more smooting.",
            default=0.1,
            min=0.0,
            max=1.0,
        ),
        "fast_mode": sk.Bool(
            name="Fast mode",
            description="A fast version of the non-local mean algorithm.",
            default=True,
        ),
        "sigma": sk.Float(
            name="Sigma",
            description="The standard deviation of the (Gaussian) noise.",
            default=0.0,
            min=0.0,
            max=10.0,
        ),
        "preserve_range": sk.Bool(
            name="Preserve range",
            description="Whether to keep the original range of values.",
            default=True,
        ),
    },
    samples=[
        {
            "image": skimage.data.coins(),
        }
    ],
)
def nl_means_denoise(
    image,
    patch_size,
    patch_distance,
    h,
    fast_mode,
    sigma,
    preserve_range,
):
    image_range = image.max() - image.min()
    h = h * image_range

    denoised = denoise_nl_means(
        image=image,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=fast_mode,
        sigma=sigma,
        preserve_range=preserve_range,
    )

    return sk.Image(
        denoised,
        name="Denoised image",
        meta={"contrast_limits": [denoised.min(), denoised.max()]},
    )


## SLIC
@sk.algorithm(
    name="Superpixels (RGB)",
    description="SLIC algorithm (Scikit-image implementation). Segment an image using k-means clustering in Color-(x,y,z) space.",
    project_url="https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic",
    tags=["Segmentation", "Scikit-image", "Demo"],
    samples=[{"image": skimage.data.astronaut()}],
    parameters={
        "image": sk.Image(
            name="Image",
            description="Input RGB image.",
            dimensionality=[2, 3],
            rgb=True,
        ),
        "n_segments": sk.Integer(
            name="N Segments",
            description="The (approximate) number of labels in the segmented output image.",
            default=100,
            min=1,
            max=1000,
            step=5,
        ),
        "compactness": sk.Float(
            name="Compactness",
            description="Balances color proximity and space proximity.",
            default=10.0,
            min=1.0,
            max=100.0,
        ),
    },
)
def slic_algo(
    image,
    n_segments,
    compactness,
):
    partitioned = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
    )

    return sk.Mask(partitioned, name="SLIC result")


## Notifications
@sk.algorithm(
    name="Notifications stream",
    description="Demo of how to send notifications.",
    parameters={
        "time_delay": sk.Integer(name="Time delay (sec)", default=1, min=1, max=5),
        "n_times": sk.Integer(name="Repetitions", default=3, min=1, max=10),
        "level": sk.Choice(
            name="Level", items=["info", "warning", "error"], default="info"
        ),
    },
)
def notif_stream(time_delay, n_times, level):
    for k in range(n_times):
        time.sleep(time_delay)
        yield sk.Notification(f"Step: {k}", meta={"level": level})


## Background subtraction
@sk.algorithm(
    name="Background subtraction",
    parameters={
        "image": sk.Image(dimensionality=[2, 3]),
        "sigma": sk.Float(name="Sigma", min=0, default=30, step=5, auto_call=True),
        "method": sk.Choice(
            name="Method",
            items=["subtract", "divide"],
            default="subtract",
            auto_call=True,
        ),
    },
    samples=[
        {
            "image": skimage.data.coins(),
            "sigma": 10,
        },
    ],
)
def background_subtract(image, sigma, method):
    blurred = gaussian(image, sigma=sigma, preserve_range=True)
    if method == "subtract":
        corrected = image - blurred
    elif method == "divide":
        corrected = image / (blurred + 1e-9)

    return sk.Image(
        corrected,
        name="Background corrected",
        meta={"contrast_limits": [corrected.min(), corrected.max()]},
    )


## Projections (max, min, etc.)
# TODO: this is not going to be compatible with 3D tiling when the Z range exceeds the tile size along Z
# But that's OK for now (a tiled Z projection along Z isn't meaningful...)
@sk.algorithm(
    name="Projections (3D -> 2D)",
    parameters={
        "image": sk.Image(name="3D image", dimensionality=[3], required=True),
        "method": sk.Choice(
            name="Method", items=["max", "min", "mean"], default="max", auto_call=True
        ),
        "axis": sk.Choice(name="Axis", items=["0", "1", "2"], auto_call=True),
    },
    samples=[{"image": skimage.data.brain()}],
)
def project(image, method, axis: str):
    proj_func = {
        "max": np.max,
        "min": np.min,
        "mean": np.mean,
    }
    proj = proj_func[method](image, axis=int(axis))
    return sk.Image(
        proj,
        name=f"Projection",
        meta={"contrast_limits": [proj.min(), proj.max()], "colormap": "viridis"},
    )


## Conway's game of life (inspired by: https://www.geeksforgeeks.org/dsa/conways-game-life-python-implementation/)
@sk.algorithm(
    name="Game of Life",
    description="Conway's game of life.",
    project_url="https://www.geeksforgeeks.org/dsa/conways-game-life-python-implementation/",
)
def conway_algo(max_iter=200, delay=0.1):
    min_val = 0
    max_val = 255
    
    N=100  # Size of the grid

    def update(grid: np.ndarray, N: int) -> np.ndarray:
        newGrid = grid.copy()
        for i in range(N):
            for j in range(N):
                total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] + 
                            grid[(i-1)%N, j] + grid[(i+1)%N, j] + 
                            grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] + 
                            grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N]))
                total = total / max_val
                if grid[i, j] == max_val:
                    if (total < 2) or (total > 3):
                        newGrid[i, j] = min_val
                else:
                    if total == 3:
                        newGrid[i, j] = max_val
        return newGrid

    grid = np.random.choice([min_val, max_val], N*N, p=[0.5, 0.5]).reshape(N, N)
    for k in range(max_iter):
        grid = update(grid, N)
        time.sleep(delay)
        yield sk.Mask(grid), f"Iterations: {k} / {max_iter}"