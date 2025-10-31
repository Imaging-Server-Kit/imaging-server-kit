import imaging_server_kit as sk
from skimage.filters import gaussian
import skimage.data
import numpy as np


### Tiled image filtering
@sk.algorithm(
    parameters={
        "sigma": sk.Float(
            name="Sigma", min=0, max=10, step=1, default=5, auto_call=True
        ),
        "mode": sk.Choice(
            name="Mode",
            items=["reflect", "constant", "nearest", "mirror", "wrap"],
            auto_call=True,
            default="reflect",
        ),
    },
    samples=[{"image": skimage.data.camera()}],
)
def sk_gaussian(image, sigma, mode):
    return gaussian(image, sigma=sigma, preserve_range=True, mode=mode)


def test_sk_gaussian():
    sample = sk_gaussian.get_sample(idx=0)
    image = sample[0].data

    results = sk_gaussian.run(
        image=image, tiled=True, tile_size_px=256, overlap_percent=0.1, randomize=True
    )

    blurred = results[0].data

    assert blurred.shape == image.shape


### Tiled segmentation of RGB
from skimage.segmentation import slic

@sk.algorithm(
    parameters={"image": sk.Image(rgb=True)},
    samples=[{"image": skimage.data.astronaut()}],
)
def sk_tiled_rgb(image):
    mask = slic(image)
    return sk.Mask(mask)


def test_sk_tiled_rgb():
    sample = sk_tiled_rgb.get_sample().to_params_dict()
    image = sample.get("image")
    results = sk_tiled_rgb.run(**sample, tiled=True)
    mask = results[0].data
    assert mask.shape == tuple([image.shape[0], image.shape[1]])


### Tiled points
@sk.algorithm
def sk_tiled_points(image):
    rx, ry = image.shape
    points = np.array([np.linspace(1, rx-1, 10), np.linspace(1, ry-1, 10)]).T
    return sk.Points(points)

@sk.algorithm
def sk_tiled_points_input(image, points):
    new_points = points.copy()
    new_points = new_points[:, ::-1]
    return sk.Points(new_points)

def test_sk_tiled_points():
    image = np.random.random((50, 50))
    results = sk_tiled_points.run(image, tiled=True, tile_size_px=25)
    points = results.read("Points").data
    assert len(points) == 40
    
    results = sk_tiled_points_input.run(image, points, tiled=True, tile_size_px=25)
    new_points = results.read("Points").data
    assert np.sum(new_points[:, 0]) - np.sum(points[:, 1]) < 1e-6


### Tiled vectors
@sk.algorithm
def sk_tiled_vectors(image):
    rx, ry = image.shape
    vector_origins = np.array([np.linspace(1, rx-1, 10), np.linspace(1, ry-1, 10)]).T
    vector_ends = np.random.random((len(vector_origins), 2))
    vectors = np.hstack((vector_origins[:, np.newaxis, :], vector_ends[:, np.newaxis, :]))
    return sk.Vectors(vectors)

@sk.algorithm
def sk_tiled_vectors_input(image, vectors):
    new_vectors = vectors.copy()
    new_vectors = new_vectors[:, :, ::-1]
    return sk.Vectors(new_vectors)

def test_sk_tiled_vectors():
    image = np.random.random((50, 50))
    results = sk_tiled_vectors.run(image, tiled=True, tile_size_px=25)
    vectors = results.read("Vectors").data
    assert len(vectors) == 40
    
    results = sk_tiled_vectors_input.run(image, vectors, tiled=True, tile_size_px=25)
    new_points = results.read("Vectors").data
    assert np.sum(new_points[:, :, 0]) - np.sum(vectors[:, :, 1]) < 1e-6


### Tiled boxes
@sk.algorithm
def sk_tiled_boxes(image):
    rx, ry = image.shape
    box_top_left = np.array([np.linspace(1, rx-10, 10), np.linspace(1, ry-10, 10)]).T
    box_widths = np.random.random(len(box_top_left)) * 5 + 3
    box_length = np.random.random(len(box_top_left)) * 5 + 3
    
    box_top_right = box_top_left.copy()
    box_bot_left = box_top_left.copy()
    box_bot_right = box_top_left.copy()
    
    box_top_right[:, 0] += box_widths
    box_bot_left[:, 1] += box_length
    
    box_bot_right[:, 0] += box_widths
    box_bot_right[:, 1] += box_length
    
    boxes = np.stack((box_top_right, box_top_left, box_bot_left, box_bot_right), axis=1)
    return sk.Boxes(boxes)

@sk.algorithm
def sk_tiled_boxes_input(image, boxes):
    new_boxes = boxes.copy()
    new_boxes = new_boxes[:, :, ::-1]
    return sk.Boxes(new_boxes)

def test_sk_tiled_boxes():
    image = np.random.random((50, 50))
    results = sk_tiled_boxes.run(image, tiled=True, tile_size_px=25)
    boxes = results.read("Boxes").data
    assert len(boxes) == 40
    
    results = sk_tiled_boxes_input.run(image, boxes, tiled=True, tile_size_px=25)
    new_boxes = results.read("Boxes").data
    assert isinstance(new_boxes, np.ndarray)