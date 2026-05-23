import imaging_server_kit as sk
from skimage.filters import gaussian
from skimage.color import gray2rgb, rgb2gray
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

    stack = sk_gaussian.run(
        image=image, tiled=True, tile_size=256, tile_overlap=0.1, tile_randomize=True
    )

    blurred = stack[0].data

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
    sample = sk_tiled_rgb.get_sample()
    image = sample.read("image").data
    sample_params = {l.name: l.data for l in sample.layers}
    stack = sk_tiled_rgb.run(**sample_params, tiled=True)
    mask = stack[0].data
    assert mask.shape == tuple([image.shape[0], image.shape[1]])


# Convert gray to RGB
@sk.algorithm(parameters={"image": sk.Image(rgb=False)})
def to_rgb(image: sk.Image):
    return sk.Image(gray2rgb(image), rgb=True)


def test_convert_to_rgb():
    image = skimage.data.coins()
    rgb_out = to_rgb.run(image=image, tiled=True, tile_overlap=0.1)
    assert len(rgb_out[0].shape) == 3
    assert rgb_out[0].shape == (image.shape[0], image.shape[1], 3)


# Convert RGB to gray
@sk.algorithm(parameters={"image": sk.Image(rgb=True)})
def to_gray(image: sk.Image):
    return sk.Image(rgb2gray(image))


def test_convert_to_gray():
    image = skimage.data.astronaut()
    gray_out = to_gray.run(image=image, tiled=True, tile_overlap=0.1)
    assert len(gray_out[0].shape) == 2
    assert gray_out[0].shape == (image.shape[0], image.shape[1])


### Tiled points
@sk.algorithm
def sk_tiled_points(image):
    rx, ry = image.shape
    points = np.array([np.linspace(1, rx - 1, 10), np.linspace(1, ry - 1, 10)]).T
    return sk.Points(points)


@sk.algorithm
def sk_tiled_points_input(image, points):
    new_points = points.copy()
    new_points = new_points[:, ::-1]
    return sk.Points(new_points)


def test_sk_tiled_points():
    image = np.random.random((50, 50))
    stack = sk_tiled_points.run(image, tiled=True, tile_size=25)
    points = stack.read("Points").data
    assert len(points) == 40

    stack = sk_tiled_points_input.run(image, points, tiled=True, tile_size=25)
    new_points = stack.read("Points").data
    assert np.sum(new_points[:, 0]) - np.sum(points[:, 1]) < 1e-6


### Tiled vectors
@sk.algorithm
def sk_tiled_vectors(image):
    rx, ry = image.shape
    vector_origins = np.array(
        [np.linspace(1, rx - 1, 10), np.linspace(1, ry - 1, 10)]
    ).T
    vector_ends = np.random.random((len(vector_origins), 2))
    vectors = np.hstack(
        (vector_origins[:, np.newaxis, :], vector_ends[:, np.newaxis, :])
    )
    return sk.Vectors(vectors)


@sk.algorithm
def sk_tiled_vectors_input(image, vectors):
    new_vectors = vectors.copy()
    new_vectors = new_vectors[:, :, ::-1]
    return sk.Vectors(new_vectors)


def test_sk_tiled_vectors():
    image = np.random.random((50, 50))
    stack = sk_tiled_vectors.run(image, tiled=True, tile_size=25)
    vectors = stack.read("Vectors").data
    assert len(vectors) == 40

    stack = sk_tiled_vectors_input.run(image, vectors, tiled=True, tile_size=25)
    new_vectors = stack.read("Vectors").data
    assert np.sum(new_vectors) - np.sum(vectors) < 1e-6


### Tiled boxes
@sk.algorithm
def sk_tiled_boxes(image):
    rx, ry = image.shape
    box_top_left = np.array(
        [np.linspace(1, rx - 10, 10), np.linspace(1, ry - 10, 10)]
    ).T
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
    stack = sk_tiled_boxes.run(image, tiled=True, tile_size=25)
    boxes = stack.read("Boxes").data
    assert len(boxes) == 40

    stack = sk_tiled_boxes_input.run(image, boxes, tiled=True, tile_size=25)
    new_boxes = stack.read("Boxes").data
    assert isinstance(new_boxes, np.ndarray)


### Tiled streaming
@sk.algorithm
def sk_streamed_tiling(image):
    for _ in range(3):
        yield sk.Image(np.random.random(image.shape), name="Random")
    return sk.String("Success", name="Success")


def test_sk_tiled_streaming():
    image = np.random.random((30, 30))
    stack = sk_streamed_tiling.run(image=image, tiled=True, tile_size=10)
    assert stack.read("Random").data.shape == image.shape
    assert stack.read("Success").data == "Success"


### Tiled max projection
from imaging_server_kit.demo import project


def test_sk_tiled_max_proj():
    image = np.random.random((10, 30, 30))
    stack = project.run(
        image=image,
        method="max",
        tiled=True,
        tile_size=[10, 15, 15],  # 3D tiles
    )
    data = stack.read("Projection").data
    max_proj = np.max(image, axis=0, keepdims=True)
    assert data.shape == max_proj.shape
    assert np.allclose(data, max_proj)


### Tiled instance mask
from skimage.morphology import label


@sk.algorithm
def label_algo(mask):
    labelled = label(mask)
    return sk.Mask(labelled, merger="instances")


def test_sk_label():
    mask = skimage.data.coins() > 100
    stack = label_algo.run(mask=mask, tiled=True, tile_overlap=0.1)
    data = stack[0].data
    labelled = label(mask)
    assert len(np.unique(labelled)) == len(np.unique(data))


### -- Selecting data -- ###


# In an Image, via indexing and select()
def test_select_indexing_image():
    data = np.random.random((20, 20, 20))
    image = sk.Image(data)

    extract1 = data[2:, 3:7]
    image_extract1 = image[2:, 3:7].data
    assert np.allclose(extract1, image_extract1)

    extract2 = data[:, :, :4]
    image_extract2 = image[:, :, :4].data
    assert np.allclose(extract2, image_extract2)


# In a Stack, via indexing and select()
def test_select_indexing_stack():
    stack = sk.Stack(
        layers=[
            sk.Image(np.random.random((10, 10))),
            sk.Mask(np.ones((10, 10))),
        ]
    )

    extract = stack[:, :3, 1:5]
    assert len(extract) == 2
    assert extract[0].data.shape[0] == 3
    assert extract[0].data.shape[1] == 4
    assert extract[1].data.shape[0] == 3
    assert extract[1].data.shape[1] == 4


### -- Merging data -- ###


### Merging images
def test_merge_images():
    img1 = sk.Image(np.ones((20, 20)))
    _sum1 = img1.data.sum()
    img2 = sk.Image(np.ones((20, 20)), position=(30, 30))
    _sum2 = img2.data.sum()

    merger = sk.LayerMerger()
    merger.merge(img1, img2)

    assert img1.shape == (50, 50)
    assert img1.data.sum() == _sum1 + _sum2


### Merging Stacks (and getting tiles) - heterogeneous data
def test_merge_stack():
    rx, ry = 30, 20

    image1 = np.ones((rx, ry))
    image2 = np.ones((rx, ry))

    mask1 = np.ones((rx, ry), dtype=np.uint8)
    mask2 = np.ones((rx, ry), dtype=np.uint8)

    points1 = np.random.random((10, 2))
    points1[:, 0] = points1[:, 0] * rx
    points1[:, 1] = points1[:, 1] * ry
    points2 = points1.copy()

    delta = np.random.random(points1.shape)
    vectors1 = np.zeros((len(points1), 2, points1.shape[1]))
    vectors1[:, 0] = points1
    vectors1[:, 1] = delta
    vectors2 = vectors1.copy()

    boxes1 = np.zeros((len(points1), 4, points1.shape[1]))
    height = np.random.random(len(points1))
    width = np.random.random(len(points1))
    boxes1[:, 0, 0] = points1[:, 0] - height
    boxes1[:, 0, 1] = points1[:, 1] - width
    boxes1[:, 1, 0] = points1[:, 0] - height
    boxes1[:, 1, 1] = points1[:, 1] + width
    boxes1[:, 3, 0] = points1[:, 0] + height
    boxes1[:, 3, 1] = points1[:, 1] - width
    boxes1[:, 2, 0] = points1[:, 0] + height
    boxes1[:, 2, 1] = points1[:, 1] + width
    boxes2 = boxes1.copy()

    img1 = sk.Image(image1)
    msk1 = sk.Mask(mask1)
    pts1 = sk.Points(points1)
    vct1 = sk.Vectors(vectors1)
    box1 = sk.Boxes(boxes1)

    position = (rx, ry)

    img2 = sk.Image(image2, position=position)
    msk2 = sk.Mask(mask2, position=position)
    pts2 = sk.Points(points2, position=position)
    vct2 = sk.Vectors(vectors2, position=position)
    box2 = sk.Boxes(boxes2, position=position)

    stack1 = sk.Stack([img1, msk1, pts1, vct1, box1])
    stack2 = sk.Stack([img2, msk2, pts2, vct2, box2])

    # Merge stacks
    stack1.merge(stack2)

    out_img1 = stack1.read(img1.name)
    out_msk1 = stack1.read(msk1.name)
    out_pts1 = stack1.read(pts1.name)
    out_vct1 = stack1.read(vct1.name)
    out_box1 = stack1.read(box1.name)

    domain1 = sk.Domain(size=(rx, ry), position=(0, 0))
    domain2 = sk.Domain(size=(rx, ry), position=(rx, ry))

    assert np.allclose(out_img1.data[:rx, :ry], image1)
    assert np.allclose(out_img1.data[rx:, ry:], image2)
    assert out_img1.data.sum() == image1.sum() + image2.sum()
    assert np.allclose(out_img1.select(domain1).data, image1)
    assert np.allclose(out_img1.select(domain2).data, image2)
    assert np.allclose(out_msk1.data[:rx, :ry], mask1)
    assert np.allclose(out_msk1.data[rx:, ry:], mask2)
    assert out_msk1.data.sum() == mask1.sum() + mask2.sum()
    assert np.allclose(out_msk1.select(domain1).data, mask1)
    assert np.allclose(out_msk1.select(domain2).data, mask2)

    assert len(out_pts1.data) == len(points1) + len(points2)
    assert np.allclose(out_pts1.select(domain1).data, points1)
    assert np.allclose(out_pts1.select(domain2).data, points2)

    assert len(out_vct1.data) == len(vectors1) + len(vectors2)
    assert np.allclose(out_vct1.select(domain1).data, vectors1)
    assert np.allclose(out_vct1.select(domain2).data, vectors2)

    assert len(out_box1.data) == len(boxes1) + len(boxes2)

    res1 = stack1.select(domain1)
    assert np.allclose(res1.read(img1.name).data, image1)
    assert np.allclose(res1.read(msk1.name).data, mask1)
    assert np.allclose(res1.read(pts1.name).data, points1)
    assert np.allclose(res1.read(vct1.name).data, vectors1)

    res2 = stack1.select(domain2)
    assert np.allclose(res2.read(img1.name).data, image2)
    assert np.allclose(res2.read(msk1.name).data, mask2)
    assert np.allclose(res2.read(pts1.name).data, points2)
    assert np.allclose(res2.read(vct1.name).data, vectors2)
