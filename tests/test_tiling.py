import imaging_server_kit as sk
from skimage.filters import gaussian
import skimage.data


@sk.algorithm(
    parameters={
        "sigma": sk.Float(
            name="Sigma", min=0, max=10, step=1, default=5, auto_call=True
        ),
        "mode": sk.DropDown(
            name="Mode",
            items=["reflect", "constant", "nearest", "mirror", "wrap"],
            auto_call=True,
            default="reflect",
        ),
    },
    sample_images=[skimage.data.camera()],
)
def sk_gaussian(image, sigma, mode):
    return gaussian(image, sigma=sigma, preserve_range=True, mode=mode)

def test_sk_gaussian():
    image = sk_gaussian.get_sample_images(first_only=True)

    results = sk_gaussian.run(
        image=image, tiled=True, tile_size_px=256, overlap_percent=0.1, randomize=True
    )

    blurred = results[0].data

    assert blurred.shape == image.shape
