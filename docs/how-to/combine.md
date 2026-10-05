# Combine algorithms

Combining algorithms into a **collection** gives users access to several algorithms from a single interface, such as the *Algorithm* dropdown in Napari or QuPath, or a single server.

## Creating a collection with `sk.combine`

Consider two separately defined algorithms:

```python
import imaging_server_kit as sk

@sk.algorithm(
    name="threshold",
    description="Segment a grayscale image based on an intensity threshold.",
)
def threshold_algo(image, threshold=128):
    mask = image > threshold
    return sk.Mask(mask, name="Binary mask")

@sk.algorithm(
    name="foreground",
    description="Compute the fraction of positive pixels in a binary mask.",
)
def foreground_fract(mask):
    fract = mask.sum() / mask.size
    return sk.Float(fract, name="Foreground fraction")
```

Combine them into a collection with `sk.combine`:

```python
multi_algo = sk.combine([threshold_algo, foreground_fract], name="segmentation-pipeline")

sk.to_napari(multi_algo)
```

Both algorithms are now available from the *Algorithm* dropdown in Napari.

## Using a collection from Python

Algorithm collections have the same methods as standalone algorithms. Use the `algorithm` argument to select an algorithm by name:

```python
import skimage.data

image = skimage.data.coins()

results = multi_algo.run(algorithm="threshold", image=image, threshold=100)
```

The list of algorithm names is available as `multi_algo.algorithms`. If `algorithm` is omitted, the first algorithm of the collection is used.

## Serving a collection

Collections can be served like a single algorithm. On the **server side**:

```python
import imaging_server_kit as sk
import skimage.data
from skimage.filters import threshold_otsu, threshold_li

@sk.algorithm(
    name="intensity-threshold",
    parameters={"threshold": sk.Integer(name="Threshold", min=0, max=255, default=128)},
    samples=[{"image": skimage.data.coins()}],
)
def threshold_algo(image, threshold):
    mask = image > threshold
    return sk.Mask(mask, name="Binary mask")

@sk.algorithm(
    name="automatic-threshold",
    parameters={"method": sk.Choice(name="Method", items=["otsu", "li"], default="otsu")},
    samples=[{"image": skimage.data.coins()}],
)
def auto_threshold(image, method):
    if method == "otsu":
        mask = image > threshold_otsu(image)
    elif method == "li":
        mask = image > threshold_li(image)
    return sk.Mask(mask, name="Binary mask")

threshold_algos = sk.combine([threshold_algo, auto_threshold], name="threshold-algos")

if __name__ == "__main__":
    sk.serve(threshold_algos)
```

Then, on the **client side**:

```python
import imaging_server_kit as sk
import skimage.data

image = skimage.data.coins()

client = sk.Client("http://localhost:8000")

print(client.algorithms)  # ['intensity-threshold', 'automatic-threshold']

thresh_results = client.run(algorithm="intensity-threshold", image=image, threshold=30)
auto_results = client.run(algorithm="automatic-threshold", image=image, method="otsu")
```

## Built-in algorithms

The Imaging Server Kit comes with a collection of common algorithms, `sk.tools`. It includes filters (Gaussian, median, Sobel, etc.), mask utilities (remove small objects, label, fill holes, etc.), math operations, thresholds (Otsu, manual), and more.

Open them in Napari from `Plugins > Imaging Server Kit > Tool algorithms`, from the command line with `serverkit tools napari` (or serve them with `serverkit tools serve`), or from Python:

```python
import imaging_server_kit as sk

sk.to_napari(sk.tools)
```

Like any collection, `sk.tools` can be combined with your own algorithms, for example `sk.combine([threshold_algo, *sk.tools.algorithms_dict.values()])`.
