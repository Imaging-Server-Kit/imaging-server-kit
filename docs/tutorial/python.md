# 4. Usage from Python

Algorithms can be used directly from Python to run computations, load samples, and more. A key idea is that the same code works for a local algorithm and for an algorithm running on a server.

## Calling an algorithm

Let's start from the threshold algorithm of the previous steps:

```python
import imaging_server_kit as sk
import skimage.data

@sk.algorithm(
    name="Intensity threshold",
    parameters={"threshold": sk.Integer(name="Threshold", min=0, max=255, default=128)},
    samples=[{"image": skimage.data.coins()}],
)
def threshold_algo(image, threshold):
    mask = image > threshold
    return sk.Mask(mask, name="Binary mask")
```

The algorithm still behaves like the original Python function:

```python
image = skimage.data.coins()

mask = threshold_algo(image, threshold=100)  # A NumPy array
```

One difference is that **parameters are validated** when the algorithm runs. Since `threshold` was annotated with `min=0`, a negative value is rejected:

```python
threshold_algo(image, threshold=-1)  # Raises a ValidationError
```

The error message explains that the threshold should be greater than or equal to zero.

## Running algorithms with `.run()`

Algorithms also have a `.run()` method:

```python
results = threshold_algo.run(image, threshold=100)

print(results)
# Stack | 1 layers | extent [0:303, 0:384]
#   #  kind  name         data
#   0  mask  Binary mask  bool (303, 384), 1 labels
```

Instead of the raw return values, `.run()` returns a `Stack`: an ordered collection of [data layers](../concepts/layers-and-stacks.md). Here, the stack holds one layer containing the segmentation mask.

You can access layers in a stack by name with `.read()`, and also by index:

```python
mask_result = results.read("Binary mask")  # <- Same as `results[0]`

print(mask_result)  # <Mask 'Binary mask' bool (303, 384), 1 labels>
```

`mask_result` is a `sk.Mask` object. The segmentation mask itself is in its `data` attribute:

```python
mask = mask_result.data  # NumPy array

print(mask.shape)  # (303, 384)
```

All data layers have a `data`, a `name`, and a `meta` attribute. See [Layers and stacks](../concepts/layers-and-stacks.md) and the [API reference](../reference/api/layers.md) for more details.

### Going further with `.run()`

- `run(..., stack=my_stack)` adds the results to an existing `Stack`.
- `run(..., stack=viewer)` sends the results straight to a Napari viewer.
- `run(..., tiled=True)` runs the algorithm tile-by-tile (see [Run algorithms tile-by-tile](../how-to/tiling.md)).
- `run(..., domain=roi)` restricts the computation to a region (see [Restrict computation to a region](../how-to/regions.md)).
- `sk.run(algo, ...)` is equivalent to `algo.run(...)`.

## Samples and docs

You can retrieve a sample with `.get_sample()` by passing the index of the sample. Samples are returned as stacks of parameter layers:

```python
sample = threshold_algo.get_sample(idx=0)

print(sample)
# Stack | 2 layers | extent [0:303, 0:384]
#   #  kind   name       data
#   0  image  image      uint8 (303, 384)
#   1  int    threshold  = 128
```

`.info()` opens the algorithm's documentation page in a web browser:

```python
threshold_algo.info()
```

## Connecting to a server with `sk.Client`

A great advantage of `.run()` is that it works the same way on a local algorithm and on a **client connected to an algorithm server**.

To try it, serve the threshold algorithm as in the [previous step](serve.md), so that it is available at http://localhost:8000. Then connect to it from Python with `sk.Client`:

```python
import imaging_server_kit as sk
import skimage.data

image = skimage.data.coins()

# Connect to the server
client = sk.Client("http://localhost:8000")

# The image and parameters are sent to the server, which runs the computation and returns the results
results = client.run(image, threshold=50)

# Segmentation mask
mask = results[0].data
```

Clients have the same methods as algorithms, including `.get_sample()` and `.info()`:

```python
sample = client.get_sample(idx=0)  # <- Retrieves the first sample from the server

client.info()  # <- Opens the documentation page
```

## Summary

- Algorithms can still be called like the original function (but parameters are validated).
- `.run()` returns a `Stack` of data layers; access layers by name with `.read()` or by index.
- Data layer have a `data` attribute to store their data (values or NumPy arrays).
- Use `.get_sample()` to retrieve samples, and `.info()` to open the documentation.
- `sk.Client` connects to a server and offers the same methods as a local algorithm.

## Next steps

Well-done! You have completed the tutorial 🚀. From here, you can learn how to:

- [Combine algorithms](../how-to/combine.md) into a single collection.
- [Stream live updates](../how-to/live-updates.md) while an algorithm runs.
- [Run algorithms tile-by-tile](../how-to/tiling.md) on large images.
- Read more about the [concepts](../concepts/algorithms.md) behind algorithms, layers, and stacks.
