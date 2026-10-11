---
name: isk-write-algorithm
description: Turn an image processing or image analysis function (segmentation, filtering, detection, thresholding, registration, tracking) into an Imaging Server Kit algorithm with `@sk.algorithm`, including parameter annotations, typed return layers (sk.Image, sk.Mask, sk.Points, sk.Boxes...), samples, metadata, live updates (yield), tiling and region support. Use this whenever the user mentions imaging_server_kit, "import imaging_server_kit as sk", serverkit, `@sk.algorithm`, or wants to wrap a scikit-image / Cellpose / StarDist / NumPy image function so it can run in Napari, QuPath or a web server, even if they do not say "algorithm".
---

# Writing an Imaging Server Kit algorithm

An algorithm is a Python function plus a description of its inputs and outputs. From that description the package generates Napari/QuPath GUIs, documentation pages and a FastAPI server, so the quality of the annotations decides how well all of those work. Spend your effort on annotating parameters and outputs correctly; the image processing logic itself stays a plain function.

```python
import imaging_server_kit as sk
from skimage.filters import gaussian

@sk.algorithm(
    name="Gaussian filter",
    description="Blur an image with a Gaussian kernel.",
    parameters={"sigma": sk.Float(name="Sigma", min=0, default=1.0, step=0.5, auto_call=True)},
)
def gaussian_algo(image, sigma):
    return sk.Image(gaussian(image, sigma=sigma, preserve_range=True), name="Filtered")
```

## Workflow

1. Write the function with plain NumPy / scikit-image logic. Prefer scikit-image over OpenCV, and `pathlib.Path` over `os.path`.
2. Annotate each parameter (next section).
3. Wrap every array output in the layer that says what it is.
4. Add `samples=[...]` with an example image so users can try it in one click.
5. Check it runs: `algo.run(image, sigma=2.0)` returns a `sk.Stack`; `results.read("Filtered")` or `results[0]` gets a layer, `.data` the array.
6. Offer a GUI (`sk.to_napari(algo)`), see the `isk-napari-qupath-gui` skill.

## Annotating parameters

Each parameter is matched with a data layer, using the first rule that applies:

1. `parameters={"name": sk.Float(...)}` in the decorator (explicit, wins over everything).
2. Type hints: `int`, `float`, `bool`, `str`, `np.ndarray` (an image), or `sk.Image` etc.
3. Default values: `sigma=1.0` gives an `sk.Float`; a layer instance such as `sk.Float(default=1.0)` also works.
4. Variable names for unannotated, un-hinted parameters: `image`, `mask`, `points`, `vectors`, `boxes`, `paths`, `tracks`.

Anything unresolved (lists, dicts, custom classes) becomes `sk.Any`: hidden in GUIs and often not servable. Avoid it for parameters users should control.

Use explicit `parameters={}` whenever a constraint matters. Common options: `name` (GUI label), `description` (tooltip and docs), `min`, `max`, `step`, `default`, `auto_call=True` (re-run when the value changes; good for cheap, interactive parameters, bad for slow models). Layer types: `sk.Float`, `sk.Integer`, `sk.Bool`, `sk.String`, `sk.Choice(items=[...], default=...)`, and the data layers `sk.Image`, `sk.Mask`, `sk.Points`, `sk.Boxes`, `sk.Vectors`, `sk.Paths`, `sk.Tracks`.

Unconstrained numeric parameters are the usual source of bugs: a negative `sigma` or an even kernel size crashes users in the GUI. Set `min`/`max`, or use `sk.Choice`, instead of validating inside the function.

## Returning outputs

Return (or `yield`) data layers. A bare NumPy array becomes an `sk.Image`, so a segmentation returned unwrapped is displayed as a grayscale image, which is the most common mistake.

| The output is | Return |
|---|---|
| a filtered/intermediate image | `sk.Image(arr, name=..., colormap="viridis")` |
| a binary or label mask | `sk.Mask(arr)` (a Labels layer in Napari, polygons in QuPath) |
| detected points | `sk.Points(coords)` |
| bounding boxes | `sk.Boxes(boxes)` |
| a number, string, flag | `sk.Float`/`sk.Integer`/`sk.String`/`sk.Bool`, or the plain Python value |
| a message to the user | `sk.Notification("...", level="info" \| "warning" \| "error")` |

Return a tuple for several outputs. Give each a distinct `name`, since layers are matched by name: same name updates the layer, a new name adds one. Extra keyword arguments (`colormap=`, ...) are forwarded as properties of the viewer layer.

## Samples and metadata

```python
@sk.algorithm(
    samples=[{"image": skimage.data.coins(), "threshold": 100}],
    tags=["Segmentation"],
)
def threshold_algo(image, threshold=128):
    """Intensity threshold."""   # used as description when none is passed
    return sk.Mask(image > threshold)
```

A sample image can be an array, a URL, or a `Path`. `algo.info()` opens the generated documentation page.

## Optional behaviors

- **Live updates**: `yield sk.Mask(mask), sk.String("...")` for intermediate results, `yield sk.Progress(k, max_val=n)` for a progress bar, and a final `return`. Use for iterative or slow algorithms so the GUI is not frozen-looking.
- **Tiling**: add `tileable=True` for filters, segmentation and detection that work locally on a tile. Then `algo.run(image, tiled=True, tile_size=64, tile_overlap=0.1)`. Without the flag, tiled runs raise `AlgorithmRuntimeError`. For instance segmentation, return `sk.Mask(labels, merger="instances")` and use a non-zero overlap so objects crossing tile borders are stitched. Do not mark algorithms that need global image statistics (e.g. a global Otsu threshold) as tileable without telling the user the result will differ.
- **Regions**: `algo.run(image, domain=sk.Domain(position=(20, 30), size=(60, 80)))` computes only inside the region and positions results in global pixel coordinates. Nothing to add in the function itself.
- **Combining**: `sk.combine([algo_a, algo_b])` groups algorithms into one collection (one dropdown in the GUI).
- **Serving**: `sk.serve(algo)` starts a FastAPI server; `sk.Client("http://localhost:8000").run(image, ...)` calls it with the same interface as a local algorithm. Parameters and outputs must be serializable, which is another reason to avoid `sk.Any`.

## Pitfalls

- Heavy model loading inside the function runs on every call. Load once at module level or lazily cache it.
- 2D vs nD: do not assume 2D unless the algorithm really is 2D; `sk.Image(dimensionality=[2])` restricts it explicitly.
- Keep new runtime dependencies out of the algorithm unless the user already uses them; check whether scikit-image / NumPy already provide the operation.
- Run the algorithm once on a sample before declaring it done, and report the actual result rather than assuming it works.
