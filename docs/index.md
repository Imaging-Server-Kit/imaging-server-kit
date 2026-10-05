# Imaging Server Kit

The **Imaging Server Kit** turns Python image processing functions into **algorithms**: objects that you can run in Napari or QuPath, serve over HTTP, run tile-by-tile, and more.

```python
import imaging_server_kit as sk

@sk.algorithm  # <- Turn your function into an algorithm
def my_algo(image, parameter):
    ...
```

<div class="grid cards" markdown>

-   :material-server-network: **Serve algorithms over HTTP**

    ---

    Turn an algorithm into a web server, then run it from [Napari](https://napari.org/stable/), [QuPath](https://qupath.github.io/), or [Python](tutorial/python.md).

    <video controls loop autoplay muted playsinline>
      <source src="assets/videos/cellpose_example.mp4" type="video/mp4">
    </video>

    [:octicons-arrow-right-24: Serving algorithms](tutorial/serve.md)

-   :material-dock-left: **Generate dock widgets**

    ---

    Run your algorithm interactively in Napari or QuPath with a parameters panel.

    <video controls loop autoplay muted playsinline>
      <source src="assets/videos/oripy_threshold.mp4" type="video/mp4">
    </video>

    [:octicons-arrow-right-24: Create an algorithm](tutorial/create-algorithm.md)

-   :material-grid: **Run tile-by-tile**

    ---

    Process images in tiles and progressively assemble the results.

    <video controls loop autoplay muted playsinline>
      <source src="assets/videos/tiles.mp4" type="video/mp4">
    </video>

    [:octicons-arrow-right-24: Tiled inference](how-to/tiling.md)

-   :material-play-speed: **Stream results**

    ---

    Send results while the algorithm runs and inspect them in real time.

    <video controls loop autoplay muted playsinline>
      <source src="assets/videos/yolo-stream.mp4" type="video/mp4">
    </video>

    [:octicons-arrow-right-24: Live updates](how-to/live-updates.md)

</div>

On top of that, you can provide [**samples**](tutorial/samples-and-metadata.md#samples) and automatically generate a [**documentation page**](tutorial/samples-and-metadata.md#metadata) for your algorithm.

## Next steps

- New to the package? [Install it](getting-started/installation.md) and [try the demos](getting-started/demos.md).
- Follow the main tutorial, starting with [creating an algorithm](tutorial/create-algorithm.md).
- Browse the how-to guides, for example [using algorithms in Napari](how-to/napari.md).
- See the [Python API reference](reference/api/algorithms.md).

!!! warning "Development status"
    The Imaging Server Kit is being actively developed and is iterating rapidly. Expect **compatibility-breaking changes** in future versions.

## License

This software is distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license.

## Acknowledgements

We thank the [Personalized Health and Related Technologies](https://www.sfa-phrt.ch/) for initially funding this project.
