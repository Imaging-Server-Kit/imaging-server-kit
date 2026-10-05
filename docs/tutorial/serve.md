# 3. Serve an algorithm

Any algorithm can be **served as a web API** via a built-in [FastAPI](https://fastapi.tiangolo.com/) server. Once served, the algorithm can be used from Napari, QuPath, or Python through HTTP requests, and from the same or another machine.

## Serving an algorithm

Pass the algorithm to `sk.serve()`. Save the following code as a Python script, for example `threshold_server.py`:

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

if __name__ == "__main__":
    sk.serve(threshold_algo)  # <- Serve the algorithm
```

Then run it from a terminal:

```sh
python threshold_server.py
```

`sk.serve()` starts a FastAPI server that exposes your algorithm through a set of predefined routes (see [HTTP endpoints](../reference/http.md)).

![Server running in a terminal](../assets/images/server_running.png)

By default, the server listens on port `8000` on all network interfaces. Open http://localhost:8000 in a browser to see the algorithm's [documentation page](samples-and-metadata.md#metadata). FastAPI also generates an interactive page at http://localhost:8000/docs that documents every route.

## Connecting from Napari

In Napari, open `Plugins > Imaging Server Kit > Connect to server`. Under *Server URL*, enter http://localhost:8000 and press *Connect*. Your threshold algorithm appears in the *Algorithm* dropdown, with the same features as in the local case: loading samples, running the algorithm, and opening its documentation.

Algorithm servers can also be used from QuPath (see [Usage with QuPath](../how-to/qupath.md)), and from Python, which will be the topic of the next step of this tutorial.

## Summary

- Use `sk.serve()` in a Python script to expose an algorithm as a web service.
- By default, the server is reachable at http://localhost:8000.
- Connect to algorithm servers from Napari with `Plugins > Imaging Server Kit > Connect to server`.

To make a server reachable from other machines, see [Serve an algorithm](../how-to/deploy-server.md).

## Next steps

Next, we will cover how to [use algorithms from Python](python.md), locally and through a server.
