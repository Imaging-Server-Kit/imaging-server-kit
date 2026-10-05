# HTTP endpoints

Algorithm servers started with [`sk.serve()`](../tutorial/serve.md) expose the routes below. In most cases, you don't need to call them directly: [`sk.Client`](../tutorial/python.md#connecting-to-a-server-with-skclient) and the Napari and QuPath widgets use them for you.

The server also serves FastAPI's interactive documentation at `/docs`, with the request and response schemas of every route.

## Server routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Home page with algorithm(s) documentation. |
| `GET` | `/algorithms` | The list of available algorithms, as `{"algorithms": [...]}`. |
| `GET` | `/version` | The version of `imaging-server-kit` running on the server. |

## Algorithm routes

In these routes, `{algorithm_name}` is the name of an algorithm, as listed by `/algorithms`.

| Method | Route | Description |
|---|---|---|
| `GET` | `/{algorithm_name}/info` | The HTML documentation page of the algorithm. |
| `GET` | `/{algorithm_name}/parameters` | The JSON schema of the algorithm parameters. |
| `GET` | `/{algorithm_name}/signature` | The parameter names of the algorithm's function, in order. |
| `GET` | `/{algorithm_name}/n_samples` | The number of samples, as `{"n_samples": n}`. |
| `GET` | `/{algorithm_name}/sample/{idx}` | The sample at index `idx`, encoded as a stack of parameter layers. |
| `GET` | `/{algorithm_name}/tileable` | Whether the algorithm can be run tile-by-tile, as `{"tileable": true}` or `false`. |
| `POST` | `/{algorithm_name}/process` | Run the algorithm. The request body holds the encoded parameters. Results are streamed back as [MessagePack](https://msgpack.org/). |
