# Algorithms

## Algorithms

An **algorithm** (`sk.Algorithm`) wraps a Python function together with a description of its inputs and outputs. It is usually created with the `@sk.algorithm` decorator.

From the function signature and the annotations, the algorithm knows:

- Which [data layer](layers-and-stacks.md) corresponds to each parameter the function takes.
- How to interpret the **outputs** of the function as data layers.
- Extra information (samples, name, description, compatibility with tiling, etc.).

This representation is what allows the package to generate user interfaces, documentation pages, and web APIs for the algorithm without extra code.

### Matching parameters with data layers

Each parameter of the function is matched with a data layer. The layer is chosen from the first of these sources that applies:

1. **Explicit annotations** in `parameters={}`, for example `parameters={"sigma": sk.Float(min=0)}`.
2. **Type hints**: `int`, `float`, `bool`, `str`, data layer classes such as `sk.Image`, or `np.ndarray` (interpreted as an image).
3. **Default values**: a data layer instance (for example `sigma=sk.Float(default=1.0)`), or a value whose type is used as in type hints (for example `sigma=1.0` gives an `sk.Float`).
4. **Variable names**, for parameters without a type hint or default value: `image`, `mask`, `points`, `vectors`, `boxes`, `paths`, and `tracks` are matched with the corresponding data layer.

Parameters that cannot be resolved, for example custom classes, lists, or dictionaries, become `sk.Any` layers. They are not shown in user interfaces, and they may not work with servers if they cannot be serialized.

### Matching outputs with data layers

The values returned (or [yielded](../how-to/live-updates.md)) by the function are converted to data layers:

- Data layers, such as `sk.Mask(mask)`, are used as they are.
- Values of simple types (`int`, `float`, `bool`, `str`, `np.ndarray`, and `None`) are converted to the matching layer. A NumPy array becomes an `sk.Image`.
- A tuple or a list is treated as several outputs, each converted separately.
- Any other value becomes an `sk.Any` layer.

Wrapping outputs in data layers explicitly is recommended for arrays, since only the data layer tells whether an array is an image, a mask, or a set of points.

## Algorithm collections

An **algorithm collection** (`sk.MultiAlgorithm`) groups several algorithms under one name. It is created with [`sk.combine`](../how-to/combine.md). In user interfaces, the algorithms of a collection appear in a single dropdown. In Python, methods take an extra `algorithm` argument to select an algorithm by name.

## Clients

A **client** (`sk.Client`) is connected to an [algorithm server](../tutorial/serve.md). It gives access to the algorithms of the server, and sends the computations to the server when it runs them.

## Shared interface

Algorithms, collections, and clients share the same interface, defined by the `AlgorithmRunner` base class:

| Method | Description |
|---|---|
| `run()` | Run an algorithm and return a [`Stack`](layers-and-stacks.md) of results. Supports [tiling](../how-to/tiling.md) and [regions](../how-to/regions.md). |
| `get_sample()` | Get a sample as a `Stack` of parameters. |
| `get_n_samples()` | Get the number of samples. |
| `info()` | Open the documentation page in a web browser. |
| `get_parameters()` | Get the parameters schema. |
| `algorithms` | The list of available algorithm names. |

As a result, the same code works on a local algorithm, an algorithm collection, or a remote server, and the same Napari widget works for all three via `sk.to_napari`.

```python
# All three work the same way:
results = threshold_algo.run(image, threshold=100)
results = multi_algo.run(image, algorithm="threshold", threshold=100)
results = sk.Client("http://localhost:8000").run(image, threshold=100)
```
