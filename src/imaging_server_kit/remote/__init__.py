from typing import Callable, Union

from imaging_server_kit.core.algorithm import Algorithm
from imaging_server_kit.core.multialgo import MultiAlgorithm

from .app import AlgorithmApp
from .client import Client


def serve(
    algorithm: Union[Algorithm, MultiAlgorithm, Callable], *args, **kwargs
) -> None:
    """Serve an algorithm, algorithm collection, or plain Python function over HTTP.

    Starts a FastAPI server (with uvicorn) and blocks until it is stopped. Run it from
    a Python script, not from a Jupyter notebook.

    Parameters
    ----------
    algorithm : Algorithm, MultiAlgorithm, or callable
        The algorithm or collection to serve. A plain Python function is converted to
        an algorithm.
    *args, **kwargs
        Passed to the server: `host` (default `"0.0.0.0"`, all network interfaces)
        and `port` (default `8000`).

    Examples
    --------
    >>> if __name__ == "__main__":
    ...     sk.serve(threshold_algo, port=8000)
    """
    from imaging_server_kit.remote.app import AlgorithmApp

    if isinstance(algorithm, Algorithm):
        algorithm_servers = [algorithm]
    elif isinstance(algorithm, MultiAlgorithm):
        algorithm_servers = list(algorithm.algorithms_dict.values())
    else:
        # Assuming the user has passed a "raw" Python function, we attempt to convert it to an Algorithm:
        algorithm = Algorithm(algorithm)
        algorithm_servers = [algorithm]

    algo_app = AlgorithmApp(algorithms=algorithm_servers, name=algorithm.name)
    algo_app.serve(*args, **kwargs)