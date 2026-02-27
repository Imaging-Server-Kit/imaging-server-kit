from typing import Callable, Union

from imaging_server_kit.core.algorithm import Algorithm
from imaging_server_kit.core.multialgo import MultiAlgorithm

from .app import AlgorithmApp
from .client import Client


def serve(
    algorithm: Union[Algorithm, MultiAlgorithm, Callable], *args, **kwargs
) -> None:
    """
    Serve an algorithm as an HTTP server.

    Parameters
    ----------
    algorithm : The algorithm object to serve.
    host : The IP of the host (default: "0.0.0.0")
    port : The network port (default: 8000)
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