import warnings
from typing import Dict, List, Optional

from imaging_server_kit.core.stack import Stack
from imaging_server_kit.core.runner import AlgorithmRunner, validate_algorithm
from imaging_server_kit.core.algorithm import Algorithm


class MultiAlgorithm(AlgorithmRunner):
    """A collection of algorithms exposed under a single interface.

    Collections are usually created with `sk.combine()` rather than instantiated
    directly. They implement the shared
    [`AlgorithmRunner`][imaging_server_kit.AlgorithmRunner] interface, where methods
    take an `algorithm` argument to select an algorithm by name.

    Parameters
    ----------
    algorithms : list of Algorithm
        The algorithms in the collection. If several algorithms have the same name,
        only the last one is kept.
    name : str, default="algorithms"
        A name for the collection.

    Attributes
    ----------
    algorithms_dict : dict
        A dictionary mapping algorithm names to algorithms.
    algorithms : list of str
        The names of the algorithms in the collection.
    """

    def __init__(self, algorithms: List[Algorithm], name: str = "algorithms"):
        self.sk_algorithms = algorithms
        self._name = name

        self._algorithms_dict: Dict[str, Algorithm] = {}
        for sk_algo in algorithms:
            if sk_algo.name in self._algorithms_dict:
                warnings.warn(
                    f"Several algorithms are named `{sk_algo.name}`; only the last one is kept.",
                    stacklevel=2,
                )
            self._algorithms_dict[sk_algo.name] = sk_algo

    @property
    def name(self) -> str:
        return self._name

    @property
    def algorithms_dict(self) -> Dict[str, Algorithm]:
        return self._algorithms_dict

    @property
    def algorithms(self) -> List[str]:
        return list(self.algorithms_dict.keys())

    @validate_algorithm
    def info(self, algorithm: Optional[str] = None):
        return self.algorithms_dict[algorithm].info(algorithm)  # type: ignore

    @validate_algorithm
    def get_parameters(self, algorithm: Optional[str] = None) -> Dict:
        return self.algorithms_dict[algorithm].get_parameters(algorithm)  # type: ignore

    @validate_algorithm
    def get_sample(
        self, algorithm: Optional[str] = None, idx: int = 0
    ) -> Optional[Stack]:
        return self.algorithms_dict[algorithm].get_sample(algorithm, idx=idx)  # type: ignore

    @validate_algorithm
    def get_n_samples(self, algorithm: Optional[str] = None) -> int:
        return self.algorithms_dict[algorithm].get_n_samples(algorithm)  # type: ignore

    @validate_algorithm
    def is_tileable(self, algorithm: Optional[str] = None) -> bool:
        return self.algorithms_dict[algorithm].is_tileable(algorithm)  # type: ignore

    @validate_algorithm
    def get_signature_params(self, algorithm: Optional[str] = None) -> List[str]:
        return self.algorithms_dict[algorithm].get_signature_params(algorithm)

    @validate_algorithm
    def __call__(self, algorithm: str, *args, **kwargs):
        return self.algorithms_dict[algorithm].__call__(*args, **kwargs)

    def _stream(self, algorithm: str, params_stack: Stack):
        for stack in self.algorithms_dict[algorithm]._stream(algorithm, params_stack):
            yield stack


def combine(algorithms: List[Algorithm], name: str = "algorithms") -> MultiAlgorithm:
    """Combine algorithms, or plain Python functions, into a collection.

    Parameters
    ----------
    algorithms : list of Algorithm or callable
        The algorithms to combine. Plain Python functions are converted to algorithms.
    name : str, default="algorithms"
        A name for the collection.

    Returns
    -------
    MultiAlgorithm
        The algorithm collection.

    Examples
    --------
    >>> multi_algo = sk.combine([threshold_algo, gaussian_algo], name="my-algorithms")
    """
    parsed_algorithms = []
    for algorithm in algorithms:
        try:
            if not callable(algorithm):
                warnings.warn(
                    f"{algorithm} is not a valid algorithm instance. Skipping it.",
                    stacklevel=2,
                )
                continue
            if not isinstance(algorithm, Algorithm):
                # We assume the user has passed a regular Python function.
                # We attempt to create an algorithm from it (for convenience)
                algorithm = Algorithm(algorithm)
            parsed_algorithms.append(algorithm)
        except Exception as e:
            warnings.warn(
                f"Could not parse this algorithm: {algorithm}. Reason: {e}",
                stacklevel=2,
            )

    return MultiAlgorithm(algorithms=parsed_algorithms, name=name)
