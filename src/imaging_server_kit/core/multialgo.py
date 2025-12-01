from typing import Callable, Dict, List, Optional

from imaging_server_kit.core.results import Results
from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.core.algorithm import Algorithm, validate_algorithm


class MultiAlgorithm(AlgorithmRunner):
    """
    A class representing a collection of algorithms, implementing the AlgorithmRunner protocol.
    - It implements the same methods as the Algorithm class.
    - Passing `algorithm=<algorithm_name>` is needed to identify the algorithm.
    """

    def __init__(self, algorithms: List[Algorithm], name: str = "algorithms"):
        self.sk_algorithms = algorithms
        self.name = name

    @property
    def algorithms_dict(self) -> Dict[str, Algorithm]:
        return {sk_algo.name: sk_algo for sk_algo in self.sk_algorithms}

    @property
    def algorithms(self) -> List[str]:
        return list(self.algorithms_dict.keys())
    
    @validate_algorithm
    def info(self, algorithm: str):
        return self.algorithms_dict[algorithm].info(algorithm)

    @validate_algorithm
    def get_parameters(self, algorithm: str) -> Dict:
        return self.algorithms_dict[algorithm].get_parameters(algorithm)

    @validate_algorithm
    def get_sample(self, algorithm: str, idx: int = 0) -> Optional[Results]:
        return self.algorithms_dict[algorithm].get_sample(algorithm, idx=idx)

    @validate_algorithm
    def get_n_samples(self, algorithm: str) -> int:
        return self.algorithms_dict[algorithm].get_n_samples(algorithm)
    
    @validate_algorithm
    def is_tileable(self, algorithm: str) -> bool:
        return self.algorithms_dict[algorithm].is_tileable(algorithm)

    @validate_algorithm
    def get_signature_params(self, algorithm: str) -> List[str]:
        return self.algorithms_dict[algorithm].get_signature_params(algorithm)

    @validate_algorithm
    def __call__(self, algorithm: str, *args, **kwargs):
        return self.algorithms_dict[algorithm].__call__(*args, **kwargs)

    def _is_stream(self, algorithm: str) -> bool:
        return self.algorithms_dict[algorithm]._is_stream(algorithm)

    def _stream(self, algorithm: str, param_results: Results):
        for results in self.algorithms_dict[algorithm]._stream(algorithm, param_results):
            yield results

    def _tile(
        self,
        algorithm: str,
        tile_size_px: int,
        overlap_percent: float,
        delay_sec: float,
        randomize: bool,
        param_results: Results,
    ):
        """Breaks down the image into tiles before sequentially processing them."""
        for results in self.algorithms_dict[algorithm]._tile(
            algorithm,
            tile_size_px,
            overlap_percent,
            delay_sec,
            randomize,
            param_results,
        ):
            yield results

    def _run(self, algorithm: str, param_results: Results) -> Results:
        return self.algorithms_dict[algorithm]._run(algorithm, param_results)


def combine(algorithms: List[Algorithm], name: str = "algorithms") -> MultiAlgorithm:
    """
    Combine multiple algorithms into an algorithm collection.

    Parameters
    ----------
    algorithms : A list of algorithm objects, or python functions. If a python function is passed, it is converted to an algorithm on the fly.
    name : A name for the algorithm collection.

    Returns
    -------
    An algorithm collection (sk.MultiAlgorithm).
    """
    parsed_algorithms = []
    for algorithm in algorithms:
        try:
            if not isinstance(algorithm, Callable):
                print(f"{algorithm} is not a valid algorithm instance. Skipping it.")
                continue
            if not isinstance(algorithm, Algorithm):
                # We assume the user has passed a regular Python function.
                # We attempt to create an algorithm from it (for convenience)
                algorithm = Algorithm(algorithm)
            parsed_algorithms.append(algorithm)
        except Exception as e:
            print(f"Could not parse this algorithm: {algorithm}. Reason: {e}")

    return MultiAlgorithm(algorithms=parsed_algorithms, name=name)
