from typing import Any, Dict, Iterable, List, Tuple

from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.core.algorithm import Algorithm, validate_algorithm


class MultiAlgorithm(AlgorithmRunner):
    def __init__(self, algorithms: List[Algorithm], name: str = "algorithms"):
        self.sk_algorithms = algorithms
        self.name = name

    @property
    def algorithms_dict(self) -> Dict:
        return {sk_algo.name: sk_algo for sk_algo in self.sk_algorithms}

    @property
    def algorithms(self) -> Iterable[str]:
        return list(self.algorithms_dict.keys())

    @validate_algorithm
    def info(self, algorithm: str):
        algorithm = self.algorithms_dict[algorithm]
        return algorithm.info(algorithm.name)

    @validate_algorithm
    def get_parameters(self, algorithm: str) -> dict:
        algorithm = self.algorithms_dict[algorithm]
        return algorithm.get_parameters(algorithm.name)

    @validate_algorithm
    def get_sample(self, algorithm: str, idx: int = 0) -> Dict[str, Any]:
        algorithm = self.algorithms_dict[algorithm]
        return algorithm.get_sample(algorithm.name, idx=idx)
    
    @validate_algorithm
    def get_n_samples(self, algorithm: str) -> int:
        algorithm = self.algorithms_dict[algorithm]
        return algorithm.get_n_samples(algorithm.name)

    @validate_algorithm
    def get_signature_params(self, algorithm: str) -> List[str]:
        algorithm = self.algorithms_dict[algorithm]
        return algorithm.get_signature_params(algorithm.name)

    def __call__(self, algorithm: str, *args, **kwargs):
        algorithm = self.algorithms_dict[algorithm]
        return algorithm.__call__(*args, **kwargs)

    def _is_stream(self, algorithm: str) -> bool:
        algorithm = self.algorithms_dict[algorithm]
        return algorithm._is_stream(algorithm.name)

    def _stream(self, algorithm: str, **algo_params):
        algorithm = self.algorithms_dict[algorithm]
        for results in algorithm._stream(algorithm.name, **algo_params):
            yield results
        return []

    def _tile(
        self,
        algorithm: str,
        tile_size_px,
        overlap_percent,
        delay_sec,
        randomize,
        **algo_params,
    ):
        """Breaks down the image into tiles before sequentially processing them."""
        algorithm = self.algorithms_dict[algorithm]

        for results in algorithm._tile(
            algorithm.name,
            tile_size_px,
            overlap_percent,
            delay_sec,
            randomize,
            **algo_params,
        ):
            yield results
        return []

    def _run(self, algorithm: str, **algo_params) -> Iterable[Tuple]:
        algorithm = self.algorithms_dict[algorithm]
        return algorithm._run(algorithm.name, **algo_params)


def combine(algorithms: List[Algorithm], name: str = "algorithms"):
    return MultiAlgorithm(algorithms=algorithms, name=name)
