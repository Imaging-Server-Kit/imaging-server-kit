from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from tqdm import tqdm

import imaging_server_kit.core._etc as etc
from imaging_server_kit.core.errors import (
    AlgorithmNotFoundError,
    AlgorithmStreamError,
    napari_available,
)
from imaging_server_kit.core.results import LayerStackBase, Results

NAPARI_INSTALLED = napari_available()


def _check_algorithm_available(algorithm, algorithms):
    if algorithm is None:
        if len(algorithms) > 0:
            return algorithms[0]
        else:
            raise AlgorithmNotFoundError(algorithm)
    else:
        if algorithm not in algorithms:
            raise AlgorithmNotFoundError(algorithm)
        else:
            return algorithm


def validate_algorithm(func):
    def wrapper(self, algorithm=None, *args, **kwargs):
        algorithm = _check_algorithm_available(algorithm, self.algorithms)
        return func(self, algorithm, *args, **kwargs)

    return wrapper


class AlgoStream:
    def __init__(self, gen):
        self._it = iter(gen)
        self.value = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._it)
        except StopIteration as e:
            self.value = e.value
            raise


def algo_stream_gen(algo_stream: AlgoStream):
    for x in algo_stream:
        yield x
    yield algo_stream.value


def update_pbar(tile_idx, n_tiles, tqdm_pbar):
    tqdm_pbar.n = tile_idx + 1
    tqdm_pbar.total = n_tiles
    tqdm_pbar.refresh()


class AlgorithmRunner(ABC):
    @property
    @abstractmethod
    def algorithms(): ...

    @abstractmethod
    def info(self, algorithm: str): ...

    @abstractmethod
    def get_parameters(self, algorithm: str) -> Dict: ...

    @abstractmethod
    def get_sample_images(
        self, algorithm: str, first_only: bool
    ) -> Iterable[np.ndarray]: ...

    @abstractmethod
    def get_signature_params(self, algorithm: str) -> List[str]: ...

    @abstractmethod
    def _is_stream(self, algorithm: str): ...

    @abstractmethod
    def _stream(self, algorithm, **algo_params): ...

    @abstractmethod
    def _tile(
        self,
        algorithm,
        tile_size_px,
        overlap_percent,
        delay_sec,
        randomize,
        **algo_params,
    ): ...

    @abstractmethod
    def _run(self, algorithm, **algo_params) -> Iterable[Tuple]: ...

    def run(
        self,
        *args,
        algorithm=None,
        tiled: bool = False,
        tile_size_px: int = 64,
        overlap_percent: float = 0.0,
        delay_sec: float = 0.0,
        randomize: bool = False,
        results: Union[LayerStackBase, "napari.Viewer"] = None,
        **algo_params,
    ) -> Union[LayerStackBase, "napari.Viewer"]:
        algorithm = _check_algorithm_available(algorithm, self.algorithms)

        # Parameters from the Pydantic model => gives defaults from the {parameters=} definition
        algo_param_defs = self.get_parameters(algorithm).get("properties")

        # Ordered list of parameter names based on the run function signature (args + kwargs)
        signature_params = self.get_signature_params(algorithm)

        # Default parameters resolution. Priority is given to defaults set in the wrapped function.
        # If no defaults are set, the defaults from the decorated parameters are used.
        resolved_params = etc.resolve_params(
            algo_param_defs,
            signature_params,
            args,
            algo_params,
        )

        if results is None:
            results = Results()

        special_napari_case = False
        if NAPARI_INSTALLED:
            import napari
            from napari_serverkit import NapariResults
            if isinstance(results, napari.Viewer):
                special_napari_case = True
                results = NapariResults(viewer=results)

        stream = self._is_stream(algorithm)

        if tiled:
            if stream:
                raise AlgorithmStreamError(
                    algorithm,
                    message="Algorithm is a stream. It cannot be run in tiled mode.",
                )
            tqdm_pbar = tqdm()
            for tile_results in self._tile(
                algorithm,
                tile_size_px,
                overlap_percent,
                delay_sec,
                randomize,
                **resolved_params,
            ):
                results.merge(
                    tile_results,
                    tiles_callback=lambda tile_idx, n_tiles: update_pbar(
                        tile_idx, n_tiles, tqdm_pbar
                    ),
                )
        else:
            if stream:
                tqdm_pbar = tqdm()
                wrap = AlgoStream(self._stream(algorithm=algorithm, **resolved_params))
                for frame_results in algo_stream_gen(wrap):
                    results.merge(frame_results)
            else:
                frame_results = self._run(algorithm=algorithm, **resolved_params)
                results.merge(frame_results)

        if special_napari_case:
            return results.viewer
        else:
            return results
