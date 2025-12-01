from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from tqdm import tqdm

import imaging_server_kit.core._etc as etc
from imaging_server_kit.core.errors import (
    AlgorithmNotFoundError,
    AlgorithmStreamError,
    AlgorithmRuntimeError,
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
        except (StopIteration, AlgorithmRuntimeError) as e:
            if isinstance(e, StopIteration):
                self.value = e.value
                raise
            elif isinstance(e, AlgorithmRuntimeError):
                raise e


def algo_stream_gen(algo_stream: AlgoStream):
    for x in algo_stream:
        yield x
    yield algo_stream.value


def update_pbar(tile_idx, n_tiles, tqdm_pbar):
    tqdm_pbar.n = tile_idx + 1
    tqdm_pbar.total = n_tiles
    tqdm_pbar.refresh()


class AlgorithmRunner(ABC):
    @property # type: ignore
    @abstractmethod
    def algorithms(): ...

    @abstractmethod
    def info(self, algorithm: str): ...

    @abstractmethod
    def get_parameters(self, algorithm: str) -> Dict: ...

    @abstractmethod
    def get_sample(self, algorithm: str, idx: int = 0) -> LayerStackBase: ...

    @abstractmethod
    def get_n_samples(self, algorithm: str) -> int: ...
    
    @abstractmethod
    def is_tileable(self, algorithm: str) -> bool: ...

    @abstractmethod
    def get_signature_params(self, algorithm: str) -> List[str]: ...

    @abstractmethod
    def _is_stream(self, algorithm: str): ...

    @abstractmethod
    def _stream(self, algorithm, param_results: Results): ...

    @abstractmethod
    def _tile(
        self,
        algorithm: Optional[str],
        tile_size_px: int,
        overlap_percent: float,
        delay_sec: float,
        randomize: bool,
        param_results: Results,
    ): ...

    @abstractmethod
    def _run(self, algorithm, param_results: Results) -> Results: ...

    def run(
        self,
        *args,
        algorithm: Optional[str]=None,
        tiled: bool = False,
        tile_size_px: int = 64,
        overlap_percent: float = 0.0,
        delay_sec: float = 0.0,
        randomize: bool = False,
        results: Union[LayerStackBase, "napari.Viewer"] = None, # type: ignore
        **algo_params,
    ) -> Union[LayerStackBase, "napari.Viewer"]: # type: ignore
        """
        Execute an algorithm with a set of parameters.
        
        Parameters
        ----------
        algorithm: The algorithm to run (only used with algorithm collections).
        tiled: Set to True for tiled inference.
        tile_size_px: Tile size in pixels.
        overlap_percent: Relative overlap between tiles.
        delay_sec: Artificial delay (sleep) time between each tile processing.
        randomize: Process tiles in a random order.
        results: An optional layer stack object to collect results into.
        """
        algorithm = _check_algorithm_available(algorithm, self.algorithms)

        # Parameters from the Pydantic model => gives defaults from the {parameters=} definition
        algo_param_defs = self.get_parameters(algorithm)["properties"]

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

        # Convert the resolved parameters to a Results object
        param_results = Results()
        for param_name, param_value in resolved_params.items():
            kind = algo_param_defs[param_name].get("param_type")
            if kind == "image":
                # TODO: this is a special case for RGB... how else could we handle that?
                rgb = algo_param_defs[param_name].get("rgb")
                param_results.create(kind, param_value, param_name, rgb=rgb)
            else:
                param_results.create(kind, param_value, param_name)

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
            if self.tileable is False:
                raise AlgorithmRuntimeError(
                    algorithm,
                    message="Algorithm cannot be run in tiled mode.",
                )
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
                param_results,
            ): # type: ignore
                results.merge(
                    tile_results,
                    tiles_callback=lambda tile_idx, n_tiles: update_pbar(
                        tile_idx, n_tiles, tqdm_pbar
                    ),
                )
        else:
            if stream:
                tqdm_pbar = tqdm()
                wrap = AlgoStream(self._stream(algorithm, param_results))
                for frame_results in algo_stream_gen(wrap):
                    results.merge(frame_results)
            else:
                run_results = self._run(algorithm, param_results)
                results.merge(run_results)

        if special_napari_case:
            return results.viewer # type: ignore
        else:
            return results
