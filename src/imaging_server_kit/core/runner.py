from abc import ABC, abstractmethod
from typing import Callable, Dict, Generator, List, Optional, Union

import imaging_server_kit.core._etc as etc
from imaging_server_kit.core.errors import (
    AlgorithmNotFoundError,
    AlgorithmRuntimeError,
    napari_available,
)
from imaging_server_kit.core.results import LayerStackBase, Results
from imaging_server_kit.core.tiling import TilingContext

NAPARI_INSTALLED = napari_available()


def _check_algorithm_available(algorithm: Optional[str], algorithms: List[str]) -> str:
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


def validate_algorithm(func: Callable) -> Callable:
    def wrapper(self, algorithm: Optional[str]=None, *args, **kwargs):
        algorithm = _check_algorithm_available(algorithm, self.algorithms)
        return func(self, algorithm, *args, **kwargs)

    return wrapper


class AlgorithmRunner(ABC):
    @property  # type: ignore
    @abstractmethod
    def algorithms() -> List[str]: ...

    @abstractmethod
    def info(self, algorithm: str) -> None: ...

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
    def _stream(self, algorithm, params_res: Results) -> Generator[Results, None, None]: ...

    def run_generator(
        self,
        algorithm: str,
        params_res: Results,
        tiling_ctx: Optional[TilingContext] = None,
    ):
        if tiling_ctx is None:
            tiling_ctx = (
                TilingContext(tile_size_px=params_res.pixel_domain)
                if params_res.pixel_domain
                else None
            )
        
        for params_tile in params_res.generate_tiles(tiling_ctx):
            for result_tile in self._stream(algorithm, params_tile):
                # Construct the progress data
                if len(params_tile) > 0:
                    params_tile_meta = params_tile[0].tile_meta
                    progress_data = params_tile_meta.tile_idx
                    progress_max_val = params_tile_meta.n_tiles
                else:
                    params_tile_meta = None
                    progress_data = 0
                    progress_max_val = 1

                # Create a progress layer at the current step
                result_tile.create(
                    kind="progress",
                    name="Tile progress",
                    data=progress_data,
                    meta={"max_val": progress_max_val},
                    tile_meta=params_tile_meta,
                )

                # Set the tile_idx, n_tiles, etc. of all result layers based on the params tile
                if params_tile_meta is not None:
                    # Set the tile index to be the current index for all result layers
                    for l in result_tile:
                        l.tile_meta.tile_idx = params_tile_meta.tile_idx
                        l.tile_meta.n_tiles = params_tile_meta.n_tiles
                        # TODO: for the tile position, should we not add the returned l.coords_min to offset them correctly?
                        l.tile_meta.coords_min = params_tile_meta.coords_min
                        l.tile_meta.overlap_px = params_tile_meta.overlap_px
                
                yield result_tile

    def run(
        self,
        *args,
        algorithm: Optional[str] = None,
        tiled: bool = False,
        tile_size_px: int = 64,
        overlap_percent: float = 0.0,
        delay_sec: float = 0.0,
        randomize: bool = False,
        results: Union[LayerStackBase, "napari.Viewer"] = None,  # type: ignore
        **algo_params,
    ) -> Union[LayerStackBase, "napari.Viewer"]:  # type: ignore
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

        # Raise if tileable is set to False and the algo is attempted to be run in tiles
        if tiled and not self.is_tileable(algorithm):
            raise AlgorithmRuntimeError(
                algorithm,
                message="Algorithm cannot be run in tiled mode.",
            )

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
        algo_params_res = Results()

        # TODO: this is a special case for RGB... how else could we handle that?
        for param_name, param_value in resolved_params.items():
            kind = algo_param_defs[param_name].get("param_type")
            if kind == "image":
                rgb = algo_param_defs[param_name].get("rgb")
                algo_params_res.create(kind, param_value, param_name, rgb=rgb)
            else:
                algo_params_res.create(kind, param_value, param_name)

        if results is None:
            results = Results()

        # Handle the special napari case
        special_napari_case = False
        if NAPARI_INSTALLED:
            import napari
            from napari_serverkit import NapariResults

            if isinstance(results, napari.Viewer):
                special_napari_case = True
                results = NapariResults(viewer=results) # type: ignore

        # Construct the tiling context
        if tiled:
            tiling_ctx = TilingContext(
                tile_size_px=tile_size_px,
                overlap_percent=overlap_percent,
                randomize=randomize,
                delay_sec=delay_sec,
            )
        else:
            tiling_ctx = None

        # Run the algorithm and assemble the results
        for tile_results in self.run_generator(algorithm, algo_params_res, tiling_ctx):
            results.merge(tile_results)
        
        # Remove the progress bar
        results.delete(layer_name="Tile progress")

        # Return the results
        if special_napari_case:
            return results.viewer  # type: ignore
        else:
            return results
