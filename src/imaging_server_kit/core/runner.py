from abc import ABC, abstractmethod
from typing import Callable, Dict, Generator, List, Optional, Union

import imaging_server_kit.core._etc as etc
from imaging_server_kit.core.errors import (
    AlgorithmNotFoundError,
    AlgorithmRuntimeError,
    napari_available,
)
from imaging_server_kit.core.stack import Stack, StackTileGenerator
from imaging_server_kit.core.tiling import TilingSpecs
from imaging_server_kit.core.domain import Domain
from imaging_server_kit.types import layer_factory

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
    def wrapper(self, algorithm: Optional[str] = None, *args, **kwargs):
        algorithm = _check_algorithm_available(algorithm, self.algorithms)
        return func(self, algorithm, *args, **kwargs)

    return wrapper


class AlgorithmRunner(ABC):
    """The algorithm runner base class, parent to sk.Algorithm, sk.MultiAlgorithm and sk.Client."""

    @property  # type: ignore
    @abstractmethod
    def algorithms() -> List[str]: ...

    @abstractmethod
    def info(self, algorithm: str) -> None: ...

    @abstractmethod
    def get_parameters(self, algorithm: str) -> Dict: ...

    @abstractmethod
    def get_sample(self, algorithm: str, idx: int = 0) -> Stack: ...

    @abstractmethod
    def get_n_samples(self, algorithm: str) -> int: ...

    @abstractmethod
    def is_tileable(self, algorithm: str) -> bool: ...

    @abstractmethod
    def get_signature_params(self, algorithm: str) -> List[str]: ...

    @abstractmethod
    def _stream(
        self, algorithm, params_stack: Stack
    ) -> Generator[Stack, None, None]: ...

    def run_generator(
        self,
        algorithm: str,
        params_stack: Stack,
        tiling_ctx: Optional[TilingSpecs] = None,
    ):
        tile_progress_needed = tiling_ctx is not None

        if tiling_ctx is None:
            # We create a single tile for the stack
            tiling_ctx = (
                TilingSpecs(tile_size=params_stack.size)
                if params_stack.size
                else None  # Happens with non-spatial inputs
            )

        stack_tile_gen = StackTileGenerator()
        for params_tile in stack_tile_gen.generate_tiles(params_stack, tiling_ctx):
            for result_tile in self._stream(algorithm, params_tile):
                # Create a progress layer at the current step
                if tile_progress_needed:
                    progress_layer = layer_factory(
                        kind="progress",
                        name="Tile progress",
                        data=params_tile.tile_meta.tile_idx + 1,
                        max_val=params_tile.tile_meta.n_tiles,
                    )
                    result_tile.add(progress_layer)

                # The result tile inherits the tile_meta of the parameters tile
                result_tile.tile_meta = params_tile.tile_meta

                yield result_tile, params_tile

    def run(
        self,
        *args,
        algorithm: Optional[str] = None,
        tiled: bool = False,
        tile_size: int = 64,
        tile_overlap: float = 0.0,
        tile_delay: float = 0.0,
        tile_randomize: bool = False,
        stack: Union[Stack, "napari.Viewer"] = None,  # type: ignore
        domain: Optional[Domain] = None,
        **algo_params,
    ) -> Union[Stack, "napari.Viewer"]:  # type: ignore
        """
        Execute an algorithm with a set of parameters.

        Parameters
        ----------
        algorithm: The algorithm to run (only used with algorithm collections).
        tiled: Set to True for tiled inference.
        tile_size: Tile size in pixels.
        tile_overlap: Relative overlap between tiles.
        tile_delay: Extra delay time in seconds between tiles.
        tile_randomize: Process tiles in a random order.
        stack: An optional layer stack object to collect results into.
        domain: An optional domain in which to restrict the computation.
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

        # Convert the resolved parameters to a Stack object
        params_stack = Stack()
        for name, data in resolved_params.items():
            kw = algo_param_defs[name]
            kind = kw.pop("param_type")
            if "anyOf" in kw:
                kw.pop("anyOf")  # added by Pydantic - we don't need it.

            param_layer = layer_factory(kind=kind, data=data, name=name, **kw)
            params_stack.add(param_layer)

        if stack is None:
            stack = Stack()

        # Handle the special napari case
        special_napari_case = False
        if NAPARI_INSTALLED:
            import napari
            from napari_serverkit import NapariStack

            if isinstance(stack, napari.Viewer):
                special_napari_case = True
                stack = NapariStack(viewer=stack)  # type: ignore

        # Construct the tiling context
        if tiled:
            tiling_ctx = TilingSpecs(
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                tile_randomize=tile_randomize,
                tile_delay=tile_delay,
            )
        else:
            tiling_ctx = None

        # If a domain is passed, restrict the computation to that domain
        if domain:
            params_stack = params_stack.select(domain)

        # Run the algorithm and assemble the stack
        for result_tile, params_tile in self.run_generator(
            algorithm, params_stack, tiling_ctx
        ):
            # If the parameters tile and result tile both have a position, 
            # the result tile's position is offset by that of the parameters tile
            if (result_tile.position is not None) and (
                params_tile.position is not None
            ):
                result_tile.position = tuple(
                    [p + q for p, q in zip(params_tile.position, result_tile.position)]
                )
            else:
                result_tile.position = params_tile.position

            # We assume that reinitializing the parameters domain on first tile
            # will be the correct behaviour most of the time.
            if params_stack.extent is None:
                # If inputs don't have an extent, we clear up the whole output
                domain_to_erase = stack.extent
            else:
                domain_to_erase = params_stack.extent

            # Merge the result tile into the stack
            stack.merge(result_tile, domain_to_erase)

        # Remove the progress bar
        stack.delete("Tile progress")

        # Return the stack
        if special_napari_case:
            return stack.viewer  # type: ignore
        else:
            return stack
