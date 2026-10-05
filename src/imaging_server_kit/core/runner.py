from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
import importlib.util

import imaging_server_kit.core._etc as etc
import imaging_server_kit.core._progress_display as progress_display
from imaging_server_kit.core.errors import (
    AlgorithmNotFoundError,
    AlgorithmRuntimeError,
)
from imaging_server_kit.core.stack import Stack, StackTileGenerator
from imaging_server_kit.core.tiling import TilingSpecs
from imaging_server_kit.core.domain import Domain
from imaging_server_kit.types import layer_factory


def napari_available() -> bool:
    return importlib.util.find_spec("napari") is not None


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
    @wraps(func)
    def wrapper(self, algorithm: Optional[str] = None, *args, **kwargs):
        algorithm: str = _check_algorithm_available(algorithm, self.algorithms)
        return func(self, algorithm, *args, **kwargs)

    return wrapper


class AlgorithmRunner(ABC):
    """Interface shared by `sk.Algorithm`, `sk.MultiAlgorithm`, and `sk.Client`.

    The same code can run an algorithm locally, as part of a collection, or remotely
    on an algorithm server. Subclasses provide `_stream()` (how to execute, or request,
    the computation for one tile of parameters) and inherit `run()`, which handles
    parameter resolution, tiling, domain restriction, and result merging identically
    across all three.

    In collections and clients, methods take an `algorithm` argument to select an
    algorithm by name. When it is omitted, the first available algorithm is used.

    Attributes
    ----------
    name : str
        A name identifying the runner.
    algorithms : list of str
        The names of the available algorithms.
    """

    @property  # type: ignore
    @abstractmethod
    def name() -> str:
        """A name identifying the runner."""

    @property  # type: ignore
    @abstractmethod
    def algorithms() -> Union[List[str], Tuple[str, ...]]:
        """The names of the available algorithms."""

    @abstractmethod
    def info(self, algorithm: Optional[str]) -> None:
        """Open the documentation page of an algorithm in a web browser.

        Parameters
        ----------
        algorithm : str, optional
            Name of the algorithm (only needed with collections and clients).
        """

    @abstractmethod
    def get_parameters(self, algorithm: Optional[str]) -> Dict:
        """Get the JSON schema of the parameters of an algorithm.

        Parameters
        ----------
        algorithm : str, optional
            Name of the algorithm (only needed with collections and clients).

        Returns
        -------
        dict
            The JSON schema of the algorithm parameters.
        """

    @abstractmethod
    def get_sample(self, algorithm: Optional[str], idx: int = 0) -> Stack:
        """Get a sample of an algorithm.

        Parameters
        ----------
        algorithm : str, optional
            Name of the algorithm (only needed with collections and clients).
        idx : int, default=0
            Index of the sample.

        Returns
        -------
        Stack
            The sample, as a stack of parameter layers, or `None` if the algorithm
            has no samples.
        """

    @abstractmethod
    def get_n_samples(self, algorithm: Optional[str]) -> int:
        """Get the number of samples of an algorithm.

        Parameters
        ----------
        algorithm : str, optional
            Name of the algorithm (only needed with collections and clients).

        Returns
        -------
        int
            The number of samples.
        """

    @abstractmethod
    def is_tileable(self, algorithm: Optional[str]) -> bool:
        """Whether an algorithm can be run tile-by-tile.

        Parameters
        ----------
        algorithm : str, optional
            Name of the algorithm (only needed with collections and clients).

        Returns
        -------
        bool
            `True` if the algorithm was defined with `tileable=True`.
        """

    @abstractmethod
    def get_signature_params(self, algorithm: Optional[str]) -> List[str]:
        """Get the parameter names of the function of an algorithm, in order.

        Parameters
        ----------
        algorithm : str, optional
            Name of the algorithm (only needed with collections and clients).

        Returns
        -------
        list of str
            The parameter names.
        """

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
        """Lower-level generator variant of `run()`.

        Parameters
        ----------
        algorithm : str
            Name of the algorithm to run.
        params_stack : Stack
            The algorithm parameters, as a stack of layers.
        tiling_ctx : TilingSpecs, optional
            Tiling specifications. If `None`, the parameters are processed as a single tile.

        Yields
        ------
        tuple of (Stack, Stack)
            One `(result_tile, params_tile)` pair per tile and per yielded result.
        """
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
        """Run an algorithm with a set of parameters.

        Parameters
        ----------
        *args
            Algorithm parameters, passed by position, matching the signature of the
            algorithm's function (e.g. `algo.run(image, threshold=100)`).
        algorithm : str, optional
            Name of the algorithm to run (only needed with collections and clients).
        tiled : bool, default=False
            Run the algorithm tile-by-tile. Requires an algorithm defined with
            `tileable=True`.
        tile_size : int or tuple of int, default=64
            Tile size in pixels: a single value, or one value per axis.
        tile_overlap : float, default=0.0
            Overlap between neighbouring tiles, relative to the tile size.
        tile_delay : float, default=0.0
            Extra delay between tiles, in seconds.
        tile_randomize : bool, default=False
            Process the tiles in a random order.
        stack : Stack or napari.Viewer, optional
            A stack, or a Napari viewer, to collect the results into. By default, a
            new stack is created.
        domain : Domain, optional
            A region to which the computation is restricted.
        **algo_params
            Algorithm parameters, passed by name.

        Returns
        -------
        Stack or napari.Viewer
            The results, as a stack of layers. If a Napari viewer was passed as
            `stack`, the viewer is returned.

        Raises
        ------
        pydantic.ValidationError
            If parameter values are invalid.
        TypeError
            If unknown parameters are passed.
        AlgorithmRuntimeError
            If `tiled=True` is used with an algorithm that is not tileable, or if the
            algorithm raises an error.
        """
        algorithm = _check_algorithm_available(algorithm, self.algorithms)

        # Raise if tileable is set to False and the algo is attempted to be run in tiles
        if tiled and not self.is_tileable(algorithm):
            raise AlgorithmRuntimeError(
                algorithm=algorithm,
                message="Algorithm cannot be run in tiled mode.",
            )

        # Parameters from the Pydantic model => gives defaults from the {parameters=} definition
        algo_param_defs = self.get_parameters(algorithm)["properties"]

        # Ordered list of parameter names based on the run function signature (args + kwargs)
        signature_params = self.get_signature_params(algorithm)

        # Catch users making typos or sending unknown parameters
        unknown_params = set(algo_params) - set(signature_params)
        if unknown_params:
            raise TypeError(
                f"{self.name} got unexpected parameter(s) {sorted(unknown_params)}; "
                f"expected one of {signature_params}"
            )

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
            kw = dict(algo_param_defs[name])  # Copy, to avoid mutating the parameters schema
            kind = kw.pop("param_type")
            if "anyOf" in kw:
                kw.pop("anyOf")  # added by Pydantic - we don't need it.

            param_layer = layer_factory(kind=kind, data=data, name=name, **kw)
            params_stack.add(param_layer)

        if stack is None:
            stack = Stack()

        # Handle the special napari case
        special_napari_case = False

        if napari_available():
            import napari

            if isinstance(stack, napari.Viewer):
                from imaging_server_kit.gui.napari_serverkit.napari_stack import (
                    NapariStack,
                )

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

        # The parameters extent doesn't change during the run (computed once)
        params_extent = params_stack.extent

        # Run the algorithm and assemble the stack
        try:
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
                if params_extent is None:
                    # If inputs don't have an extent, we clear up the whole output
                    # (re-evaluated, since the output grows as results are merged)
                    domain_to_erase = stack.extent
                else:
                    domain_to_erase = params_extent

                # Merge the result tile into the stack
                stack.merge(result_tile, domain_to_erase)
        finally:
            # Stop the terminal progress bars, even if the run fails or is interrupted
            progress_display.stop()

        # Remove the progress bar
        stack.delete("Tile progress")

        # Return the stack
        if special_napari_case:
            return stack.viewer  # type: ignore
        else:
            return stack
