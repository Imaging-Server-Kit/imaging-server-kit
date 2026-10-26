import inspect
from types import ModuleType

import imaging_server_kit as sk

# The example algorithms are also available directly from `imaging_server_kit.demo`
from .examples import *
from . import examples, tools


def _collect(module: ModuleType, name: str) -> sk.MultiAlgorithm:
    """Combine the algorithms defined in a module, sorted alphabetically by name."""
    algos = [obj for _, obj in inspect.getmembers(module) if isinstance(obj, sk.Algorithm)]
    algos.sort(key=lambda obj: obj.name)
    return sk.combine(algos, name=name)


multi_algo_demos = _collect(examples, name="demo")
multi_algo_tools = _collect(tools, name="tools")
