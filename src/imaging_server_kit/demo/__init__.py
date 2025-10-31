from .examples import *

import imaging_server_kit as sk
import importlib
import inspect

module = importlib.import_module("imaging_server_kit.demo.examples")

# Collect all algorithms in the module
example_algos = [
    obj for name, obj in inspect.getmembers(module)
    if isinstance(obj, sk.Algorithm)
]

multi_algo_examples = sk.combine(example_algos, name="demo")