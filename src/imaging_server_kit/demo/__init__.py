from .examples import *

import imaging_server_kit as sk
import importlib
import inspect

# Import the examples
module_examples = importlib.import_module("imaging_server_kit.demo.examples")

# Collect all algorithms examples in the module
example_algos = [
    obj
    for name, obj in inspect.getmembers(module_examples)
    if isinstance(obj, sk.Algorithm)
]

# Sort algos alphabetically by name
example_algos.sort(key=lambda obj: obj.name)

multi_algo_demos = sk.combine(example_algos, name="demo")

# Import the tools
module_tools = importlib.import_module("imaging_server_kit.demo.tools")

# Collect all algorithm tools in the module
tools_algos = [
    obj
    for name, obj in inspect.getmembers(module_tools)
    if isinstance(obj, sk.Algorithm)
]

# Sort tools alphabetically by name
tools_algos.sort(key=lambda obj: obj.name)

multi_algo_tools = sk.combine(tools_algos, name="tools")
