"""Utility functions."""

import importlib.resources
import os
import shutil
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import yaml
from jinja2 import Template

from imaging_server_kit.core.results import Results, LayerStackBase
from imaging_server_kit.core.tiling import generate_nd_tiles

templates_dir = Path(
    importlib.resources.files("imaging_server_kit.core").joinpath("templates") # type: ignore
)
static_dir = Path(
    importlib.resources.files("imaging_server_kit.core").joinpath("static") # type: ignore
)


def parse_algo_params_schema(algo_params_schema: Dict[str, Any]) -> Dict[str, Any]:
    algo_params: Dict = algo_params_schema["properties"]
    required_params = algo_params_schema.get("required")
    for param in algo_params:
        if required_params is None:
            algo_params[param]["required"] = False
        else:
            algo_params[param]["required"] = param in required_params
    return algo_params


def open_doc_link(algo_params_schema: Dict, algo_info: Dict) -> None:
    algo_params = parse_algo_params_schema(algo_params_schema)

    with open(templates_dir / "info.html") as f:
        template = Template(f.read())

    rendered_html = template.render(
        {"algo_info": algo_info, "algo_params": algo_params}
    )

    out_dir = Path.home() / ".serverkit"

    if not out_dir.exists():
        os.mkdir(out_dir)

    output_path = out_dir / "output.html"
    css_dir = out_dir / "static" / "css"

    if not css_dir.exists():
        os.makedirs(css_dir)

    css_path = static_dir / "css" / "info.css"
    local_css_path = css_dir / "info.css"
    shutil.copyfile(css_path, local_css_path)

    file_url = f"file://{local_css_path.as_posix()}"
    rendered_html = rendered_html.replace("/static/css/info.css", file_url)

    output_path.write_text(rendered_html, encoding="utf-8")

    webbrowser.open(output_path.resolve().as_uri())


def parse_algo_info(
    metadata_file: Union[str, Path],
    name: str,
    description: str,
    project_url: str,
    tags: List[str],
):
    if Path(metadata_file).exists():
        with open(metadata_file, "r") as file:
            algo_info = yaml.safe_load(file)
    else:
        algo_info = {
            "name": name,
            "description": description,
            "project_url": project_url,
            "tags": tags,
        }
    return algo_info


def generate_tiles(
    param_results: LayerStackBase,
    tile_size_px: int,
    overlap_percent: float,
    delay_sec: float,
    randomize: bool,
):
    """Yield (tile_results, tile_info) for valid tiles."""
    pixel_domain = param_results.get_pixel_domain()
    for tile_info in generate_nd_tiles(
        pixel_domain=pixel_domain,
        tile_size_px=tile_size_px,
        overlap_percent=overlap_percent,
        delay_sec=delay_sec,
        randomize=randomize,
    ):
        collected = []
        for layer in param_results:
            data, meta = layer.get_tile(tile_info)
            if hasattr(data, "shape"):
                if not all(data.shape):
                    break
            collected.append((layer, data, meta))
        else:
            tile_results = Results()
            for layer, data, meta in collected:
                tile_results.create(
                    kind=layer.kind,
                    data=data,
                    name=layer.name,
                    meta=meta,
                )
            yield tile_results, tile_info


def resolve_params(
    algo_param_defs: Dict,
    signature_params: List[str],
    args: Tuple,
    algo_params: Dict,
) -> Dict:
    """Implement a parameters resolution strategy from the explicit parameter annotations, and function signature."""
    resolved = {}

    # Keyword arguments
    provided_kwargs = set(signature_params[: len(args)])
    set_intersect = provided_kwargs.intersection(set(algo_params.keys()))
    if len(set_intersect) > 0:
        raise TypeError(f"Multiple values provided for parameter: {set_intersect}")

    # First, fill the ordered_params based on the provided args
    for k, arg_value in enumerate(args):
        resolved[signature_params[k]] = arg_value

    # Next, fill the ordered_params based on the provided algo_params
    for algo_param_key, algo_param_value in algo_params.items():
        if algo_param_key not in resolved:
            resolved[algo_param_key] = algo_param_value
            
    # Default values
    param_defaults = {
        param_name: algo_param_defs[param_name].get("default")
        for param_name in algo_param_defs.keys()
    }

    # Lastly, fill the remaining ordered_params based on the decorator defaults
    for signature_param in signature_params:
        if signature_param not in resolved:
            resolved[signature_param] = param_defaults.get(signature_param)

    return resolved
