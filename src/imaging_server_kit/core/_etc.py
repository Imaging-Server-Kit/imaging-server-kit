"""Diverse utility functions."""

import importlib.resources
import os
import shutil
import webbrowser
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from jinja2 import Template

from imaging_server_kit.core.tiling import generate_nd_tiles
from imaging_server_kit.types import DATA_TYPES

templates_dir = Path(
    importlib.resources.files("imaging_server_kit.core").joinpath("templates")
)
static_dir = Path(
    importlib.resources.files("imaging_server_kit.core").joinpath("static")
)


def parse_algo_params_schema(algo_params_schema):
    algo_params = algo_params_schema.get("properties")
    required_params = algo_params_schema.get("required")
    for param in algo_params.keys():
        if required_params is None:
            algo_params[param]["required"] = False
        else:
            algo_params[param]["required"] = param in required_params
    return algo_params


def open_doc_link(algo_params_schema, algo_info):
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


def parse_algo_info(metadata_file: Union[str, Path], name, title, description, project_url, tags):
    if Path(metadata_file).exists():
        with open(metadata_file, "r") as file:
            algo_info = yaml.safe_load(file)
    else:
        algo_info = {
            "name": name,
            "title": title,
            "description": description,
            "project_url": project_url,
            "tags": tags,
        }
    return algo_info


def get_pixel_domain(param_types, param_values):
    domains = []
    for [param_value, param_type] in zip(param_values, param_types):
        domain = DATA_TYPES.get(param_type).pixel_domain(param_value)
        if domain is not None:
            domains.append(domain)
    pixel_domain = np.max(np.stack(domains), axis=0)
    return pixel_domain


def generate_tiles(algo_param_defs, algo_params, tile_size_px, overlap_percent, delay_sec, randomize):
    param_keys = list(algo_params.keys())
    param_values = list(algo_params.values())
    
    param_types = []
    for key in param_keys:
        algo_def = algo_param_defs.get(key)
        if algo_def:
            param_types.append(algo_def.get("param_type"))

    pixel_domain = get_pixel_domain(param_types, param_values)

    first_tile = True
    for tile_info in generate_nd_tiles(
        pixel_domain=pixel_domain,
        tile_size_px=tile_size_px,
        overlap_percent=overlap_percent,
        delay_sec=delay_sec,
        randomize=randomize,
    ):
        if first_tile:
            tile_info.get("tile_params")["first_tile"] = True
            first_tile = False

        algo_params_tile = {}
        invalid = False
        for [(key, param_value), param_type] in zip(
            algo_params.items(), param_types
        ):
            tile = DATA_TYPES.get(param_type)._get_tile(param_value, tile_info)
            if hasattr(tile, "shape"):
                if not all(tile.shape):
                    invalid = True
            algo_params_tile[key] = tile

        if invalid:
            # Skip running the algo when an invalid tile was found.
            continue

        yield algo_params_tile, tile_info


def resolve_params(algo_param_defs, signature_params, args, algo_params):
    """Implement a parameters resolution strategy from the explicit parameter annotations, and function signature."""
    # Default values
    param_defaults = {
        param_name: algo_param_defs.get(param_name).get("default")
        for param_name in algo_param_defs.keys()
    }

    resolved_params = {}

    # Keyword arguments
    provided_kwargs = set(signature_params[: len(args)])
    set_intersect = provided_kwargs.intersection(set(algo_params.keys()))
    if len(set_intersect) > 0:
        raise TypeError(f"Multiple values provided for parameter: {set_intersect}")

    # First, fill the ordered_params based on the provided args
    for k, arg_value in enumerate(args):
        resolved_params[signature_params[k]] = arg_value

    # Next, fill the ordered_params based on the provided algo_params
    for algo_param_key, algo_param_value in algo_params.items():
        if algo_param_key not in resolved_params:
            resolved_params[algo_param_key] = algo_param_value

    # Lastly, fill the remaining ordered_params based on the decorator defaults
    for signature_param in signature_params:
        if signature_param not in resolved_params:
            resolved_params[signature_param] = param_defaults.get(signature_param)

    return resolved_params
