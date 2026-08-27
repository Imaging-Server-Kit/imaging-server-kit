from typing import Callable, Dict, List, Optional, Type
from dataclasses import dataclass
import numpy as np
import pandas as pd

import napari
import napari.layers
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtWidgets import QProgressBar

from imaging_server_kit.core.stack import Stack
from imaging_server_kit.types import Layer, layer_factory
from imaging_server_kit.gui.common.parameter_panel import ParameterPanel


NAPARI_LAYER_MAPPINGS: Dict[str, Type[napari.layers.Layer]] = {
    "image": napari.layers.Image,
    "mask": napari.layers.Labels,
    "points": napari.layers.Points,
    "boxes": napari.layers.Shapes,
    "paths": napari.layers.Shapes,
    "vectors": napari.layers.Vectors,
    "tracks": napari.layers.Tracks,
}


def _set_layer_features(napari_layer: napari.layers.Layer, value) -> None:
    try:
        setattr(napari_layer, "features", value)
    except Exception as e:
        print(f"⚠️ Could not set the layer features attribute: {e}")


def _set_layer_attributes_from_meta(meta: Dict, napari_layer: napari.layers.Layer):
    # Set the features first
    if "features" in meta:
        value = meta["features"]
        
        if isinstance(napari_layer, napari.layers.Labels):
            if "label" in value.keys():
                df = pd.DataFrame(value)
                
                # For Napari to display the correct label on hover, we need to sort rows by 'label' and add missing labels (incl. 0)
                full_labels = pd.DataFrame({"label": np.arange(df.label.max()+1)})
                df_merged = pd.merge(full_labels, df, how="outer", on="label")
                value = df_merged.sort_values(by="label")
                
                _set_layer_features(napari_layer, value)
            else:
                print("⚠️ For mask data, features must contain a `label` column.")
        else:
            _set_layer_features(napari_layer, value)
    
    if "position" in meta:
        value = meta["position"]
        if value is not None:
            try:
                setattr(napari_layer, "translate", value)
            except Exception:
                print("⚠️ Could not set the layer translate attribute.")

    for key, value in meta.items():
        if key not in ["tile_params", "name", "features", "ndim"]:
            try:
                setattr(napari_layer, key, value)
            except Exception:
                print("⚠️ Could not set this layer property: ", key)


@dataclass
class UpdateContext:
    viewer: napari.Viewer
    layer: Layer
    pbar: QProgressBar


def _napari_layer_update(ctx: UpdateContext):
    for l in ctx.viewer.layers:
        if l.name == ctx.layer.name:
            if ctx.layer.data is not None:
                # Assign to the Napari layer data the data from the corresponding context layer
                l.data = ctx.layer.data

                if ctx.layer.position is not None:
                    l.translate = ctx.layer.position

                _set_layer_attributes_from_meta(ctx.layer.meta, l)


def _notification_update(ctx: UpdateContext):
    if ctx.layer.data is not None:
        level = ctx.layer.meta.get("level", "info")
        if level == "error":
            show_error(ctx.layer.data)
        elif level == "warning":
            show_warning(ctx.layer.data)
        else:
            show_info(ctx.layer.data)


def _textlayer_update(ctx: UpdateContext):
    ctx.viewer.text_overlay.visible = True
    ctx.viewer.text_overlay.text = str(ctx.layer.data)


def _pbar_update(ctx: UpdateContext):
    if ctx.layer.data is not None:
        ctx.pbar.setValue(ctx.layer.data)
        ctx.pbar.setMaximum(ctx.layer.meta["max_val"])


class NapariStack(Stack):
    """Stack synced with a Napari Viewer."""

    def __init__(
        self,
        viewer: Optional[napari.Viewer] = None,
        pbar: Optional[QProgressBar] = None,
        layers: Optional[List[Layer]] = None,
        params_panel: Optional[ParameterPanel] = None,
    ):
        # Create a Viewer
        if viewer is None:
            self.viewer = napari.Viewer()
        else:
            self.viewer = viewer

        # Progress bar (shared from ServerKitWidget)
        if pbar is None:
            self.pbar = QProgressBar()
        else:
            self.pbar = pbar

        super().__init__(layers=layers)

        # Instanciate layers and add the existing Napari viewer layers to the stack
        for l in self.viewer.layers:
            self._handle_new_napari_layer(l)

        # Connect viewer events (layer add/remove/rename)
        self.connect_layer_added_event(self.sync_layer_added)
        self.connect_layer_removed_event(self.sync_layer_removed)
        self.connect_layer_renamed_event(self.sync_layer_renamed)
        
        if params_panel is not None:
            self.parameters_panel = params_panel
            self.connect_layer_added_event(self._on_layer_change)
            self.connect_layer_removed_event(self._on_layer_change)
            self.connect_layer_renamed_event(self._on_layer_change)
            self._on_layer_change(None)
            
    def connect_layer_renamed_event(self, func: Callable):
        self.viewer.layers.events.inserted.connect(
            lambda e: e.value.events.name.connect(func)
        )

    def connect_layer_added_event(self, func: Callable):
        self.viewer.layers.events.inserted.connect(func)

    def connect_layer_removed_event(self, func: Callable):
        self.viewer.layers.events.removed.connect(func)

    def sync_layer_added(self, e):
        added_napari_layer = e.source[-1]
        self._handle_new_napari_layer(added_napari_layer)

    def sync_layer_renamed(self, e):
        viewer_layer_names = [l.name for l in self.viewer.layers]
        new_name = e.source
        for layer in self.layers:
            if layer.name not in viewer_layer_names:
                layer.name = new_name

    def sync_layer_removed(self, e):
        layer_name = e.value.name
        self.delete(layer_name)

    def _handle_new_napari_layer(self, napari_layer):
        existing_layer = self.read(napari_layer.name)
        if existing_layer is not None:
            return
        # layer_to_kind = {}  # TODO: better approach...
        if isinstance(napari_layer, napari.layers.Image):
            kind = "image"
            data = napari_layer.data
        elif isinstance(napari_layer, napari.layers.Labels):
            kind = "mask"
            data = napari_layer.data
        elif isinstance(napari_layer, napari.layers.Points):
            kind = "points"
            data = napari_layer.data
        elif isinstance(napari_layer, napari.layers.Tracks):
            kind = "tracks"
            data = napari_layer.data
        elif isinstance(napari_layer, napari.layers.Vectors):
            kind = "vectors"
            data = napari_layer.data
        elif isinstance(napari_layer, napari.layers.Shapes):
            # TODO: For now, when a `Shapes` layer is created, we assume it's meant to contain boxes (rectangles).
            # So, it won't work with algorithms that would use annotated "Paths" as input (quite rare).
            kind = "boxes"
            data = None  # instead of []
        else:
            print("Could not convert this layer: ", napari_layer)
            return

        # Keep track of the new Napari layer in the layer stack (without any layer metadata)
        # TODO: check this
        layer = layer_factory(kind=kind, name=napari_layer.name, data=data)
        self.add(layer)

    def _post_add(self, layer: Layer) -> Layer:
        if layer.data is None:
            return layer

        kind = layer.kind
        data = layer.data
        name = layer.name
        meta = layer.meta

        if layer.name in [l.name for l in self.viewer.layers]:
            return layer

        napari_layer = None
        if kind == "image":
            napari_layer = self.viewer.add_image(data, name=name)
        elif kind == "mask":
            napari_layer = self.viewer.add_labels(data.astype(np.uint16), name=name)
        elif kind == "points":
            napari_layer = self.viewer.add_points(data, name=name)
        elif kind in ["boxes", "paths"]:
            if "shape_type" in meta:  # Make sure it isn't used twice
                meta.pop("shape_type")
            if kind == "boxes":
                napari_layer = self.viewer.add_shapes(
                    data, name=name, shape_type="rectangle"
                )
            elif kind == "paths":
                napari_layer = self.viewer.add_shapes(
                    data, name=name, shape_type="path"
                )
        elif kind == "vectors":
            napari_layer = self.viewer.add_vectors(data, name=name)
        elif kind == "tracks":
            napari_layer = self.viewer.add_tracks(data, name=name)

        if napari_layer is not None:
            _set_layer_attributes_from_meta(meta, napari_layer)

        return layer

    def _post_delete(self, name: str) -> None:
        """Hook called after layer deletion."""
        for idx, l in enumerate(self.viewer.layers):
            if l.name == name:
                self.viewer.layers.pop(idx)

    def _post_merge(self, dst_layers: List[Layer]) -> None:
        for layer in dst_layers:
            update_hooks = {
                "image": _napari_layer_update,
                "mask": _napari_layer_update,
                "points": _napari_layer_update,
                "boxes": _napari_layer_update,
                "paths": _napari_layer_update,
                "vectors": _napari_layer_update,
                "tracks": _napari_layer_update,
                "notification": _notification_update,
                "float": _textlayer_update,
                "int": _textlayer_update,
                "bool": _textlayer_update,
                "str": _textlayer_update,
                "choice": _textlayer_update,
                "progress": _pbar_update,
            }
            update_func: Optional[Callable] = update_hooks.get(layer.kind)

            # Before doing the napari layer update, make sure the viewer layers exist by running post_add..
            # A little weird, but for now the only solution that seems to work!
            if update_func is _napari_layer_update:
                if not layer.name in [l.name for l in self.viewer.layers]:
                    self._post_add(layer)

            if update_func is not None:
                ctx = UpdateContext(viewer=self.viewer, layer=layer, pbar=self.pbar)
                update_func(ctx)

    def _on_layer_change(self, *args, **kwargs):
        for kind, cb_list in self.parameters_panel.layer_comboboxes.items():
            layer_type: Type[napari.layers.Layer] = NAPARI_LAYER_MAPPINGS[kind]
            for cb in cb_list:
                cb.clear()
                for layer in self.viewer.layers:
                    if isinstance(layer, layer_type):

                        # Napari layers data are not always in the format expected by serverkit, so we do the conversion here
                        # and assign serverkit-formatted data to the combobox data attributes, which get retreived later as parameters

                        # For boxes, extract the rectangle data from shapes layers (and convert them to Numpy)
                        if kind == "boxes":
                            data = None
                            if isinstance(layer.data, list):
                                if len(layer.data) > 0:
                                    rectangle_data = []
                                    for d, t in zip(layer.data, layer.shape_type):
                                        if t == "rectangle":
                                            rectangle_data.append(d)
                                    if len(rectangle_data) > 0:
                                        data = np.array(rectangle_data)
                            cb.addItem(layer.name, data)

                        # For paths, extract the path data from shapes layers
                        elif kind == "paths":
                            data = None
                            if isinstance(layer.data, list):
                                if len(layer.data) > 0:
                                    path_data = []
                                    for d, t in zip(layer.data, layer.shape_type):
                                        if t == "rectangle":
                                            path_data.append(d)
                                    if len(path_data) > 0:
                                        data = path_data
                            cb.addItem(layer.name, data)

                        else:
                            cb.addItem(layer.name, layer.data)