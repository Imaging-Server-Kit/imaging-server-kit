from typing import Callable, Dict, Optional
from dataclasses import dataclass
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

NAPARI_LAYER_TYPES = ["image", "mask", "points", "boxes", "paths", "vectors", "tracks"]


@dataclass
class UIStateItem:
    param_type: str
    qt_widget: QWidget
    qt_widget_setter_func: Optional[Callable]
    widget_value_recover_func: Callable


class ParameterPanel:
    def __init__(self, trigger: Callable):
        self._trigger_func = trigger

        self.ui_state: Dict[str, UIStateItem] = {}
        self.layer_comboboxes = {}

        self.widget = QGroupBox()
        self.widget.setTitle("Parameters")

        self.layout = QGridLayout()
        self.widget.setLayout(self.layout)

    def update(self, schema: Dict):
        # Clean-up the previous dynamic UI layout
        for i in reversed(range(self.layout.count())):
            ui_item = self.layout.itemAt(i)
            if ui_item is not None:
                ui_item_widget = ui_item.widget()
                if ui_item_widget is not None:
                    ui_item_widget.setParent(None)

        # Generate the new dynamic UI state and layout
        self.ui_state: Dict[str, UIStateItem] = {}
        for k, (param_name, param_values) in enumerate(schema["properties"].items()):
            # Add the right UI element based on the retreived parameter type.
            param_type = param_values.get("param_type")

            if param_type == "choice":
                qt_widget = QComboBox()
                # If there is only one element, we get a `const` attribute instead of `enum`
                if param_values.get("enum") is None:
                    qt_widget.addItem(param_values.get("const"))
                else:
                    qt_widget.addItems(param_values.get("enum"))
                qt_widget.setCurrentText(param_values.get("default", ""))
                if param_values.get("auto_call"):
                    qt_widget.currentTextChanged.connect(self._trigger_func)
                qt_widget_setter_func = qt_widget.setCurrentText
                widget_value_recover_func = lambda qt_widget: qt_widget.currentText()
            elif param_type == "int":
                qt_widget = QSpinBox()
                if param_values.get("min"):
                    qt_widget.setMinimum(param_values.get("min"))
                if param_values.get("max"):
                    qt_widget.setMaximum(param_values.get("max"))
                qt_widget.setValue(param_values.get("default", 0))
                if param_values.get("step"):
                    qt_widget.setSingleStep(param_values.get("step"))
                if param_values.get("auto_call"):
                    qt_widget.valueChanged.connect(self._trigger_func)
                qt_widget_setter_func = qt_widget.setValue
                widget_value_recover_func = lambda qt_widget: int(qt_widget.value())
            elif param_type == "float":
                qt_widget = QDoubleSpinBox()
                if param_values.get("min"):
                    qt_widget.setMinimum(param_values.get("min"))
                if param_values.get("max"):
                    qt_widget.setMaximum(param_values.get("max"))
                qt_widget.setValue(param_values.get("default", 0.0))
                if param_values.get("step"):
                    qt_widget.setSingleStep(param_values.get("step"))
                if param_values.get("auto_call"):
                    qt_widget.valueChanged.connect(self._trigger_func)
                qt_widget_setter_func = qt_widget.setValue
                widget_value_recover_func = lambda qt_widget: float(qt_widget.value())
            elif param_type == "bool":
                qt_widget = QCheckBox()
                qt_widget.setChecked(param_values.get("default", False))
                if param_values.get("auto_call"):
                    qt_widget.stateChanged.connect(self._trigger_func)
                qt_widget_setter_func = qt_widget.setChecked
                widget_value_recover_func = lambda qt_widget: qt_widget.isChecked()
            elif param_type == "str":
                qt_widget = QLineEdit()
                qt_widget.setText(param_values.get("default", ""))
                qt_widget_setter_func = qt_widget.setText
                widget_value_recover_func = lambda qt_widget: qt_widget.text()
            elif param_type == "notification":
                # A notification input (probably never going to happen)
                qt_widget = QLineEdit()
                qt_widget.setText(param_values.get("default", ""))
                qt_widget_setter_func = qt_widget.setText
                widget_value_recover_func = lambda qt_widget: qt_widget.text()
            elif param_type == "null":
                # Ignore Null parameters
                qt_widget = None
                qt_widget_setter_func = None
                widget_value_recover_func = lambda qt_widget: None
            else:
                # Numpy layers
                if param_type not in NAPARI_LAYER_TYPES:
                    qt_widget = None
                    self.layer_comboboxes[param_type] = []
                else:
                    qt_widget = QComboBox()
                    if param_type not in self.layer_comboboxes:
                        self.layer_comboboxes[param_type] = []
                    self.layer_comboboxes[param_type].append(qt_widget)
                qt_widget_setter_func = None
                widget_value_recover_func = lambda qt_widget: None

            if qt_widget is not None:
                self.layout.addWidget(QLabel(param_values.get("title")), k, 0)
                self.layout.addWidget(qt_widget, k, 1)

            state_item = UIStateItem(
                param_type=param_type,
                qt_widget=qt_widget,
                qt_widget_setter_func=qt_widget_setter_func,
                widget_value_recover_func=widget_value_recover_func,
            )

            self.ui_state[param_name] = state_item

    def get_algo_params(self) -> Dict:
        """Create a dictionary representation of parameter values based on the UI state."""
        algo_params = {}
        for name, state_item in self.ui_state.items():
            if state_item.param_type in NAPARI_LAYER_TYPES:
                if state_item.qt_widget.currentText():
                    data = state_item.qt_widget.currentData()
                else:
                    data = None
            else:
                data = state_item.widget_value_recover_func(state_item.qt_widget)
            algo_params[name] = data
        return algo_params

    def manage_cbs_events(self, worker):
        """Whenever a worker returns, we update the napari layer comboboxes to their current index (instead of resetting it)"""
        for cb_list in self.layer_comboboxes.values():
            for cb in cb_list:
                worker.returned.connect(lambda _: cb.setCurrentIndex(cb.currentIndex()))
