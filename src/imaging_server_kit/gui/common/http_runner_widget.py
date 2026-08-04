from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QLineEdit,
    QGridLayout,
    QWidget,
)
import imaging_server_kit as sk
from imaging_server_kit.core.errors import AlgorithmServerError
from imaging_server_kit.remote.client import ServerRequestError
from imaging_server_kit.gui.common import RunnerWidget


class HttpRunnerWidget(RunnerWidget):
    def __init__(self, default_url="http://localhost:8000"):
        client = sk.Client()
        super().__init__(runner=client)

        # Layout and widget
        self.full_widget = QWidget()
        layout = QGridLayout()
        self.full_widget.setLayout(layout)

        # Server URL
        layout.addWidget(QLabel("Server URL"), 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.server_url_field = QLineEdit()
        self.server_url_field.setText(default_url)
        layout.addWidget(self.server_url_field, 0, 1)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._connect_from_btn)
        layout.addWidget(self.connect_btn, 0, 2)

        # Add the base runner widget
        layout.addWidget(self._widget, 1, 0, 1, 3)

    @property
    def widget(self) -> QWidget:
        return self.full_widget

    def _connect_from_btn(self):
        self.cb_algorithms.clear()
        server_url = self.server_url_field.text()

        try:
            self.runner.connect(server_url)
        except (ServerRequestError, AlgorithmServerError) as e:
            # TODO: could we `log` the error instead?
            print(f"Could not connect to server on {server_url}")

        self.cb_algorithms.addItems(self.runner.algorithms)
