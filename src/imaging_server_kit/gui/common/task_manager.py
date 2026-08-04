from typing import Callable
from napari.qt.threading import thread_worker, GeneratorWorker

from imaging_server_kit.gui.common.parameter_panel import ParameterPanel


class TaskManager:
    def __init__(
        self,
        grayout_ui: Callable,
        ungrayout_ui: Callable,
        parameters_panel: ParameterPanel,
    ):
        self.ungrayout_func = ungrayout_ui
        self.grayout_func = grayout_ui
        self.active_workers = []
        self.parameters_panel = parameters_panel

    def add_active(self, task: Callable, return_func: Callable):
        worker = thread_worker(task)()

        worker.returned.connect(return_func)
        worker.returned.connect(self._worker_stopped)
        worker.errored.connect(self._worker_errored)

        if isinstance(worker, GeneratorWorker):
            worker.yielded.connect(return_func)
            worker.aborted.connect(self._worker_stopped)

        self.parameters_panel.manage_cbs_events(worker)

        self.active_workers.append(worker)
        self.grayout_func()
        worker.start()

    def cancel_all(self):
        for worker in self.active_workers:
            worker.quit()

    def _worker_stopped(self):
        self.ungrayout_func()
        self.active_workers.clear()  # TODO: only stop the worker that finished?
    
    def _worker_errored(self, e: Exception):
        self._worker_stopped()