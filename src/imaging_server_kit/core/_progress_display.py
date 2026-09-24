"""Terminal progress bars (based on rich), driven by `Progress` layers.

A single live display is shared by all progress sources, with one bar per source (keyed by layer name),
so that e.g. tile progress and an algorithm's own progress are shown as independent bars.
"""

import atexit
from typing import Dict, Optional, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

_progress: Optional[Progress] = None
_tasks: Dict[str, TaskID] = {}
# Final (completed, total) of the bars when the display last stopped by itself.
# Progress layers can refresh several times with the same value; this avoids re-opening a finished bar.
_finished: Dict[str, Tuple[float, float]] = {}


def _live_display_supported() -> bool:
    """Rich live displays in Jupyter require ipywidgets."""
    if not Console().is_jupyter:
        return True
    try:
        import ipywidgets  # noqa: F401 - only checking availability
    except ImportError:
        return False
    return True


def update(key: str, completed: int, total: int) -> None:
    """Create or update the progress bar for `key`. The display stops once all bars are complete."""
    global _progress

    if not _live_display_supported():
        # Plain-text fallback (e.g. Jupyter without ipywidgets)
        print(f"\r{key}: {completed}/{total}", end="\n" if completed >= total else "")
        return

    if _progress is None:
        if _finished.get(key) == (completed, total):
            return
        _progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        _progress.start()

    task_id = _tasks.get(key)
    if task_id is None:
        task_id = _progress.add_task(key, total=total)
        _tasks[key] = task_id
    elif completed < _progress.tasks[task_id].completed:
        # The same source started over (e.g. an algorithm's progress in the next tile)
        _progress.reset(task_id)

    _progress.update(task_id, completed=completed, total=total)

    if all(task.finished for task in _progress.tasks):
        tasks_by_id = {task.id: task for task in _progress.tasks}
        finished = {
            name: (tasks_by_id[task_id].completed, tasks_by_id[task_id].total)
            for name, task_id in _tasks.items()
        }
        _stop_display()
        _finished.update(finished)


def stop() -> None:
    """Stop the live display, if any. Safe to call multiple times."""
    _stop_display()
    _finished.clear()


def _stop_display() -> None:
    global _progress

    if _progress is not None:
        _progress.stop()
    _progress = None
    _tasks.clear()


atexit.register(stop)
