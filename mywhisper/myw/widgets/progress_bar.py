from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Label, ProgressBar, Static


class PipelineProgressBar(Static):
    """
    Render pipeline progress and current step.
    """

    progress = reactive(0.0)
    step = reactive("")
    message = reactive("")

    def compose(self) -> ComposeResult:
        yield ProgressBar(total=100, show_eta=False, id="pipeline-progress-bar")
        yield Label("", id="pipeline-progress-label")

    def watch_progress(self, progress: float) -> None:
        bar = self.query_one("#pipeline-progress-bar", ProgressBar)
        bar.progress = int(progress * 100)

    def watch_step(self, step: str) -> None:
        self._update_label()

    def watch_message(self, message: str) -> None:
        self._update_label()

    def show(self, *, step: str, progress: float, message: str) -> None:
        self.visible = True
        self.step = step
        self.progress = progress
        self.message = message

    def hide(self) -> None:
        self.visible = False
        self.progress = 0.0
        self.step = ""
        self.message = ""
        self._update_label()

    def _update_label(self) -> None:
        label = self.query_one("#pipeline-progress-label", Label)
        if not self.visible:
            label.update("")
            return
        label.update(f"{self.step.title()} — {self.message}")

