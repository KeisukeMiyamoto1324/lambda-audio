import threading

import plotext as plt
from rich.ansi import AnsiDecoder
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.text import Text


class TrainingLossPlot:
    def __init__(self, height: int = 28) -> None:
        self.height = height
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.lock = threading.Lock()

    def update(self, step: int, loss: float) -> None:
        with self.lock:
            if self.steps and self.steps[-1] == step:
                self.losses[-1] = loss
                return
            self.steps.append(step)
            self.losses.append(loss)

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        with self.lock:
            if not self.losses:
                yield Text("Train Loss graph will appear after the first step.", style="dim")
                return

            # ---------------------------------------------------------
            # Build ANSI text at Rich's current width for in-place rendering.
            # ---------------------------------------------------------
            figure = plt.figure
            figure.clear()
            figure.plot_size(options.max_width or console.width, self.height)
            figure.theme("dark")
            figure.title(f"Train Loss — step {self.steps[-1]} — {self.losses[-1]:.4f}")
            figure.label("optimizer step", "x")
            figure.label("loss", "y")
            figure.draw(figure.signal(self.steps, self.losses, marker="braille").lines())
            plot = figure.build().string()

        yield Group(*AnsiDecoder().decode(plot))
