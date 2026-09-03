"""Shared fixtures for the pylj test suite."""

import matplotlib
import pytest

matplotlib.use("Agg")


class DrawingHandle:
    """Stand-in for the IPython display handle that renders on every update.

    Rendering makes matplotlib validate the data on every artist, so a viewer
    that would fail in a notebook fails here too.
    """

    def __init__(self) -> None:
        self.updates = 0

    def update(self, fig) -> None:
        fig.canvas.draw()
        self.updates += 1


@pytest.fixture
def drawing_display(monkeypatch):
    """Route viewer figures to DrawingHandle objects instead of IPython.

    Returns the list of handles created, in order, so tests can count updates.
    """
    handles: list[DrawingHandle] = []

    def fake_display(fig, display_id=True):
        handle = DrawingHandle()
        handles.append(handle)
        return handle

    monkeypatch.setattr("pylj.sample._display.display", fake_display)
    return handles
