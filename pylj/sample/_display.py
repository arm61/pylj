"""Figure creation and display for the pylj viewers."""

from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure

FIGURE_WIDTH = {"small": 2.0, "medium": 4.0, "large": 8.0}


class DisplayHandle(Protocol):
    """Anything that can be handed a figure to show."""

    def update(self, fig: Figure, /) -> None:
        """Show ``fig``, replacing whatever was shown before."""
        ...


class _NullHandle:
    """Stands in for the IPython display handle when no kernel is running."""

    def update(self, fig: Figure, /) -> None:
        """Do nothing; there is nowhere to send the figure."""


def _open_display(fig: Figure) -> DisplayHandle:
    """Return a handle that pushes ``fig`` to the notebook, or a no-op handle."""
    if get_ipython() is None:
        return _NullHandle()
    return display(fig, display_id=True)


def environment(panes: int, size: str = "medium") -> tuple[Figure, Axes | np.ndarray]:
    """Create the figure grid for a viewer.

    Args:
        panes: Number of plots: 1, 2 or 4.
        size: Overall figure size: 'small', 'medium' or 'large'.

    Returns:
        A tuple of the figure and its axes: a single Axes for one pane, a 1-D
        array for two and a 2-by-2 array for four.

    Raises:
        ValueError: If ``panes`` or ``size`` is not one of the allowed values.
    """
    if size not in FIGURE_WIDTH:
        raise ValueError(f"size must be 'small', 'medium' or 'large', not {size!r}")
    width = FIGURE_WIDTH[size]
    if panes == 1:
        fig, axes = plt.subplots(figsize=(width, width))
    elif panes == 2:
        fig, axes = plt.subplots(1, 2, figsize=(2 * width, width))
    elif panes == 4:
        fig, axes = plt.subplots(2, 2, figsize=(2 * width, 2 * width))
    else:
        raise ValueError(f"panes must be 1, 2 or 4, not {panes!r}")
    return fig, axes
