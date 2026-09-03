"""Figure creation and display for the pylj viewers."""

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from IPython.display import DisplayHandle, display
from matplotlib.axes import Axes
from matplotlib.figure import Figure

FIGURE_WIDTH = {"small": 2.0, "medium": 4.0, "large": 8.0}


class _NullHandle:
    """Stands in for the IPython display handle when no kernel is running."""

    def update(self, fig: Figure) -> None:
        """Do nothing; there is nowhere to send the figure."""


def _open_display(fig: Figure) -> DisplayHandle | _NullHandle:
    """Return a handle that pushes ``fig`` to the notebook, or a no-op handle."""
    if get_ipython() is None:
        return _NullHandle()
    return display(fig, display_id=True) or _NullHandle()


def environment(
    panes: int, size: str = "medium"
) -> tuple[Figure, Axes | np.ndarray, DisplayHandle | _NullHandle]:
    """Create the figure grid for a viewer and register it for display.

    Args:
        panes: Number of plots: 1, 2 or 4.
        size: Overall figure size: 'small', 'medium' or 'large'.

    Returns:
        A tuple of the figure, the axes (a single Axes for one pane, a 1-D
        array for two and a 2-by-2 array for four) and a display handle whose
        ``update(fig)`` pushes the figure to the notebook. Outside a Jupyter
        kernel the handle does nothing.

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
    handle = _open_display(fig)
    return fig, axes, handle
