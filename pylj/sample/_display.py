"""Figure creation and display for the pylj viewers."""

from __future__ import annotations

import matplotlib.pyplot as plt
from IPython.display import display

FIGURE_SCALE = {"small": 2.0, "medium": 1.0, "large": 0.5}


class _NullHandle:
    """Stands in for the IPython display handle when no kernel is running."""

    def update(self, fig) -> None:
        """Do nothing; there is nowhere to send the figure."""


def environment(panes: int, size: str = "medium"):
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
    if size not in FIGURE_SCALE:
        raise ValueError(f"size must be 'small', 'medium' or 'large', not {size!r}")
    scale = FIGURE_SCALE[size]
    if panes == 1:
        fig, axes = plt.subplots(figsize=(4 / scale, 4 / scale))
    elif panes == 2:
        fig, axes = plt.subplots(1, 2, figsize=(8 / scale, 4 / scale))
    elif panes == 4:
        fig, axes = plt.subplots(2, 2, figsize=(8 / scale, 8 / scale))
    else:
        raise ValueError(f"The number of panes must be 1, 2 or 4, not {panes}")
    handle = display(fig, display_id=True)
    if handle is None:
        handle = _NullHandle()
    return fig, axes, handle
