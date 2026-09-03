"""Individual plots that make up a viewer.

A pane draws one quantity into one matplotlib Axes. ``setup`` creates the
artists and static decoration once; ``update`` pushes the current state of the
system into those artists. Panes hold any history they accumulate across
updates.
"""

import numpy as np
from matplotlib.axes import Axes

LINE_COLOUR = "#34a5daff"
BOLTZMANN = 1.3806e-23  # J / K
LABEL_SIZE = 16


def _fit_axes(
    ax: Axes, x, y, *, x_from_zero: bool = True, y_from_zero: bool = False
) -> None:
    """Fit the axis limits to the data, leaving them alone when there is none.

    Args:
        ax: Axes to adjust.
        x: x data.
        y: y data.
        x_from_zero: Start the x axis at zero rather than at the first point.
        y_from_zero: Start the y axis at zero rather than below the minimum.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return
    x_low = 0.0 if x_from_zero else float(x.min())
    if x.max() > x_low:
        ax.set_xlim(x_low, float(x.max()))
    y_low = 0.0 if y_from_zero else float(y.min())
    y_high = float(y.max())
    span = y_high - y_low if y_high > y_low else (abs(y_high) or 1.0)
    ax.set_ylim(y_low if y_from_zero else y_low - 0.05 * span, y_high + 0.05 * span)


class Pane:
    """One plot within a viewer."""

    def setup(self, ax: Axes, system) -> None:
        """Create the artists and static decoration for this pane.

        Args:
            ax: Axes to draw into.
            system: The simulation being visualised.
        """
        raise NotImplementedError

    def update(self, ax: Axes, system) -> None:
        """Push the current state of the system into the artists.

        Args:
            ax: Axes this pane was set up in.
            system: The simulation being visualised.
        """
        raise NotImplementedError


class CellPane(Pane):
    """The particles drawn to scale inside the simulation cell."""

    def setup(self, ax: Axes, system) -> None:
        for _ in system.diameters:
            ax.plot([], [], "o", markeredgecolor="black")
        ax.set_xlim(0, system.box_length)
        ax.set_ylim(0, system.box_length)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        self.update(ax, system)

    def update(self, ax: Axes, system) -> None:
        types = np.asarray(system.particles["types"])
        ax.apply_aspect()
        axes_width_points = ax.get_position().width * ax.figure.get_figwidth() * 72
        for index, (line, diameter) in enumerate(zip(ax.lines, system.diameters)):
            mask = types == str(index)
            line.set_data(
                system.particles["xposition"][mask], system.particles["yposition"][mask]
            )
            line.set_markersize(diameter / system.box_length * axes_width_points)
