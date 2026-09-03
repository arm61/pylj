"""Viewers compose panes into one live figure."""

import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib.axes import Axes

from pylj.sample._display import environment
from pylj.sample.panes import (
    CellPane,
    CustomPane,
    EnergyPane,
    ForcePane,
    MaxwellBoltzmannPane,
    MSDPane,
    Pane,
    PressurePane,
    RDFPane,
    ScatteringPane,
    TemperaturePane,
)
from pylj.util import System


class Viewer:
    """A figure of one, two or four panes that redraws on demand.

    Args:
        system: The simulation to visualise.
        panes: The panes to show, in reading order across the grid.
        size: Figure size: 'small', 'medium' or 'large'.
    """

    def __init__(self, system: System, panes: list[Pane], size: str = "medium") -> None:
        self.panes = list(panes)
        self.fig, axes, self.handle = environment(len(self.panes), size)
        self.axes: list[Axes] = [axes] if isinstance(axes, Axes) else list(axes.ravel())
        try:
            for pane, ax in zip(self.panes, self.axes, strict=True):
                pane.setup(ax, system)
        except Exception:
            plt.close(self.fig)
            raise
        self.fig.tight_layout()
        self.update(system)
        plt.close(self.fig)

    def update(self, system: System) -> None:
        """Redraw every pane from the current state of the system.

        Args:
            system: The simulation to visualise.
        """
        for pane, ax in zip(self.panes, self.axes, strict=True):
            pane.update(ax, system)
        self.handle.update(self.fig)

    def average(self) -> None:
        """Show the average of every update so far on panes that keep one.

        Raises:
            ValueError: If no pane keeps a history.
        """
        if not any(pane.keeps_history for pane in self.panes):
            raise ValueError("None of this viewer's panes keeps a history to average")
        for pane, ax in zip(self.panes, self.axes, strict=True):
            pane.average(ax)
        self.handle.update(self.fig)


class JustCell(Viewer):
    """The particle positions only."""

    def __init__(self, system: System, size: str = "medium") -> None:
        super().__init__(system, [CellPane()], size)


class Energy(Viewer):
    """The particle positions and the total energy."""

    def __init__(self, system: System, size: str = "medium") -> None:
        super().__init__(system, [CellPane(), EnergyPane()], size)


class MaxBolt(Viewer):
    """The particle positions and a histogram of particle speeds."""

    def __init__(self, system: System, size: str = "medium") -> None:
        super().__init__(system, [CellPane(), MaxwellBoltzmannPane()], size)


class RDF(Viewer):
    """The particle positions and the radial distribution function."""

    def __init__(self, system: System, size: str = "medium") -> None:
        super().__init__(system, [CellPane(), RDFPane()], size)


class CellPlus(Viewer):
    """The particle positions and one plot of data supplied by the caller.

    Args:
        system: The simulation to visualise.
        xlabel: Label of the custom plot's x axis.
        ylabel: Label of the custom plot's y axis.
        size: Figure size: 'small', 'medium' or 'large'.
    """

    def __init__(self, system: System, xlabel: str, ylabel: str, size: str = "medium") -> None:
        self.custom = CustomPane(xlabel, ylabel)
        super().__init__(system, [CellPane(), self.custom], size)

    def update(
        self,
        system: System,
        xdata: npt.ArrayLike | None = None,
        ydata: npt.ArrayLike | None = None,
    ) -> None:
        """Redraw the cell and, if given, replace the custom plot's data.

        Args:
            system: The simulation to visualise.
            xdata: x values for the custom plot.
            ydata: y values for the custom plot.

        Raises:
            ValueError: If exactly one of ``xdata`` and ``ydata`` is given.
        """
        if (xdata is None) != (ydata is None):
            raise ValueError("xdata and ydata must be given together")
        if xdata is not None and ydata is not None:
            self.custom.set_data(xdata, ydata)
        super().update(system)


class Interactions(Viewer):
    """Positions, temperature, pressure and total force against time."""

    def __init__(self, system: System, size: str = "medium") -> None:
        panes = [CellPane(), TemperaturePane(), PressurePane(), ForcePane()]
        super().__init__(system, panes, size)


class Phase(Viewer):
    """Positions, total energy, mean squared displacement and g(r)."""

    def __init__(self, system: System, size: str = "medium") -> None:
        panes = [CellPane(), EnergyPane(), MSDPane(), RDFPane()]
        super().__init__(system, panes, size)


class Scattering(Viewer):
    """Positions, g(r), mean squared displacement and the scattering profile."""

    def __init__(self, system: System, size: str = "medium") -> None:
        panes = [CellPane(), RDFPane(), MSDPane(), ScatteringPane()]
        super().__init__(system, panes, size)
