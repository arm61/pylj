"""Viewers compose panes into one live figure."""

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib.axes import Axes

from pylj.md import MDSimulation
from pylj.sample import _display
from pylj.sample._display import environment
from pylj.sample.panes import (
    CellPane,
    CustomPane,
    EnergyPane,
    MaxwellBoltzmannPane,
    MSDPane,
    Pane,
    PressurePane,
    RDFPane,
    ScatteringPane,
    TemperaturePane,
)
from pylj.simulation import Simulation


class Viewer:
    """A figure of one, two or four panes that redraws on demand.

    Args:
        simulation: The simulation to visualise.
        panes: The panes to show, in reading order across the grid.
        size: Figure size: 'small', 'medium' or 'large'.

    Raises:
        ValueError: If a pane plots molecular dynamics samples and the
            simulation is a Monte Carlo one, which records none.
    """

    def __init__(self, simulation: Simulation, panes: list[Pane], size: str = "medium") -> None:
        self.panes = list(panes)
        md_only = [type(pane).__name__ for pane in self.panes if pane.needs_md]
        if md_only and not isinstance(simulation, MDSimulation):
            raise ValueError(
                f"{type(self).__name__} plots molecular dynamics samples "
                f"({', '.join(md_only)}), which a Monte Carlo simulation does not record. "
                "Use JustCell, Energy, RDF or CellPlus with a Monte Carlo simulation, or "
                "build the simulation with MDSimulation.initialise."
            )
        self.fig, axes = environment(len(self.panes), size)
        self.axes: list[Axes] = [axes] if isinstance(axes, Axes) else list(axes.ravel())
        try:
            for pane, ax in zip(self.panes, self.axes, strict=True):
                pane.setup(ax, simulation)
            self.fig.tight_layout()
            for pane, ax in zip(self.panes, self.axes, strict=True):
                pane.update(ax, simulation)
        except BaseException:
            plt.close(self.fig)
            raise
        self.handle = _display._open_display(self.fig)
        plt.close(self.fig)

    def update(self, simulation: Simulation) -> None:
        """Redraw every pane from the current state of the simulation.

        Args:
            simulation: The simulation to visualise.
        """
        for pane, ax in zip(self.panes, self.axes, strict=True):
            pane.update(ax, simulation)
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
    """The atom positions only.

    Args:
        simulation: The simulation to visualise.
        size: Figure size: 'small', 'medium' or 'large'.
        diameter: Drawn diameter of the atoms, in Angstrom, one value or
            one per species; by default the separation at the minimum of
            each species' own pair energy.
    """

    def __init__(
        self,
        simulation: Simulation,
        size: str = "medium",
        diameter: float | Iterable[float] | None = None,
    ) -> None:
        super().__init__(simulation, [CellPane(diameter)], size)


class Energy(Viewer):
    """The atom positions and the energy: total for molecular dynamics,
    potential for Monte Carlo.

    Args:
        simulation: The simulation to visualise.
        size: Figure size: 'small', 'medium' or 'large'.
        diameter: Drawn diameter of the atoms, in Angstrom, one value or
            one per species; by default the separation at the minimum of
            each species' own pair energy.
    """

    def __init__(
        self,
        simulation: Simulation,
        size: str = "medium",
        diameter: float | Iterable[float] | None = None,
    ) -> None:
        super().__init__(simulation, [CellPane(diameter), EnergyPane()], size)


class MaxBolt(Viewer):
    """The atom positions and a histogram of atom speeds.

    Args:
        simulation: The simulation to visualise.
        size: Figure size: 'small', 'medium' or 'large'.
        diameter: Drawn diameter of the atoms, in Angstrom, one value or
            one per species; by default the separation at the minimum of
            each species' own pair energy.
    """

    def __init__(
        self,
        simulation: Simulation,
        size: str = "medium",
        diameter: float | Iterable[float] | None = None,
    ) -> None:
        super().__init__(simulation, [CellPane(diameter), MaxwellBoltzmannPane()], size)


class RDF(Viewer):
    """The atom positions and the radial distribution function.

    Args:
        simulation: The simulation to visualise.
        size: Figure size: 'small', 'medium' or 'large'.
        diameter: Drawn diameter of the atoms, in Angstrom, one value or
            one per species; by default the separation at the minimum of
            each species' own pair energy.
    """

    def __init__(
        self,
        simulation: Simulation,
        size: str = "medium",
        diameter: float | Iterable[float] | None = None,
    ) -> None:
        super().__init__(simulation, [CellPane(diameter), RDFPane()], size)


class CellPlus(Viewer):
    """The atom positions and one plot of data supplied by the caller.

    Args:
        simulation: The simulation to visualise.
        xlabel: Label of the custom plot's x axis.
        ylabel: Label of the custom plot's y axis.
        size: Figure size: 'small', 'medium' or 'large'.
        diameter: Drawn diameter of the atoms, in Angstrom, one value or
            one per species; by default the separation at the minimum of
            each species' own pair energy.
    """

    def __init__(
        self,
        simulation: Simulation,
        xlabel: str,
        ylabel: str,
        size: str = "medium",
        diameter: float | Iterable[float] | None = None,
    ) -> None:
        self.custom = CustomPane(xlabel, ylabel)
        super().__init__(simulation, [CellPane(diameter), self.custom], size)

    def update(
        self,
        simulation: Simulation,
        xdata: npt.ArrayLike | None = None,
        ydata: npt.ArrayLike | None = None,
    ) -> None:
        """Redraw the cell and, if given, replace the custom plot's data.

        Args:
            simulation: The simulation to visualise.
            xdata: x values for the custom plot.
            ydata: y values for the custom plot.

        Raises:
            ValueError: If exactly one of ``xdata`` and ``ydata`` is given.
        """
        if (xdata is None) != (ydata is None):
            raise ValueError("xdata and ydata must be given together")
        if xdata is not None and ydata is not None:
            self.custom.set_data(xdata, ydata)
        super().update(simulation)


class Interactions(Viewer):
    """Positions, temperature, pressure and total energy against time.

    Args:
        simulation: The simulation to visualise.
        size: Figure size: 'small', 'medium' or 'large'.
        diameter: Drawn diameter of the atoms, in Angstrom, one value or
            one per species; by default the separation at the minimum of
            each species' own pair energy.
    """

    def __init__(
        self,
        simulation: Simulation,
        size: str = "medium",
        diameter: float | Iterable[float] | None = None,
    ) -> None:
        panes = [CellPane(diameter), TemperaturePane(), PressurePane(), EnergyPane()]
        super().__init__(simulation, panes, size)


class Phase(Viewer):
    """Positions, total energy, mean squared displacement and g(r).

    Args:
        simulation: The simulation to visualise.
        size: Figure size: 'small', 'medium' or 'large'.
        diameter: Drawn diameter of the atoms, in Angstrom, one value or
            one per species; by default the separation at the minimum of
            each species' own pair energy.
    """

    def __init__(
        self,
        simulation: Simulation,
        size: str = "medium",
        diameter: float | Iterable[float] | None = None,
    ) -> None:
        panes = [CellPane(diameter), EnergyPane(), MSDPane(), RDFPane()]
        super().__init__(simulation, panes, size)


class Scattering(Viewer):
    """Positions, g(r), mean squared displacement and the scattering profile.

    Args:
        simulation: The simulation to visualise.
        size: Figure size: 'small', 'medium' or 'large'.
        diameter: Drawn diameter of the atoms, in Angstrom, one value or
            one per species; by default the separation at the minimum of
            each species' own pair energy.
    """

    def __init__(
        self,
        simulation: Simulation,
        size: str = "medium",
        diameter: float | Iterable[float] | None = None,
    ) -> None:
        panes = [CellPane(diameter), RDFPane(), MSDPane(), ScatteringPane()]
        super().__init__(simulation, panes, size)
