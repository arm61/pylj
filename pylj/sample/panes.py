"""Individual plots that make up a viewer.

A pane draws one quantity into one matplotlib Axes. ``setup`` creates the
artists and static decoration once; ``update`` pushes the current state of the
system into those artists. Panes hold any history they accumulate across
updates.
"""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from pylj.constants import BOLTZMANN
from pylj.util import System

LINE_COLOUR = "#34a5daff"
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
    """One plot within a viewer.

    Panes that accumulate a history across updates set ``keeps_history`` and
    override ``average``.

    Attributes:
        keeps_history: Whether this pane accumulates a history across
            updates that ``average`` can summarise.
    """

    keeps_history: bool = False

    def setup(self, ax: Axes, system: System) -> None:
        """Create the artists and static decoration for this pane.

        Args:
            ax: Axes to draw into.
            system: The simulation being visualised.
        """
        raise NotImplementedError

    def update(self, ax: Axes, system: System) -> None:
        """Push the current state of the system into the artists.

        Args:
            ax: Axes this pane was set up in.
            system: The simulation being visualised.
        """
        raise NotImplementedError

    def average(self, ax: Axes) -> None:
        """Show the average of every update so far, for panes that keep one.

        Panes that keep a history of their updates (``RDFPane``,
        ``ScatteringPane``) override this. Other panes do nothing.

        Args:
            ax: Axes this pane was set up in.
        """


class CellPane(Pane):
    """The particles drawn to scale inside the simulation cell."""

    def setup(self, ax: Axes, system: System) -> None:
        for _ in system.diameters:
            ax.plot([], [], "o", markeredgecolor="black")
        ax.set_xlim(0, system.box_length)
        ax.set_ylim(0, system.box_length)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        self.update(ax, system)

    def update(self, ax: Axes, system: System) -> None:
        types = np.asarray(system.particles["types"])
        ax.apply_aspect()
        axes_width_points = ax.get_window_extent().width / ax.figure.dpi * 72
        for index, (line, diameter) in enumerate(zip(ax.lines, system.diameters)):
            mask = types == str(index)
            line.set_data(
                system.particles["xposition"][mask], system.particles["yposition"][mask]
            )
            line.set_markersize(diameter / system.box_length * axes_width_points)


class _SeriesPane(Pane):
    """A sampled quantity plotted against simulation time.

    Subclasses name the ``System`` attribute holding the samples and the
    y-axis label.

    Attributes:
        attribute: Name of the ``System`` attribute holding the sample array
            to plot on the y axis.
        ylabel: Label for the y axis.
        y_from_zero: Whether the y axis should start at zero rather than
            below the minimum of the data.
    """

    attribute: str
    ylabel: str
    y_from_zero: bool = False

    def setup(self, ax: Axes, system: System) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_ylabel(self.ylabel, fontsize=LABEL_SIZE)
        ax.set_xlabel("Time/s", fontsize=LABEL_SIZE)
        self.update(ax, system)

    def update(self, ax: Axes, system: System) -> None:
        x = system.step_sample * system.timestep_length
        y = getattr(system, self.attribute)
        ax.lines[0].set_data(x, y)
        _fit_axes(ax, x, y, y_from_zero=self.y_from_zero)


class TemperaturePane(_SeriesPane):
    """Instantaneous temperature against time."""

    attribute = "temperature_sample"
    ylabel = "Temperature/K"


class PressurePane(_SeriesPane):
    """Instantaneous two-dimensional pressure against time."""

    attribute = "pressure_sample"
    ylabel = "Pressure/N m$^{-1}$"


class ForcePane(_SeriesPane):
    """Sum of the pair forces against time."""

    attribute = "force_sample"
    ylabel = "Force/N"


class MSDPane(_SeriesPane):
    """Mean squared displacement against time."""

    attribute = "msd_sample"
    ylabel = "MSD/m$^2$"
    y_from_zero = True


class EnergyPane(Pane):
    """Total energy of the system.

    For an MD system this is the potential energy plus the kinetic energy
    ``N k_B T`` of ``N`` particles in two dimensions, against time. For an
    MC system it is the potential energy against step.
    """

    def setup(self, ax: Axes, system: System) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_ylabel("Energy/J", fontsize=LABEL_SIZE)
        xlabel = "Time/s" if system.simulation == "md" else "Step"
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)

    def update(self, ax: Axes, system: System) -> None:
        if system.simulation == "md":
            x = system.step_sample * system.timestep_length
            kinetic = system.number_of_particles * BOLTZMANN * system.temperature_sample
            y = system.energy_sample + kinetic
        else:
            x = system.step_sample
            y = system.energy_sample
        ax.lines[0].set_data(x, y)
        _fit_axes(ax, x, y)


class RDFPane(Pane):
    """Radial distribution function of the current configuration.

    Keeps every g(r) it has drawn so ``average`` can show the mean.
    """

    BINS = 100
    keeps_history = True

    def __init__(self) -> None:
        self.r = np.array([])
        self.history: list[np.ndarray] = []

    def setup(self, ax: Axes, system: System) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_xlim(0, system.box_length / 2)
        ax.set_yticks([])
        ax.set_ylabel("RDF", fontsize=LABEL_SIZE)
        ax.set_xlabel("r/m", fontsize=LABEL_SIZE)
        # History-keeping panes draw on the viewer's first update rather than
        # here, so setup does not itself contribute an entry to the history.

    def update(self, ax: Axes, system: System) -> None:
        edges = np.linspace(0, system.box_length / 2, self.BINS + 1)
        dr = edges[1] - edges[0]
        r = edges[:-1] + dr / 2
        counts, _ = np.histogram(system.distances, bins=edges)
        n = system.number_of_particles
        pairs = n * (n - 1) / 2
        ideal = pairs * 2 * np.pi * r * dr / system.box_length**2
        gr = counts / ideal
        self.r = r
        self.history.append(gr)
        ax.lines[0].set_data(r, gr)
        _fit_axes(ax, r, gr, y_from_zero=True)

    def average(self, ax: Axes) -> None:
        """Replace the current g(r) with the mean of every update so far."""
        gr = np.mean(self.history, axis=0)
        ax.lines[0].set_data(self.r, gr)
        _fit_axes(ax, self.r, gr, y_from_zero=True)


class ScatteringPane(Pane):
    """Scattering profile I(q) from the Debye sum over pair distances.

    Keeps every profile it has drawn so ``average`` can show the mean.
    """

    Q_MAX = 1e11  # 1/m; an empirical upper limit that shows the first few peaks for
    # argon-sized particles
    POINTS = 1000
    SKIP = 20  # lowest-q points, where the box periodicity dominates
    BLOCK = 64  # q values per block, bounding the q-by-pairs temporary
    keeps_history = True

    def __init__(self) -> None:
        self.q = np.array([])
        self.history: list[np.ndarray] = []

    def setup(self, ax: Axes, system: System) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_yticks([])
        ax.set_ylabel("I(q)", fontsize=LABEL_SIZE)
        ax.set_xlabel("q/m$^{-1}$", fontsize=LABEL_SIZE)
        # History-keeping panes draw on the viewer's first update rather than
        # here, so setup does not itself contribute an entry to the history.

    def update(self, ax: Axes, system: System) -> None:
        q = np.linspace(2 * np.pi / system.box_length, self.Q_MAX, self.POINTS)[self.SKIP :]
        intensity = np.empty_like(q)
        for start in range(0, q.size, self.BLOCK):
            block = q[start : start + self.BLOCK]
            qr = np.outer(block, system.distances)
            intensity[start : start + self.BLOCK] = np.sum(np.sinc(qr / np.pi), axis=1)
        intensity = np.clip(intensity, 0, None)
        self.q = q
        self.history.append(intensity)
        ax.lines[0].set_data(q, intensity)
        _fit_axes(ax, q, intensity, x_from_zero=False, y_from_zero=True)

    def average(self, ax: Axes) -> None:
        """Replace the current I(q) with the mean of every update so far."""
        intensity = np.mean(self.history, axis=0)
        ax.lines[0].set_data(self.q, intensity)
        _fit_axes(ax, self.q, intensity, x_from_zero=False, y_from_zero=True)


class MaxwellBoltzmannPane(Pane):
    """Histogram of particle speeds accumulated over every update."""

    BINS = 25

    def __init__(self) -> None:
        self.speeds = np.array([])

    def setup(self, ax: Axes, system: System) -> None:
        ax.step([], [], where="post", color=LINE_COLOUR)
        ax.set_ylabel("PDF", fontsize=LABEL_SIZE)
        ax.set_xlabel("Velocity/ms$^{-1}$", fontsize=LABEL_SIZE)

    def update(self, ax: Axes, system: System) -> None:
        particles = system.particles
        speeds = np.hypot(particles["xvelocity"], particles["yvelocity"])
        self.speeds = np.append(self.speeds, speeds)
        density, edges = np.histogram(self.speeds, bins=self.BINS, density=True)
        ax.lines[0].set_data(edges[:-1], density)
        _fit_axes(ax, edges[:-1], density, y_from_zero=True)


class CustomPane(Pane):
    """A line plot of data supplied by the caller through ``set_data``."""

    def __init__(self, xlabel: str, ylabel: str) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x = np.array([])
        self.y = np.array([])

    def set_data(self, x: npt.ArrayLike, y: npt.ArrayLike) -> None:
        """Store the data to draw on the next update.

        Args:
            x: x data to plot.
            y: y data to plot.
        """
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def setup(self, ax: Axes, system: System) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_xlabel(self.xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(self.ylabel, fontsize=LABEL_SIZE)

    def update(self, ax: Axes, system: System) -> None:
        ax.lines[0].set_data(self.x, self.y)
        _fit_axes(ax, self.x, self.y, x_from_zero=False)
