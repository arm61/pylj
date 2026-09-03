"""Individual plots that make up a viewer.

A pane draws one quantity into one matplotlib Axes. ``setup`` creates the
artists and static decoration once; ``update`` pushes the current state of the
system into those artists. Panes hold any history they accumulate across
updates.
"""

import warnings

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from pylj.constants import BOLTZMANN
from pylj.util import System

LINE_COLOUR = "#34a5daff"
LABEL_SIZE = 16


def _fit_axes(
    ax: Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    x_from_zero: bool = True,
    y_from_zero: bool = False,
) -> None:
    """Fit the axis limits to the data, leaving them alone when there is none.

    Args:
        ax: Axes to adjust.
        x: x data.
        y: y data.
        x_from_zero: Start the x axis at zero rather than at the first point.
        y_from_zero: Start the y axis at zero rather than below the minimum.

    Warns:
        RuntimeWarning: If the data holds a non-finite value, which limits
            cannot be fitted to.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        warnings.warn(
            "Non-finite values in the data; axis limits left unchanged. The "
            "simulation may have diverged.",
            RuntimeWarning,
            stacklevel=2,
        )
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

    ``setup`` creates the artists; the viewer draws them by calling
    ``update``. A pane that keeps every curve it draws sets
    ``keeps_history = True`` and overrides ``average``.

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

        Panes that keep a history of their updates override this to draw the
        mean of that history. Panes that keep no history do nothing, and
        report that by leaving ``keeps_history`` false.

        Args:
            ax: Axes this pane was set up in.
        """


def _require_md(pane: Pane, system: System) -> None:
    """Refuse to plot MD-only samples for a Monte Carlo system.

    Args:
        pane: The pane being set up, named in the message.
        system: The simulation being visualised.

    Raises:
        ValueError: If the system is not a molecular dynamics simulation.
    """
    if system.simulation != "md":
        raise ValueError(
            f"{type(pane).__name__} plots molecular dynamics samples, which a Monte "
            "Carlo system does not record. Use JustCell, Energy or RDF with a Monte "
            "Carlo system, or build the system with md.initialise."
        )


class _HistoryPane(Pane):
    """A pane that keeps every curve it has drawn so ``average`` can show the mean.

    Attributes:
        history: One entry per update, in the order they were drawn.
    """

    keeps_history = True

    def __init__(self) -> None:
        self.history: list[np.ndarray] = []

    def average(self, ax: Axes) -> None:
        """Replace the drawn curve with the mean of every update so far.

        Args:
            ax: Axes this pane was set up in.
        """
        raise NotImplementedError


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

    def update(self, ax: Axes, system: System) -> None:
        types = np.asarray(system.particles["types"])
        # Settle the axes box to the equal aspect before its width is read.
        ax.apply_aspect()
        # Marker sizes are in points, and there are 72 points to the inch.
        axes_width_points = ax.get_window_extent().width / ax.figure.dpi * 72
        for index, (line, diameter) in enumerate(zip(ax.lines, system.diameters, strict=True)):
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
        _require_md(self, system)
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_ylabel(self.ylabel, fontsize=LABEL_SIZE)
        ax.set_xlabel("Time/s", fontsize=LABEL_SIZE)

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


class RDFPane(_HistoryPane):
    """Radial distribution function of the current configuration.

    Keeps every g(r) it has drawn so ``average`` can show the mean.
    """

    BINS = 100

    def __init__(self) -> None:
        super().__init__()
        self.r = np.array([])

    def setup(self, ax: Axes, system: System) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_xlim(0, system.box_length / 2)
        ax.set_yticks([])
        ax.set_ylabel("RDF", fontsize=LABEL_SIZE)
        ax.set_xlabel("r/m", fontsize=LABEL_SIZE)

    def update(self, ax: Axes, system: System) -> None:
        edges = np.linspace(0, system.box_length / 2, self.BINS + 1)
        dr = edges[1] - edges[0]
        r = edges[:-1] + dr / 2
        counts, _ = np.histogram(system.distances, bins=edges)
        n = system.number_of_particles
        pairs = n * (n - 1) / 2
        # The ideal-gas count for the N(N - 1) / 2 pairs, spread evenly over
        # the box, in a 2D shell of area 2 pi r dr at radius r.
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


class ScatteringPane(_HistoryPane):
    """Scattering profile I(q) from the Debye sum over pair distances.

    Keeps every profile it has drawn so ``average`` can show the mean.
    """

    # An empirical upper limit, in 1/m, that shows the first few peaks for
    # argon-sized particles.
    Q_MAX = 1e11
    POINTS = 1000
    SKIP = 20  # lowest-q points, where the box periodicity dominates
    # q values per block; np.sinc allocates several temporaries of this size
    # times the pair count
    BLOCK = 16

    def __init__(self) -> None:
        super().__init__()
        self.q = np.array([])

    def setup(self, ax: Axes, system: System) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_yticks([])
        ax.set_ylabel("I(q)", fontsize=LABEL_SIZE)
        ax.set_xlabel("q/m$^{-1}$", fontsize=LABEL_SIZE)

    def update(self, ax: Axes, system: System) -> None:
        q = np.linspace(2 * np.pi / system.box_length, self.Q_MAX, self.POINTS)[self.SKIP :]
        intensity = np.empty_like(q)
        for start in range(0, q.size, self.BLOCK):
            block = q[start : start + self.BLOCK]
            qr = np.outer(block, system.distances)
            intensity[start : start + self.BLOCK] = np.sum(np.sinc(qr / np.pi), axis=1)
        # The Debye sum is truncated to a finite set of pairs, so it can come
        # out slightly negative; an intensity cannot be.
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
    """Histogram of the speeds of every particle at every update so far.

    It keeps no per-frame history, so it has no average.
    """

    BINS = 25

    def __init__(self) -> None:
        self.speeds = np.array([])

    def setup(self, ax: Axes, system: System) -> None:
        _require_md(self, system)
        ax.step([], [], where="post", color=LINE_COLOUR)
        ax.set_ylabel("PDF", fontsize=LABEL_SIZE)
        ax.set_xlabel("Speed/m s$^{-1}$", fontsize=LABEL_SIZE)

    def update(self, ax: Axes, system: System) -> None:
        particles = system.particles
        speeds = np.hypot(particles["xvelocity"], particles["yvelocity"])
        self.speeds = np.append(self.speeds, speeds)
        density, edges = np.histogram(self.speeds, bins=self.BINS, density=True)
        plateau = np.append(density, density[-1])
        ax.lines[0].set_data(edges, plateau)
        _fit_axes(ax, edges, plateau, y_from_zero=True)


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

        Raises:
            ValueError: If ``x`` and ``y`` do not have the same shape, or if
                either contains a value that is not finite.
        """
        x_data = np.atleast_1d(np.asarray(x, dtype=float))
        y_data = np.atleast_1d(np.asarray(y, dtype=float))
        if x_data.shape != y_data.shape:
            raise ValueError(
                "x and y must have the same shape, but they are "
                f"{x_data.shape} and {y_data.shape}"
            )
        if not (np.isfinite(x_data).all() and np.isfinite(y_data).all()):
            raise ValueError("x and y must contain only finite values")
        self.x = x_data
        self.y = y_data

    def setup(self, ax: Axes, system: System) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_xlabel(self.xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(self.ylabel, fontsize=LABEL_SIZE)

    def update(self, ax: Axes, system: System) -> None:
        ax.lines[0].set_data(self.x, self.y)
        _fit_axes(ax, self.x, self.y, x_from_zero=False)
