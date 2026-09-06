"""Individual plots that make up a viewer.

A pane draws one quantity into one matplotlib Axes. ``setup`` creates the
artists and static decoration once; ``update`` pushes the current state of the
simulation into those artists. Panes hold any history they accumulate across
updates.
"""

import warnings
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from pylj import pairwise
from pylj.mc import MCSimulation
from pylj.md import MDSimulation
from pylj.potentials import PairPotential
from pylj.simulation import Simulation

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
    """Fit the axis limits to the data, leaving the limits unchanged when there is no
    data to fit them to.

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
        needs_md: Whether this pane can only plot molecular dynamics samples.
            If any pane in a viewer sets this, the viewer refuses a Monte Carlo
            simulation.
    """

    keeps_history: bool = False
    needs_md: bool = False

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        """Create the artists and static decoration for this pane.

        Args:
            ax: Axes to draw into.
            simulation: The simulation being visualised.
        """
        raise NotImplementedError

    def update(self, ax: Axes, simulation: Simulation) -> None:
        """Push the current state of the simulation into the artists.

        Args:
            ax: Axes this pane was set up in.
            simulation: The simulation being visualised.
        """
        raise NotImplementedError

    def average(self, ax: Axes) -> None:
        """Show the average of every update so far, for panes that keep a history.

        Panes that keep a history of their updates override this to draw the
        mean of that history. A pane that keeps no history does nothing here,
        and says so by leaving ``keeps_history`` false.

        Args:
            ax: Axes this pane was set up in.
        """


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

        Subclasses leave the curve alone when ``history`` is empty, as there
        is then nothing to average.

        Args:
            ax: Axes this pane was set up in.
        """
        raise NotImplementedError


def _potential_minimum(potential: PairPotential) -> float:
    """Return the separation at the minimum of a pair potential, in metres.

    The energy is evaluated on a logarithmic grid of separations from 0.1 to 50
    Angstrom. The search starts at the highest energy on that grid and takes
    the lowest energy beyond it, so that a barrier at short range is stepped
    over rather than mistaken for the well. For a Lennard-Jones potential the
    minimum found is at 2^(1/6) sigma, and for a square well it is at the hard-
    core diameter. For a Buckingham potential the highest point is the barrier
    that separates the well from the fall to minus infinity at short range, and
    the minimum lies in the well beyond that barrier.

    Args:
        potential: The pair potential.

    Returns:
        The separation at the energy minimum, in metres.

    Raises:
        ValueError: If the energy has no minimum between the barrier and 50
            Angstrom, as for a purely repulsive potential.
    """
    r = np.logspace(-11, np.log10(5e-9), 4000)
    energy = np.asarray(potential.energies(r), dtype=float)
    barrier = int(np.argmax(energy))
    well = barrier + int(np.argmin(energy[barrier:]))
    if well == r.size - 1:
        raise ValueError(
            f"{type(potential).__name__} has no minimum between 0.1 and 50 Angstrom to "
            "size the particles by; pass diameter= to the pane or the viewer."
        )
    return float(r[well])


def _drawn_diameters(
    simulation: Simulation, diameter: float | Iterable[float] | None
) -> list[float]:
    """Return the drawn diameter of each species, in metres.

    Args:
        simulation: The simulation being visualised.
        diameter: The diameter to draw, in Angstrom. A single value is used for
            every species; a sequence gives one value per species, in order.
            ``None`` uses the separation at the minimum of each species' own
            pair energy.

    Returns:
        One diameter per species, in the order of
        ``simulation.configuration.species``.

    Raises:
        ValueError: If the number of diameters differs from the number of
            species, a diameter is not positive and finite, or a diameter is
            below 0.01. A value that small is almost certainly in metres,
            given where an Angstrom-sized diameter would fall.
    """
    species = simulation.configuration.species
    if diameter is None:
        return [
            _potential_minimum(pairwise.pair_potential(simulation.pair_potentials, one, one))
            for one in species
        ]
    if isinstance(diameter, Iterable):
        values = [float(d) for d in diameter]
    else:
        values = [float(diameter)] * len(species)
    if len(values) != len(species):
        raise ValueError(
            f"Expected {len(species)} diameters, one per species, but got {len(values)}"
        )
    for value in values:
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"Every diameter must be positive and finite, but got {value}")
        if value < 0.01:
            raise ValueError(
                f"The diameter is in Angstrom, and {value} looks like a value in metres. "
                "An Angstrom is 1e-10 metres."
            )
    return [value * 1e-10 for value in values]


class CellPane(Pane):
    """The particles drawn to scale inside the simulation cell.

    Each species is drawn with its own marker. The drawn diameter is a
    display choice; by default it is the separation at the minimum of the
    species' own pair energy, which for a Lennard-Jones potential is
    2^(1/6) sigma.

    Args:
        diameter: Drawn diameter of the particles, in Angstrom: one value
            for every species, or one per species in the order of
            ``Configuration.species``. Each value must be positive and at
            least 0.01, as smaller values are metres mistaken for Angstrom.

    Attributes:
        diameters: The drawn diameter of each species, in metres, set by
            ``setup``.
        box: The side length of the box the axes span, in metres, set by
            ``setup``.
    """

    def __init__(self, diameter: float | Iterable[float] | None = None) -> None:
        self.diameter = diameter
        self.diameters: list[float] = []
        self.box = 0.0

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        self.diameters = _drawn_diameters(simulation, self.diameter)
        self.box = simulation.configuration.box
        for _ in self.diameters:
            ax.plot([], [], "o", markeredgecolor="black")
        ax.set_xlim(0, self.box)
        ax.set_ylim(0, self.box)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def update(self, ax: Axes, simulation: Simulation) -> None:
        configuration = simulation.configuration
        # Settle the axes box to the equal aspect before its width is read.
        ax.apply_aspect()
        # Marker sizes are in points, and there are 72 points to the inch.
        axes_width_points = ax.get_window_extent().width / ax.figure.dpi * 72
        for index, diameter in enumerate(self.diameters):
            line = ax.lines[index]
            position = configuration.position[configuration.species_index == index]
            line.set_data(position[:, 0], position[:, 1])
            line.set_markersize(diameter / self.box * axes_width_points)


class _SeriesPane(Pane):
    """A sampled quantity plotted against simulation time.

    Subclasses name the ``MDSamples`` attribute holding the samples and the
    y-axis label.

    Attributes:
        attribute: Name of the ``MDSamples`` attribute holding the sample
            array to plot on the y axis.
        ylabel: Label for the y axis.
        y_from_zero: Whether the y axis should start at zero rather than
            below the minimum of the data.
    """

    needs_md = True
    attribute: str
    ylabel: str
    y_from_zero: bool = False

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_ylabel(self.ylabel, fontsize=LABEL_SIZE)
        ax.set_xlabel("Time/s", fontsize=LABEL_SIZE)

    def update(self, ax: Axes, simulation: Simulation) -> None:
        assert isinstance(simulation, MDSimulation)  # needs_md is set
        x = simulation.samples.step * simulation.timestep
        y = getattr(simulation.samples, self.attribute)
        ax.lines[0].set_data(x, y)
        _fit_axes(ax, x, y, y_from_zero=self.y_from_zero)


class TemperaturePane(_SeriesPane):
    """Instantaneous temperature against time."""

    attribute = "temperature"
    ylabel = "Temperature/K"


class PressurePane(_SeriesPane):
    """Instantaneous two-dimensional pressure against time."""

    attribute = "pressure"
    ylabel = "Pressure/N m$^{-1}$"


class MSDPane(_SeriesPane):
    """Mean squared displacement against time."""

    attribute = "msd"
    ylabel = "MSD/m$^2$"
    y_from_zero = True


def _energy_series(
    simulation: Simulation,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the x and y data of the energy pane.

    Args:
        simulation: The simulation being visualised.

    Returns:
        Time and the total energy for a molecular dynamics simulation; step
        and the potential energy for a Monte Carlo one.

    Raises:
        TypeError: If the simulation records no energy.
    """
    if isinstance(simulation, MDSimulation):
        return simulation.samples.step * simulation.timestep, simulation.samples.total_energy
    if isinstance(simulation, MCSimulation):
        return simulation.samples.step, simulation.samples.potential_energy
    raise TypeError(
        f"EnergyPane needs an MDSimulation or an MCSimulation, not {type(simulation).__name__}"
    )


class EnergyPane(Pane):
    """The energy of the system.

    For a molecular dynamics simulation this is the total energy, potential
    plus kinetic, against time. For a Monte Carlo simulation it is the
    potential energy against step; there is no kinetic energy to add.
    """

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_ylabel("Energy/J", fontsize=LABEL_SIZE)
        xlabel = "Time/s" if isinstance(simulation, MDSimulation) else "Step"
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)

    def update(self, ax: Axes, simulation: Simulation) -> None:
        x, y = _energy_series(simulation)
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

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_xlim(0, simulation.configuration.box / 2)
        ax.set_ylabel("RDF", fontsize=LABEL_SIZE)
        ax.set_xlabel("r/m", fontsize=LABEL_SIZE)

    def update(self, ax: Axes, simulation: Simulation) -> None:
        configuration = simulation.configuration
        box = configuration.box
        edges = np.linspace(0, box / 2, self.BINS + 1)
        dr = edges[1] - edges[0]
        r = edges[:-1] + dr / 2
        distance, _ = pairwise.dist(configuration.position, box)
        counts, _ = np.histogram(distance, bins=edges)
        n = configuration.number_of_particles
        pairs = n * (n - 1) / 2
        if pairs == 0:
            # A single particle has no pairs, and so no radial distribution
            # function to draw or to average.
            ax.lines[0].set_data([], [])
            return
        # The ideal-gas count for the N(N - 1) / 2 pairs, spread evenly over
        # the box, in a 2D shell of area 2 pi r dr at radius r.
        ideal = pairs * 2 * np.pi * r * dr / box**2
        gr = counts / ideal
        self.r = r
        self.history.append(gr)
        ax.lines[0].set_data(r, gr)
        _fit_axes(ax, r, gr, y_from_zero=True)

    def average(self, ax: Axes) -> None:
        """Replace the current g(r) with the mean of every update so far.

        Leaves the curve alone when nothing has been drawn to average.
        """
        if not self.history:
            return
        gr = np.mean(self.history, axis=0)
        ax.lines[0].set_data(self.r, gr)
        _fit_axes(ax, self.r, gr, y_from_zero=True)


class ScatteringPane(_HistoryPane):
    """Scattering profile I(q) from the Debye sum over pair distances.

    The Debye sum for ``N`` identical scatterers is ``N`` from each particle
    scattering on its own, plus ``2 sin(q r) / (q r)`` for each pair at
    distance ``r``.

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

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_yticks([])
        ax.set_ylabel("I(q)", fontsize=LABEL_SIZE)
        ax.set_xlabel("q/m$^{-1}$", fontsize=LABEL_SIZE)

    def update(self, ax: Axes, simulation: Simulation) -> None:
        configuration = simulation.configuration
        distance, _ = pairwise.dist(configuration.position, configuration.box)
        q = np.linspace(2 * np.pi / configuration.box, self.Q_MAX, self.POINTS)[self.SKIP :]
        intensity = np.empty_like(q)
        for start in range(0, q.size, self.BLOCK):
            block = q[start : start + self.BLOCK]
            qr = np.outer(block, distance)
            intensity[start : start + self.BLOCK] = np.sum(np.sinc(qr / np.pi), axis=1)
        intensity = configuration.number_of_particles + 2 * intensity
        self.q = q
        self.history.append(intensity)
        ax.lines[0].set_data(q, intensity)
        _fit_axes(ax, q, intensity, x_from_zero=False, y_from_zero=True)

    def average(self, ax: Axes) -> None:
        """Replace the current I(q) with the mean of every update so far.

        Leaves the curve alone when nothing has been drawn to average.
        """
        if not self.history:
            return
        intensity = np.mean(self.history, axis=0)
        ax.lines[0].set_data(self.q, intensity)
        _fit_axes(ax, self.q, intensity, x_from_zero=False, y_from_zero=True)


class MaxwellBoltzmannPane(Pane):
    """Histogram of the speeds of every particle at every update so far.

    The histogram already pools every update, so there is no separate history
    to average and this pane has no average to show."""

    needs_md = True
    BINS = 25

    def __init__(self) -> None:
        self.speeds = np.array([])

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        ax.step([], [], where="post", color=LINE_COLOUR)
        ax.set_ylabel("PDF", fontsize=LABEL_SIZE)
        ax.set_xlabel("Speed/m s$^{-1}$", fontsize=LABEL_SIZE)

    def update(self, ax: Axes, simulation: Simulation) -> None:
        assert isinstance(simulation, MDSimulation)  # needs_md is set
        speeds = np.linalg.norm(simulation.configuration.velocity, axis=1)
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
                f"x and y must have the same shape, but they are {x_data.shape} and {y_data.shape}"
            )
        if not (np.isfinite(x_data).all() and np.isfinite(y_data).all()):
            raise ValueError("x and y must contain only finite values")
        self.x = x_data
        self.y = y_data

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_xlabel(self.xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(self.ylabel, fontsize=LABEL_SIZE)

    def update(self, ax: Axes, simulation: Simulation) -> None:
        ax.lines[0].set_data(self.x, self.y)
        _fit_axes(ax, self.x, self.y, x_from_zero=False)
