"""Individual plots that make up a viewer.

A pane draws one quantity into one matplotlib Axes. ``setup`` creates the
artists and static decoration once; ``update`` pushes the current state of the
simulation into those artists.
"""

import warnings
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from numpy.typing import NDArray

from pylj.configuration import Configuration
from pylj.mc import MCSimulation
from pylj.md import MDSimulation
from pylj.potentials import PairPotential
from pylj.simulation import Simulation

LINE_COLOUR = "#34a5daff"


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
    span = y_high - y_low
    if span <= 1e-6 * max(abs(y_high), abs(y_low)):
        # A series held constant by a thermostat spans only rounding error;
        # show it with a margin of one per cent of its value rather than
        # magnifying the noise.
        margin = 0.01 * abs(y_high) or 1.0
    else:
        margin = 0.05 * span
    ax.set_ylim(y_low if y_from_zero else y_low - margin, y_high + margin)
    ax.ticklabel_format(axis="y", useOffset=False)


class Pane:
    """One plot within a viewer.

    ``setup`` creates the artists; the viewer draws them by calling
    ``update``. A pane whose curve has a mean over the simulation's
    trajectory overrides ``average``.

    Attributes:
        needs_md: Whether this pane can only plot molecular dynamics samples.
            If any pane in a viewer sets this, the viewer refuses a Monte Carlo
            simulation.
    """

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

    def average(self, ax: Axes, simulation: Simulation) -> None:
        """Draw the mean over the simulation's trajectory, for panes that have one.

        A pane that draws a running series has nothing to average and leaves
        its axes alone.

        Args:
            ax: Axes this pane was set up in.
            simulation: The simulation being visualised.
        """


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
            "size the atoms by; pass diameter= to the pane or the viewer."
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
        return [_potential_minimum(simulation.model.potential(one, one)) for one in species]
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


def _with_periodic_images(
    position: NDArray[np.float64], box: float, radius: float
) -> NDArray[np.float64]:
    """Return the positions with a copy of each atom that overhangs an edge.

    An atom whose centre is within ``radius`` of an edge of the box is
    drawn again one box length away, so the part of its disc that hangs over
    the edge appears at the opposite edge, where it belongs.

    Args:
        position: The atom positions, shape ``(N, 2)``, in metres.
        box: The side length of the box, in metres.
        radius: The drawn radius of the atoms, in metres.

    Returns:
        The positions followed by the images, shape ``(N + images, 2)``.
    """
    images = [position]
    for shift_x in (-box, 0.0, box):
        for shift_y in (-box, 0.0, box):
            if shift_x == 0.0 and shift_y == 0.0:
                continue
            shifted = position + np.array([shift_x, shift_y])
            overhangs = np.all((shifted > -radius) & (shifted < box + radius), axis=1)
            images.append(shifted[overhangs])
    return np.concatenate(images)


class CellPane(Pane):
    """The atoms drawn to scale inside the simulation cell.

    Each species is drawn with its own marker. The drawn diameter is a
    display choice; by default it is the separation at the minimum of the
    species' own pair energy, which for a Lennard-Jones potential is
    2^(1/6) sigma. An atom that overhangs an edge of the box is drawn
    again at the opposite edge, since the box is periodic and that is where
    the overhanging part of it is.

    Args:
        diameter: Drawn diameter of the atoms, in Angstrom: one value
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
            drawn = _with_periodic_images(position, self.box, diameter / 2)
            line.set_data(drawn[:, 0], drawn[:, 1])
            line.set_markersize(diameter / self.box * axes_width_points)


class _SeriesPane(Pane):
    """A sampled quantity plotted against simulation time.

    Subclasses name the ``MDSamples`` attribute holding the samples and the
    y-axis label.

    Attributes:
        attribute: Name of the ``MDSamples`` attribute holding the sample
            array to plot on the y axis.
        ylabel: Label for the y axis.
        scale: Factor the sample values are multiplied by before plotting,
            to convert from SI to the unit named in ``ylabel``.
        y_from_zero: Whether the y axis should start at zero rather than
            below the minimum of the data.
    """

    needs_md = True
    attribute: str
    ylabel: str
    scale: float = 1.0
    y_from_zero: bool = False

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_ylabel(self.ylabel)
        ax.set_xlabel("Time / ps")

    def update(self, ax: Axes, simulation: Simulation) -> None:
        assert isinstance(simulation, MDSimulation)  # needs_md is set
        x = simulation.samples.step * simulation.timestep * 1e12
        y = getattr(simulation.samples, self.attribute) * self.scale
        ax.lines[0].set_data(x, y)
        _fit_axes(ax, x, y, y_from_zero=self.y_from_zero)


class TemperaturePane(_SeriesPane):
    """Instantaneous temperature against time."""

    attribute = "temperature"
    ylabel = "Temperature / K"


class PressurePane(_SeriesPane):
    """Instantaneous two-dimensional pressure against time."""

    attribute = "pressure"
    ylabel = "Pressure / N m$^{-1}$"


class MSDPane(_SeriesPane):
    """Mean squared displacement against time."""

    attribute = "msd"
    ylabel = "MSD / Angstrom$^2$"
    scale = 1e20
    y_from_zero = True


def _energy_series(
    simulation: Simulation,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the x and y data of the energy pane.

    Args:
        simulation: The simulation being visualised.

    Returns:
        Time in picoseconds and the total energy for a molecular dynamics
        simulation; step and the potential energy for a Monte Carlo one.

    Raises:
        TypeError: If the simulation records no energy.
    """
    if isinstance(simulation, MDSimulation):
        time = simulation.samples.step * simulation.timestep * 1e12
        return time, simulation.samples.total_energy
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
        ax.set_ylabel("Energy / J")
        xlabel = "Time / ps" if isinstance(simulation, MDSimulation) else "Step"
        ax.set_xlabel(xlabel)

    def update(self, ax: Axes, simulation: Simulation) -> None:
        x, y = _energy_series(simulation)
        ax.lines[0].set_data(x, y)
        _fit_axes(ax, x, y)


class RDFPane(Pane):
    """Radial distribution function.

    ``update`` draws g(r) of the current configuration and ``average`` draws
    it averaged over the frames the simulation has sampled.
    """

    BINS = 100

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_xlim(0, simulation.configuration.box / 2 * 1e10)
        ax.set_ylabel("g(r)")
        ax.set_xlabel("r / Angstrom")

    def update(self, ax: Axes, simulation: Simulation) -> None:
        r, gr = simulation.configuration.rdf(self.BINS)
        self._draw(ax, r, gr)

    def average(self, ax: Axes, simulation: Simulation) -> None:
        """Draw g(r) averaged over the trajectory.

        Leaves the curve alone before anything has been sampled.

        Args:
            ax: Axes this pane was set up in.
            simulation: The simulation being visualised.
        """
        if len(simulation.trajectory) == 0:
            return
        r, gr = simulation.trajectory.rdf(self.BINS)
        self._draw(ax, r, gr)

    @staticmethod
    def _draw(ax: Axes, r: NDArray[np.float64], gr: NDArray[np.float64]) -> None:
        if not gr.any():
            # g(r) is zero everywhere when there are no pairs to bin, as for
            # a single atom, and there is then no curve to draw.
            ax.lines[0].set_data([], [])
            return
        r = r * 1e10
        ax.lines[0].set_data(r, gr)
        _fit_axes(ax, r, gr, y_from_zero=True)


class ScatteringPane(Pane):
    """Scattering profile I(q) from the Debye sum over pair distances.

    The Debye sum for ``N`` identical scatterers is ``N`` from each atom
    scattering on its own, plus ``2 J0(q r)`` for each pair at distance
    ``r``, the two-dimensional form.

    ``update`` draws I(q) of the current configuration and ``average`` draws
    it averaged over the frames the simulation has sampled.
    """

    # An empirical upper limit, in 1/m, that shows the first few peaks for
    # argon-sized atoms.
    Q_MAX = 1e11
    POINTS = 1000
    SKIP = 20  # lowest-q points, where the box periodicity dominates

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        ax.plot([], [], color=LINE_COLOUR)
        ax.set_yticks([])
        ax.set_ylabel("I(q)")
        ax.set_xlabel("q / m$^{-1}$")

    def _q(self, configuration: Configuration) -> NDArray[np.float64]:
        """The q values to draw, in 1/m, from the box to ``Q_MAX``."""
        return np.linspace(2 * np.pi / configuration.box, self.Q_MAX, self.POINTS)[self.SKIP :]

    def update(self, ax: Axes, simulation: Simulation) -> None:
        q = self._q(simulation.configuration)
        self._draw(ax, q, simulation.configuration.scattering(q))

    def average(self, ax: Axes, simulation: Simulation) -> None:
        """Draw I(q) averaged over the trajectory.

        Leaves the curve alone before anything has been sampled.

        Args:
            ax: Axes this pane was set up in.
            simulation: The simulation being visualised.
        """
        if len(simulation.trajectory) == 0:
            return
        q = self._q(simulation.configuration)
        self._draw(ax, q, simulation.trajectory.scattering(q))

    @staticmethod
    def _draw(ax: Axes, q: NDArray[np.float64], intensity: NDArray[np.float64]) -> None:
        ax.lines[0].set_data(q, intensity)
        _fit_axes(ax, q, intensity, x_from_zero=False, y_from_zero=True)


class MaxwellBoltzmannPane(Pane):
    """Histogram of the speeds of every atom at every update so far.

    The histogram already pools the speeds of every update, so this pane has
    no separate average to show."""

    needs_md = True
    BINS = 25

    def __init__(self) -> None:
        self.speeds = np.array([])

    def setup(self, ax: Axes, simulation: Simulation) -> None:
        ax.step([], [], where="post", color=LINE_COLOUR)
        ax.set_ylabel("PDF")
        ax.set_xlabel("Speed / m s$^{-1}$")

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
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

    def update(self, ax: Axes, simulation: Simulation) -> None:
        ax.lines[0].set_data(self.x, self.y)
        _fit_axes(ax, self.x, self.y, x_from_zero=False)
