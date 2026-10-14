"""Draws the atoms in the box for Simulation.view and Simulation.animate."""

import base64
import io
import math
import operator
from collections.abc import Iterable, Sequence
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.colors import is_color_like, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType

from pylj.configuration import Configuration
from pylj.md import MDSimulation
from pylj.model import Model
from pylj.potentials import check_positive_finite
from pylj.sample.panes import _potential_minimum, _with_periodic_images
from pylj.simulation import Simulation
from pylj.trajectory import Trajectory

RGBA = tuple[float, float, float, float]


def _atom_diameters(
    configuration: Configuration, model: Model, diameter: float | Iterable[float] | None
) -> list[float]:
    """Returns the drawn diameter of each atom, in Angstrom.

    Args:
        configuration: The configuration to draw.
        model: The model, which gives each species its default diameter.
        diameter: ``None`` draws an atom with a diameter equal to half the
            separation at which two atoms of its species have their lowest
            pair energy. A single value applies to every atom. A sequence
            gives one value per atom.

    Returns:
        One diameter per atom.

    Raises:
        ValueError: If a sequence does not have one value per atom, or a
            value is not positive and finite.
    """
    atoms = configuration.number_of_atoms
    if diameter is None:
        defaults = [
            _potential_minimum(model.potential(one, one)) / 2 for one in configuration.species
        ]
        return [defaults[index] for index in configuration.species_index]
    if isinstance(diameter, Iterable):
        values = [float(value) for value in diameter]
        if len(values) != atoms:
            raise ValueError(f"diameter has {len(values)} values, but there are {atoms} atoms")
    else:
        values = [float(diameter)] * atoms
    for value in values:
        check_positive_finite("diameter", value)
    return values


def _atom_colours(
    configuration: Configuration, colour: ColorType | Sequence[ColorType] | None
) -> list[RGBA]:
    """Returns the colour of each atom.

    Args:
        configuration: The configuration to draw.
        colour: ``None`` gives each species its own colour. The first
            species takes the first colour in matplotlib's colour cycle, the
            second species takes the second colour, and so on. A single
            matplotlib colour applies to every atom. A sequence gives one
            colour per atom.

    Returns:
        One colour per atom, as a tuple of red, green, blue and alpha values.

    Raises:
        ValueError: If a colour is not one matplotlib recognises, or a
            sequence does not have one colour per atom.
    """
    atoms = configuration.number_of_atoms
    if colour is None:
        return [to_rgba(f"C{index}") for index in configuration.species_index]
    if isinstance(colour, str) or is_color_like(colour):
        return [to_rgba(cast(ColorType, colour))] * atoms
    values = [to_rgba(one) for one in cast(Sequence[ColorType], colour)]
    if len(values) != atoms:
        raise ValueError(f"colour has {len(values)} values, but there are {atoms} atoms")
    return values


class _Discs(Line2D):
    """A line of markers that draws atoms of one colour and one diameter as discs.

    The discs are drawn to scale with the axes.

    Args:
        diameter: Drawn diameter, in Angstrom.
        colour: Fill colour.
    """

    def __init__(self, diameter: float, colour: RGBA) -> None:
        super().__init__(
            [], [], linestyle="none", marker="o", markerfacecolor=colour, markeredgewidth=0
        )
        self.diameter = diameter

    def draw(self, renderer: RendererBase) -> None:
        axes = self.axes
        assert axes is not None  # matplotlib draws the discs only after they join an axes
        # matplotlib sets the size of a marker in points, not in Angstrom.
        # The size in points that matches the diameter depends on the limits
        # and the width of the axes. Both can change between one drawing of
        # the figure and the next. The marker size is therefore set here, each
        # time the discs are drawn. There are 72 points to the inch.
        origin, edge = axes.transData.transform([(0.0, 0.0), (self.diameter, 0.0)])
        self.set_markersize((edge[0] - origin[0]) / axes.figure.dpi * 72)
        super().draw(renderer)


class _AtomDrawing:
    """Draws the atoms to scale in the periodic box.

    Atoms that share a colour and a diameter are drawn as one line of
    markers. ``draw`` moves the markers to the positions of a configuration.

    Args:
        ax: Axes to draw into.
        box: Side length of the box, in Angstrom.
        diameters: Drawn diameter of each atom, in Angstrom.
        colours: Colour of each atom.

    Attributes:
        groups: One entry for each line of markers. Each entry holds the
            indices of the atoms in the line, and the line itself.
    """

    def __init__(self, ax: Axes, box: float, diameters: list[float], colours: list[RGBA]) -> None:
        self.ax = ax
        self.box = box
        members: dict[tuple[RGBA, float], list[int]] = {}
        for atom, look in enumerate(zip(colours, diameters, strict=True)):
            members.setdefault(look, []).append(atom)
        self.groups: list[tuple[list[int], _Discs]] = []
        for (colour, diameter), atoms in members.items():
            discs = _Discs(diameter, colour)
            ax.add_line(discs)
            self.groups.append((atoms, discs))
        ax.set_xlim(0, box)
        ax.set_ylim(0, box)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def draw(self, configuration: Configuration, title: str) -> None:
        """Moves the atoms to the positions in ``configuration`` and sets the title.

        An atom that overhangs an edge of the box is drawn again at the
        opposite edge.

        Args:
            configuration: The configuration to draw.
            title: The title of the axes.
        """
        for atoms, discs in self.groups:
            positions = configuration.positions[atoms] % self.box
            drawn = _with_periodic_images(positions, self.box, discs.diameter / 2)
            discs.set_data(drawn[:, 0], drawn[:, 1])
        self.ax.set_title(title, fontsize=12)


def _frame_index(count: int, frame: int) -> int:
    """Returns the index of frame ``frame`` in a trajectory of ``count`` frames.

    The index counts from zero. A negative ``frame`` counts back from the last
    frame.

    Raises:
        IndexError: If the trajectory has no frame ``frame``.
    """
    frame = operator.index(frame)
    if not -count <= frame < count:
        raise IndexError(f"Frame {frame} is outside the trajectory, which has {count} frames")
    return frame % count


def _frame_titles(trajectory: Trajectory) -> list[str]:
    """Returns the title of every frame in the trajectory."""
    times = trajectory.times
    if times is None:
        return [f"Frame {index}" for index in range(len(trajectory))]
    decimals = next(
        (d for d in range(6) if np.allclose(times, np.round(times, d), rtol=0, atol=1e-9)), 6
    )
    return [f"{time:.{decimals}f} ps" for time in times]


def _current_title(simulation: Simulation) -> str:
    """Returns the title of the current configuration.

    The title is the simulated time. A Monte Carlo simulation has no time, so
    the title is the step instead.
    """
    if isinstance(simulation, MDSimulation):
        return f"{simulation.time:g} ps"
    return f"Step {simulation.steps}"


def view(
    simulation: Simulation,
    frame: int | None,
    diameter: float | Iterable[float] | None,
    colour: ColorType | Sequence[ColorType] | None,
    figsize: tuple[float, float],
    ax: Axes | None,
) -> Axes:
    """Draws one configuration, as :meth:`pylj.simulation.Simulation.view` describes."""
    if frame is None:
        configuration = simulation.configuration
        title = _current_title(simulation)
    else:
        trajectory = simulation.trajectory
        trajectory._check_frames()
        index = _frame_index(len(trajectory), frame)
        configuration = trajectory[index]
        title = _frame_titles(trajectory)[index]
    diameters = _atom_diameters(configuration, simulation.model, diameter)
    colours = _atom_colours(configuration, colour)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="tight")
    drawing = _AtomDrawing(ax, configuration.box, diameters, colours)
    drawing.draw(configuration, title)
    return ax


#: The advice given when an animation is larger than ``animation.embed_limit``.
_FEWER_FRAMES = (
    "Choose fewer frames, for example every second frame with frames=slice(None, None, 2). "
    "Alternatively, set matplotlib.rcParams['animation.embed_limit'] to a larger number of MB."
)


def _frame_indices(trajectory: Trajectory, frames: slice | Iterable[int] | None) -> list[int]:
    """Returns the index of each frame that ``frames`` selects.

    The indices count from zero.

    Raises:
        ValueError: If the trajectory has no frames, or ``frames`` selects none.
        IndexError: If ``frames`` names a frame the trajectory does not have.
    """
    trajectory._check_frames()
    count = len(trajectory)
    if frames is None:
        return list(range(count))
    if isinstance(frames, slice):
        indices = list(range(count)[frames])
    else:
        indices = [_frame_index(count, frame) for frame in frames]
    if not indices:
        raise ValueError(f"frames={frames!r} selects none of the trajectory's {count} frames")
    return indices


def _frame_megabytes(fig: Figure) -> float:
    """Returns the size, in megabytes, that the figure takes as one frame of an animation.

    matplotlib saves each frame in the format ``animation.frame_format`` and
    encodes the saved frame in base64. matplotlib counts the size of the
    encoded frames against ``animation.embed_limit``.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format=matplotlib.rcParams["animation.frame_format"])
    return len(base64.encodebytes(buffer.getvalue())) / 2**20


def animate(
    simulation: Simulation,
    frames: slice | Iterable[int] | None,
    fps: float,
    diameter: float | Iterable[float] | None,
    colour: ColorType | Sequence[ColorType] | None,
    figsize: tuple[float, float],
    progress: bool,
) -> HTML:
    """Animates recorded frames, as :meth:`pylj.simulation.Simulation.animate` describes."""
    # Imported here so that view does not trigger tqdm's warning about ipywidgets.
    from tqdm.auto import tqdm

    check_positive_finite("fps", fps)
    trajectory = simulation.trajectory
    indices = _frame_indices(trajectory, frames)
    titles = _frame_titles(trajectory)
    first = trajectory[indices[0]]
    # A notebook shows every figure that pyplot holds at the end of the cell.
    # This figure is made without pyplot, so that the notebook shows only the
    # animation.
    fig = Figure(figsize=figsize, layout="tight")
    ax = fig.add_subplot()
    drawing = _AtomDrawing(
        ax,
        first.box,
        _atom_diameters(first, simulation.model, diameter),
        _atom_colours(first, colour),
    )
    with matplotlib.rc_context({"animation.frame_format": "svg", "svg.fonttype": "none"}):
        drawing.draw(first, titles[indices[0]])
        estimate = len(indices) * _frame_megabytes(fig)
        limit = matplotlib.rcParams["animation.embed_limit"]
        if estimate > limit:
            raise ValueError(
                f"The animation of {len(indices)} frames would take about {estimate:.1f} MB, "
                f"more than matplotlib's animation.embed_limit of {limit:g} MB. {_FEWER_FRAMES}"
            )
        # matplotlib drops every frame beyond animation.embed_limit while it
        # builds an animation. The limit is therefore removed for the build.
        # The finished animation is checked against the limit below.
        matplotlib.rcParams["animation.embed_limit"] = math.inf
        with tqdm(
            total=len(indices), desc="Drawing frames", disable=not progress, leave=False
        ) as bar:

            def draw_frame(index: int) -> tuple[()]:
                drawing.draw(trajectory[index], titles[index])
                bar.update()
                return ()

            animation = FuncAnimation(
                fig,
                draw_frame,
                frames=indices,
                init_func=tuple,
                interval=1000 / fps,
                cache_frame_data=False,
            )
            player = animation.to_jshtml()
    size = len(player) / 2**20
    if size > limit:
        raise ValueError(
            f"The animation of {len(indices)} frames takes {size:.1f} MB, more than "
            f"matplotlib's animation.embed_limit of {limit:g} MB. {_FEWER_FRAMES}"
        )
    return HTML(player)
