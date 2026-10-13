"""Pictures and animations of the atoms in the box, for Simulation.view and
Simulation.animate."""

import base64
import io
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
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType
from numpy.typing import NDArray
from tqdm.auto import tqdm

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
        diameter: ``None`` gives each species the separation at the minimum
            of its own pair energy. A single value applies to every atom, and
            a sequence gives one value per atom.

    Returns:
        One diameter per atom.

    Raises:
        ValueError: If a sequence does not have one value per atom, or a
            value is not positive and finite.
    """
    atoms = configuration.number_of_atoms
    if diameter is None:
        defaults = [_potential_minimum(model.potential(one, one)) for one in configuration.species]
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
        colour: ``None`` gives each species its own colour, the first from
            matplotlib's colour cycle for the first species and so on. A
            single matplotlib colour applies to every atom, and a sequence
            gives one colour per atom.

    Returns:
        One colour per atom, as red, green, blue and alpha.

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
    """Atoms of one colour and one diameter, drawn as discs to scale with the box.

    Args:
        diameter: Drawn diameter, in Angstrom.
        box: Side length of the box, in Angstrom.
        colour: Fill colour.
    """

    def __init__(self, diameter: float, box: float, colour: RGBA) -> None:
        super().__init__(
            [], [], linestyle="none", marker="o", markerfacecolor=colour, markeredgecolor="black"
        )
        self.diameter = diameter
        self.box = box

    def draw(self, renderer: RendererBase) -> None:
        axes = self.axes
        assert axes is not None  # discs are drawn only once added to an axes
        # A marker's size is in points, so it is set from the width of the
        # axes each time the discs are drawn, after the figure's layout has
        # fixed that width. There are 72 points to the inch.
        width_points = axes.bbox.width / axes.figure.dpi * 72
        self.set_markersize(self.diameter / self.box * width_points)
        super().draw(renderer)


class _AtomDrawing:
    """The atoms drawn to scale in the periodic box.

    Atoms that share a colour and a diameter are drawn as one line of
    markers. ``draw`` moves the markers to the positions of a configuration.

    Args:
        ax: Axes to draw into.
        box: Side length of the box, in Angstrom.
        diameters: Drawn diameter of each atom, in Angstrom.
        colours: Colour of each atom.

    Attributes:
        groups: For each line of markers, the indices of its atoms and the line.
    """

    def __init__(self, ax: Axes, box: float, diameters: list[float], colours: list[RGBA]) -> None:
        self.ax = ax
        self.box = box
        members: dict[tuple[RGBA, float], list[int]] = {}
        for atom, look in enumerate(zip(colours, diameters, strict=True)):
            members.setdefault(look, []).append(atom)
        self.groups: list[tuple[list[int], _Discs]] = []
        for (colour, diameter), atoms in members.items():
            discs = _Discs(diameter, box, colour)
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
            positions = configuration.positions[atoms]
            drawn = _with_periodic_images(positions, self.box, discs.diameter / 2)
            discs.set_data(drawn[:, 0], drawn[:, 1])
        self.ax.set_title(title, fontsize=12)


def _frame_index(count: int, frame: int) -> int:
    """Returns ``frame`` as an index from zero into a trajectory of ``count`` frames.

    Raises:
        IndexError: If the trajectory has no frame ``frame``.
    """
    frame = operator.index(frame)
    if not -count <= frame < count:
        raise IndexError(f"Frame {frame} is outside the trajectory, which has {count} frames")
    return frame % count


def _frame_title(times: NDArray[np.float64] | None, index: int) -> str:
    """Returns the title of a recorded frame: its time, or its number if it has no time."""
    if times is None:
        return f"Frame {index}"
    return f"{times[index]:g} ps"


def _current_title(simulation: Simulation) -> str:
    """Returns the title of the current configuration: the time, or the Monte Carlo step."""
    if isinstance(simulation, MDSimulation):
        return f"{simulation.time:g} ps"
    return f"Step {simulation.steps}"


def _new_figure(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Returns a new figure with one axes, which pyplot does not keep open.

    The figure is made through pyplot, which sets up how a notebook shows
    figures, and then closed, so pyplot does not show it again at the end of
    the cell.
    """
    fig, ax = plt.subplots(figsize=figsize, layout="tight")
    plt.close(fig)
    return fig, ax


def view(
    simulation: Simulation,
    frame: int | None,
    diameter: float | Iterable[float] | None,
    colour: ColorType | Sequence[ColorType] | None,
    figsize: tuple[float, float],
    ax: Axes | None,
) -> Figure | SubFigure:
    """Draws one configuration; see :meth:`pylj.simulation.Simulation.view`."""
    if frame is None:
        configuration = simulation.configuration
        title = _current_title(simulation)
    else:
        trajectory = simulation.trajectory
        trajectory._check_frames()
        index = _frame_index(len(trajectory), frame)
        configuration = trajectory[index]
        title = _frame_title(trajectory.times, index)
    if ax is None:
        _, ax = _new_figure(figsize)
    drawing = _AtomDrawing(
        ax,
        configuration.box,
        _atom_diameters(configuration, simulation.model, diameter),
        _atom_colours(configuration, colour),
    )
    drawing.draw(configuration, title)
    return ax.figure


#: How far above the estimated size an animation may grow while it is built:
#: later frames can be a little larger than the first.
_ESTIMATE_MARGIN = 1.1


def _frame_indices(trajectory: Trajectory, frames: slice | Iterable[int] | None) -> list[int]:
    """Returns the indices, from zero, of the frames ``frames`` selects.

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
    """Returns the size of the figure as one frame of an animation, in megabytes.

    This is the size matplotlib counts against ``animation.embed_limit``: the
    figure saved in the format ``animation.frame_format`` and encoded in base64.
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
    """Animates recorded frames; see :meth:`pylj.simulation.Simulation.animate`."""
    check_positive_finite("fps", fps)
    trajectory = simulation.trajectory
    indices = _frame_indices(trajectory, frames)
    times = trajectory.times
    first = trajectory[indices[0]]
    fig, ax = _new_figure(figsize)
    drawing = _AtomDrawing(
        ax,
        first.box,
        _atom_diameters(first, simulation.model, diameter),
        _atom_colours(first, colour),
    )
    with matplotlib.rc_context({"animation.frame_format": "svg", "svg.fonttype": "none"}):
        drawing.draw(first, _frame_title(times, indices[0]))
        estimate = len(indices) * _frame_megabytes(fig)
        limit = matplotlib.rcParams["animation.embed_limit"]
        if estimate > limit:
            raise ValueError(
                f"The animation of {len(indices)} frames would take about {estimate:.1f} MB, "
                f"more than matplotlib's animation.embed_limit of {limit:g} MB. Animate fewer "
                "frames, for example every second one with frames=slice(None, None, 2), or "
                "raise matplotlib.rcParams['animation.embed_limit'], in MB."
            )
        matplotlib.rcParams["animation.embed_limit"] = max(limit, _ESTIMATE_MARGIN * estimate)
        with tqdm(total=len(indices), desc="Drawing frames", disable=not progress) as bar:

            def draw_frame(index: int) -> tuple[()]:
                drawing.draw(trajectory[index], _frame_title(times, index))
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
    if player.count('"data:image/svg+xml;base64,') < len(indices):
        raise ValueError(
            "matplotlib dropped frames from the animation because it grew larger than "
            "animation.embed_limit. Animate fewer frames, for example every second one with "
            "frames=slice(None, None, 2), or raise matplotlib.rcParams['animation.embed_limit'], "
            "in MB."
        )
    return HTML(player)
