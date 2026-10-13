"""Tests for Simulation.view and Simulation.animate."""

import base64
import os
import re
import subprocess
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from IPython.display import HTML
from matplotlib.colors import to_rgba
from numpy.testing import assert_allclose

from pylj.configuration import Configuration
from pylj.mc import MCSimulation
from pylj.md import MDSimulation
from pylj.model import Model
from pylj.potentials import PairPotential
from pylj.sample import _atoms
from pylj.sample._atoms import _atom_colours, _atom_diameters
from pylj.tests.argon import ARGON, ARGON_MODEL, LARGER, MIXTURE_MODEL

# Four atoms well inside a 20 Angstrom box, so none overhangs an edge.
MIDDLE = [[6.0, 6.0], [14.0, 6.0], [6.0, 14.0], [14.0, 14.0]]


def mixture() -> Configuration:
    """Four atoms that alternate between argon and the larger species."""
    return Configuration(np.array(MIDDLE), (ARGON, LARGER), np.array([0, 1, 0, 1]), 20.0)


def test_default_diameter_is_half_each_species_potential_minimum():
    # For Lennard-Jones the energy minimum is at 2^(1/6) sigma.
    argon, larger = 2 ** (1 / 6) * 3.372 / 2, 2 ** (1 / 6) * 5.0 / 2
    diameters = _atom_diameters(mixture(), MIXTURE_MODEL, None)
    assert_allclose(diameters, [argon, larger, argon, larger], rtol=2e-3)


def test_one_diameter_applies_to_every_atom():
    assert _atom_diameters(mixture(), MIXTURE_MODEL, 3.0) == [3.0, 3.0, 3.0, 3.0]


def test_a_diameter_sequence_gives_one_per_atom():
    assert _atom_diameters(mixture(), MIXTURE_MODEL, [1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]


def test_a_diameter_sequence_must_have_one_value_per_atom():
    with pytest.raises(ValueError, match="diameter has 2 values, but there are 4 atoms"):
        _atom_diameters(mixture(), MIXTURE_MODEL, [3.0, 4.0])


@pytest.mark.parametrize("diameter", [0.0, -1.0, np.inf, np.nan, [3.0, 3.0, 0.0, 3.0]])
def test_a_diameter_must_be_positive_and_finite(diameter):
    with pytest.raises(ValueError, match="diameter must be positive and finite"):
        _atom_diameters(mixture(), MIXTURE_MODEL, diameter)


def test_default_colour_is_one_per_species():
    first, second = to_rgba("C0"), to_rgba("C1")
    assert _atom_colours(mixture(), None) == [first, second, first, second]


@pytest.mark.parametrize("colour", ["red", "#ff0000", (1.0, 0.0, 0.0)])
def test_one_colour_applies_to_every_atom(colour):
    assert _atom_colours(mixture(), colour) == [to_rgba("red")] * 4


def test_a_colour_sequence_gives_one_per_atom():
    colours = ["red", "blue", "red", "green"]
    assert _atom_colours(mixture(), colours) == [to_rgba(one) for one in colours]


def test_a_colour_sequence_must_have_one_value_per_atom():
    with pytest.raises(ValueError, match="colour has 2 values, but there are 4 atoms"):
        _atom_colours(mixture(), ["red", "blue"])


def test_a_misspelt_colour_is_named_in_the_error():
    with pytest.raises(ValueError, match="'bleu'"):
        _atom_colours(mixture(), "bleu")


def positioned(positions) -> MCSimulation:
    """A Monte Carlo simulation of argon atoms at ``positions`` in a 20 Angstrom box."""
    positions = np.asarray(positions, dtype=float)
    species_index = np.zeros(len(positions), dtype=np.int64)
    configuration = Configuration(positions, (ARGON,), species_index, 20.0)
    return MCSimulation(configuration, ARGON_MODEL, 100)


def sampled_md(frames: int) -> MDSimulation:
    """A four-atom molecular dynamics run that has recorded one frame every 5 steps."""
    simulation = MDSimulation.initialise(
        ARGON_MODEL, number_of_atoms=4, temperature=100, box=20, seed=0
    )
    for _ in range(frames):
        for _ in range(5):
            simulation.step()
        simulation.sample()
    return simulation


def sampled_mc(frames: int) -> MCSimulation:
    """A four-atom Monte Carlo run that has recorded one frame after each step."""
    simulation = MCSimulation.initialise(
        ARGON_MODEL, number_of_atoms=4, temperature=100, box=20, seed=0
    )
    for _ in range(frames):
        simulation.step()
        simulation.sample()
    return simulation


@pytest.fixture(autouse=True)
def close_figures():
    """Closes the figures that view leaves open for pyplot to show."""
    yield
    plt.close("all")


def drawn_positions(ax):
    """Returns the position of every disc drawn in the axes, sorted."""
    return sorted(
        (x, y) for line in ax.lines for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True)
    )


def drawn_and_true_diameter_px(ax, diameter):
    """Returns the drawn width of a disc in the first line of markers, and the
    width of ``diameter`` in the axes, both in pixels."""
    origin, edge = ax.transData.transform([(0, 0), (diameter, 0)])
    return ax.lines[0].get_markersize() * ax.figure.dpi / 72, edge[0] - origin[0]


def test_view_draws_the_current_configuration_or_a_recorded_frame():
    simulation = positioned(MIDDLE)
    simulation.sample()
    moved = np.array(MIDDLE) + 1.0
    simulation.configuration.positions = moved
    assert_allclose(drawn_positions(simulation.view()), sorted(map(tuple, moved)))
    assert_allclose(drawn_positions(simulation.view(frame=0)), sorted(map(tuple, MIDDLE)))


def test_view_titles():
    md = sampled_md(3)
    assert md.view().get_title() == "0.15 ps"
    assert md.view(frame=0).get_title() == "0.05 ps"
    assert md.view(frame=-1).get_title() == "0.15 ps"
    mc = sampled_mc(2)
    assert mc.view().get_title() == "Step 2"
    assert mc.view(frame=-1).get_title() == "Frame 1"


def test_view_draws_periodic_images_of_atoms_at_the_edges():
    # The atoms are drawn 4 Angstrom across. The atom at x = 0.5 overhangs the
    # left edge, so it is drawn again at x = 20.5. The atom in the corner
    # overhangs two edges and is drawn four times. The atom in the middle is
    # drawn once.
    simulation = positioned([[0.5, 10.0], [0.5, 0.5], [10.0, 10.0]])
    expected = [(0.5, 0.5), (0.5, 10.0), (0.5, 20.5), (10.0, 10.0)]
    expected += [(20.5, 0.5), (20.5, 10.0), (20.5, 20.5)]
    assert_allclose(drawn_positions(simulation.view(diameter=4.0)), expected)


def test_view_draws_one_line_per_colour_and_diameter():
    assert len(positioned(MIDDLE).view().lines) == 1
    ax = positioned(MIDDLE).view(diameter=[3, 3, 3, 4], colour=["red", "blue", "blue", "blue"])
    ax.figure.canvas.draw()
    assert len(ax.lines) == 3
    (red,) = [line for line in ax.lines if line.get_markerfacecolor() == to_rgba("red")]
    assert_allclose(red.get_xydata(), [MIDDLE[0]])
    (large,) = [line for line in ax.lines if line.get_markersize() > red.get_markersize()]
    assert_allclose(large.get_xydata(), [MIDDLE[3]])
    assert large.get_markersize() == pytest.approx(red.get_markersize() * 4 / 3)


def test_view_discs_are_drawn_to_scale():
    ax = positioned(MIDDLE).view(diameter=4.0)
    ax.figure.canvas.draw()
    assert_allclose(*drawn_and_true_diameter_px(ax, 4.0))


def test_view_discs_stay_to_scale_when_the_figure_is_laid_out_again():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    positioned(MIDDLE).view(diameter=4.0, ax=axes[1])
    fig.canvas.draw()
    fig.subplots_adjust(wspace=0.8)
    fig.canvas.draw()
    assert_allclose(*drawn_and_true_diameter_px(axes[1], 4.0))


def test_view_discs_stay_to_scale_when_the_axes_are_zoomed():
    ax = positioned(MIDDLE).view(diameter=4.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.figure.canvas.draw()
    assert_allclose(*drawn_and_true_diameter_px(ax, 4.0))


def test_view_draws_into_given_axes_and_returns_them():
    _, axes = plt.subplots(1, 3)
    assert positioned(MIDDLE).view(ax=axes[1]) is axes[1]
    assert axes[1].lines
    assert not axes[0].lines


def test_view_figure_has_the_given_size_and_is_left_open_for_pyplot_to_show():
    ax = positioned(MIDDLE).view(figsize=(3, 5))
    assert_allclose(ax.figure.get_size_inches(), [3, 5])
    assert ax.figure.number in plt.get_fignums()


def test_view_of_a_purely_repulsive_potential_asks_for_a_diameter():
    class Repulsive(PairPotential):
        def energies(self, dr):
            return (3.0 / np.asarray(dr, dtype=float)) ** 12

        def forces(self, dr):
            dr = np.asarray(dr, dtype=float)
            return 12 * (3.0 / dr) ** 12 / dr

    configuration = Configuration(np.array(MIDDLE), (ARGON,), np.zeros(4, dtype=np.int64), 20.0)
    simulation = MCSimulation(configuration, Model.single(ARGON, Repulsive()), 100)
    with pytest.raises(ValueError, match="Pass diameter= to choose the drawn size"):
        simulation.view()


def test_view_of_a_frame_needs_a_recorded_frame():
    with pytest.raises(ValueError, match="The trajectory has no frames"):
        positioned(MIDDLE).view(frame=0)


@pytest.mark.parametrize("frame", [2, -3])
def test_view_of_a_frame_outside_the_trajectory(frame):
    with pytest.raises(IndexError, match=f"Frame {frame} is outside the trajectory, which has 2"):
        sampled_mc(2).view(frame=frame)


def test_view_does_not_import_tqdm():
    code = (
        "import sys\n"
        "from pylj.md import MDSimulation\n"
        "from pylj.tests.argon import ARGON_MODEL\n"
        "MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20).view()\n"
        "sys.exit('tqdm' in sys.modules)\n"
    )
    agg = os.environ | {"MPLBACKEND": "Agg"}
    assert subprocess.run([sys.executable, "-c", code], env=agg).returncode == 0


def test_the_simulation_modules_do_not_import_matplotlib():
    code = (
        "import sys, pylj.simulation, pylj.md, pylj.mc, pylj.trajectory; "
        "sys.exit('matplotlib' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


FRAME = re.compile(r'frames\[\d+\] = "data:image/svg\+xml;base64,(.*?)"', re.DOTALL)


def frame_svgs(player):
    """Returns every frame of an animation as SVG text, in order."""
    return [
        base64.b64decode(encoded.replace("\\\n", "")).decode()
        for encoded in FRAME.findall(player.data)
    ]


def frame_titles(player):
    """Returns the title of every frame in an animation, in order."""
    return [re.search(r"<text[^>]*>([^<]*)</text>", svg).group(1) for svg in frame_svgs(player)]


def test_animate_draws_each_frame_at_its_own_positions():
    simulation = positioned([[5.0, 10.0]])
    simulation.sample()
    simulation.configuration.positions = np.array([[15.0, 10.0]])
    simulation.sample()
    first, second = (
        float(re.search(r'<use [^>]*x="([^"]+)"', svg).group(1))
        for svg in frame_svgs(simulation.animate(progress=False))
    )
    assert first < second


def test_animate_draws_atoms_in_the_given_colour():
    svgs = frame_svgs(sampled_md(3).animate(colour="red", progress=False))
    assert all("#ff0000" in svg for svg in svgs)


def test_animate_has_one_frame_per_recorded_frame_titled_with_its_time():
    player = sampled_md(3).animate(progress=False)
    assert isinstance(player, HTML)
    assert frame_titles(player) == ["0.05 ps", "0.10 ps", "0.15 ps"]


def test_animate_gives_every_time_the_decimal_places_its_frames_need():
    simulation = MDSimulation.initialise(
        ARGON_MODEL, number_of_atoms=4, temperature=100, box=20, seed=0
    )
    for _ in range(3):
        for _ in range(20):
            simulation.step()
        simulation.sample()
    assert frame_titles(simulation.animate(progress=False)) == ["0.2 ps", "0.4 ps", "0.6 ps"]


def test_animate_titles_untimed_frames_with_their_number():
    assert frame_titles(sampled_mc(3).animate(progress=False)) == ["Frame 0", "Frame 1", "Frame 2"]


@pytest.mark.parametrize(
    ("frames", "titles"),
    [
        ([3, 0, -1], ["0.20 ps", "0.05 ps", "0.20 ps"]),
        (slice(1, None, 2), ["0.10 ps", "0.20 ps"]),
        (range(2), ["0.05 ps", "0.10 ps"]),
    ],
)
def test_animate_plays_the_selected_frames_in_order(frames, titles):
    assert frame_titles(sampled_md(4).animate(frames, progress=False)) == titles


def test_animate_needs_a_recorded_frame():
    simulation = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
    with pytest.raises(ValueError, match="The trajectory has no frames"):
        simulation.animate()


def test_animate_refuses_a_selection_of_no_frames():
    with pytest.raises(ValueError, match="selects none of the trajectory's 3 frames"):
        sampled_md(3).animate(slice(5, None))


def test_animate_refuses_a_frame_outside_the_trajectory():
    with pytest.raises(IndexError, match="Frame 3 is outside the trajectory, which has 3 frames"):
        sampled_md(3).animate([0, 3])


@pytest.mark.parametrize("fps", [0, -1, np.inf, np.nan])
def test_animate_needs_a_positive_finite_frame_rate(fps):
    with pytest.raises(ValueError, match="fps must be positive and finite"):
        sampled_md(3).animate(fps=fps)


def test_animate_refuses_an_animation_larger_than_the_embed_limit():
    with matplotlib.rc_context({"animation.embed_limit": 1e-4}):
        with pytest.raises(ValueError, match="more than matplotlib's animation.embed_limit"):
            sampled_md(3).animate(progress=False)


def test_animate_keeps_every_frame_while_it_builds_an_animation_too_large(monkeypatch):
    # Each frame of 120 atoms is more than twice as large as the rest of the
    # player. The limit is set to the rest of the player plus one and a half
    # frames. If matplotlib built the player under that limit, it would keep one
    # frame and drop the others. The shortened player would then fit within the
    # limit.
    simulation = MDSimulation.initialise(
        ARGON_MODEL, number_of_atoms=120, temperature=100, box=60, seed=0
    )
    for _ in range(3):
        simulation.sample()
    one = len(simulation.animate([0], progress=False).data)
    frame = len(simulation.animate([0, 0], progress=False).data) - one
    rest = one - frame
    assert frame > 2 * rest
    monkeypatch.setattr(_atoms, "_frame_megabytes", lambda fig: 0.0)
    with matplotlib.rc_context({"animation.embed_limit": (rest + 1.5 * frame) / 2**20}):
        with pytest.raises(ValueError, match="The animation of 3 frames takes"):
            simulation.animate(progress=False)


def test_animate_shows_progress_unless_asked_not_to(capsys):
    simulation = sampled_md(3)
    simulation.animate()
    assert "Drawing frames" in capsys.readouterr().err
    simulation.animate(progress=False)
    assert "Drawing frames" not in capsys.readouterr().err


def test_animate_leaves_no_figure_open_and_no_setting_changed():
    simulation = sampled_md(3)
    open_before = plt.get_fignums()
    settings_before = dict(matplotlib.rcParams)
    simulation.animate(progress=False)
    with matplotlib.rc_context({"animation.embed_limit": 1e-4}):
        limited = dict(matplotlib.rcParams)
        with pytest.raises(ValueError):
            simulation.animate(progress=False)
        assert dict(matplotlib.rcParams) == limited
    assert plt.get_fignums() == open_before
    assert dict(matplotlib.rcParams) == settings_before
