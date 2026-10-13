"""Tests for Simulation.view and Simulation.animate."""

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from numpy.testing import assert_allclose

from pylj.configuration import Configuration
from pylj.mc import MCSimulation
from pylj.md import MDSimulation
from pylj.sample._atoms import _atom_colours, _atom_diameters
from pylj.tests.argon import ARGON, ARGON_MODEL, LARGER, MIXTURE_MODEL

# Four atoms well inside a 20 Angstrom box, so none overhangs an edge.
MIDDLE = [[6.0, 6.0], [14.0, 6.0], [6.0, 14.0], [14.0, 14.0]]


def mixture() -> Configuration:
    """Four atoms that alternate between argon and the larger species."""
    return Configuration(np.array(MIDDLE), (ARGON, LARGER), np.array([0, 1, 0, 1]), 20.0)


def test_default_diameter_is_each_species_potential_minimum():
    # For Lennard-Jones the energy minimum is at 2^(1/6) sigma.
    argon, larger = 2 ** (1 / 6) * 3.372, 2 ** (1 / 6) * 5.0
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


def drawn_positions(fig):
    """Every disc drawn in the figure's first axes, sorted."""
    lines = fig.axes[0].lines
    return sorted(
        (x, y) for line in lines for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True)
    )


def drawn_and_true_diameter_px(ax, diameter):
    """The drawn width of the first line's discs and the width of ``diameter``, in pixels."""
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
    assert md.view().axes[0].get_title() == "0.15 ps"
    assert md.view(frame=0).axes[0].get_title() == "0.05 ps"
    assert md.view(frame=-1).axes[0].get_title() == "0.15 ps"
    mc = sampled_mc(2)
    assert mc.view().axes[0].get_title() == "Step 2"
    assert mc.view(frame=-1).axes[0].get_title() == "Frame 1"


def test_view_draws_periodic_images_of_atoms_at_the_edges():
    # With a 4 Angstrom drawn diameter: an atom at x = 0.5 overhangs the left
    # edge and is drawn again at x = 20.5; one in the corner is drawn four
    # times; one in the middle once.
    simulation = positioned([[0.5, 10.0], [0.5, 0.5], [10.0, 10.0]])
    expected = [(0.5, 0.5), (0.5, 10.0), (0.5, 20.5), (10.0, 10.0)]
    expected += [(20.5, 0.5), (20.5, 10.0), (20.5, 20.5)]
    assert_allclose(drawn_positions(simulation.view(diameter=4.0)), expected)


def test_view_draws_one_line_per_colour_and_diameter():
    simulation = positioned(MIDDLE)
    assert len(simulation.view().axes[0].lines) == 1
    fig = simulation.view(diameter=[3, 3, 3, 4], colour=["red", "red", "blue", "red"])
    looks = sorted(
        (line.get_markerfacecolor(), len(line.get_xdata())) for line in fig.axes[0].lines
    )
    assert looks == sorted([(to_rgba("red"), 2), (to_rgba("blue"), 1), (to_rgba("red"), 1)])


def test_view_discs_are_drawn_to_scale():
    fig = positioned(MIDDLE).view(diameter=4.0)
    fig.canvas.draw()
    assert_allclose(*drawn_and_true_diameter_px(fig.axes[0], 4.0))


def test_view_discs_stay_to_scale_when_the_figure_is_laid_out_again():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    positioned(MIDDLE).view(diameter=4.0, ax=axes[1])
    fig.canvas.draw()
    fig.subplots_adjust(wspace=0.8)
    fig.canvas.draw()
    assert_allclose(*drawn_and_true_diameter_px(axes[1], 4.0))
    plt.close(fig)


def test_view_draws_into_given_axes_and_returns_their_figure():
    fig, axes = plt.subplots(1, 3)
    assert positioned(MIDDLE).view(ax=axes[1]) is fig
    assert axes[1].lines
    assert not axes[0].lines
    plt.close(fig)


def test_view_figure_has_the_given_size_and_is_not_left_open():
    open_before = plt.get_fignums()
    fig = positioned(MIDDLE).view(figsize=(3, 5))
    assert_allclose(fig.get_size_inches(), [3, 5])
    assert plt.get_fignums() == open_before


def test_view_of_a_frame_needs_a_recorded_frame():
    with pytest.raises(ValueError, match="The trajectory has no frames"):
        positioned(MIDDLE).view(frame=0)


@pytest.mark.parametrize("frame", [2, -3])
def test_view_of_a_frame_outside_the_trajectory(frame):
    with pytest.raises(IndexError, match=f"Frame {frame} is outside the trajectory, which has 2"):
        sampled_mc(2).view(frame=frame)


def test_the_simulation_modules_do_not_import_matplotlib():
    code = (
        "import sys, pylj.simulation, pylj.md, pylj.mc, pylj.trajectory; "
        "sys.exit('matplotlib' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
