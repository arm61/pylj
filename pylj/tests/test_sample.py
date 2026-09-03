"""Tests for the visualisation package."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pylj import mc, md
from pylj.sample import environment
from pylj.sample.panes import (
    CellPane,
    EnergyPane,
    ForcePane,
    MSDPane,
    PressurePane,
    TemperaturePane,
)

TWO_TYPES = [[1.363e-134, 9.273e-78], [1.365e-130, 9.278e-77]]


def test_environment_rejects_other_pane_counts():
    with pytest.raises(ValueError):
        environment(3)


def test_environment_rejects_unknown_size():
    with pytest.raises(ValueError):
        environment(1, size="huge")


@pytest.mark.parametrize("panes, shape", [(1, ()), (2, (2,)), (4, (2, 2))])
def test_environment_axes_shape(drawing_display, panes, shape):
    fig, axes, handle = environment(panes)
    assert np.shape(axes) == shape
    plt.close(fig)


def test_environment_without_kernel(capsys):
    fig, axes, handle = environment(1)
    handle.update(fig)
    assert capsys.readouterr().out == ""
    plt.close(fig)


@pytest.mark.parametrize("size, width", [("small", 2.0), ("medium", 4.0), ("large", 8.0)])
def test_environment_figure_size(drawing_display, size, width):
    fig, axes, handle = environment(1, size=size)
    assert fig.get_size_inches()[0] == width
    plt.close(fig)


def test_cell_pane_draws_each_type_separately(drawing_display):
    system = md.initialise(4, 100, 20, "square", constants=TWO_TYPES)
    fig, ax, handle = environment(1)
    CellPane().setup(ax, system)
    x = system.particles["xposition"]
    assert_allclose(ax.lines[0].get_xdata(), x[[0, 2]])
    assert_allclose(ax.lines[1].get_xdata(), x[[1, 3]])
    plt.close(fig)


def test_cell_pane_marker_matches_particle_diameter(drawing_display):
    system = md.initialise(4, 100, 20, "square", diameter=4.0)
    fig, ax, handle = environment(1)
    pane = CellPane()
    pane.setup(ax, system)

    def drawn_and_true_diameter_px():
        origin, edge = ax.transData.transform([(0, 0), (system.diameters[0], 0)])
        return ax.lines[0].get_markersize() * fig.dpi / 72, edge[0] - origin[0]

    assert_allclose(*drawn_and_true_diameter_px())
    fig.tight_layout()
    pane.update(ax, system)
    assert_allclose(*drawn_and_true_diameter_px())
    plt.close(fig)


def sampled_md_system(steps: int, every: int):
    """Run an MD loop, sampling on every ``every``-th step."""
    system = md.initialise(4, 100, 20, "square")
    for _ in range(steps):
        system.integrate(md.velocity_verlet)
        system.step += 1
        system.time += system.timestep_length
        if system.step % every == 0:
            system.md_sample()
    return system


def sampled_mc_system(steps: int):
    """Run an MC loop that samples once per step, starting at step 0."""
    system = mc.initialise(4, 100, 20, "square")
    system.old_energy = system.energies.sum()
    system.mc_sample()
    for _ in range(steps):
        system.step += 1
        system.select_random_particle()
        system.new_random_position()
        system.compute_energy()
        system.new_energy = system.energies.sum()
        if mc.metropolis(100, system.old_energy, system.new_energy, n=0.5):
            system.accept()
        else:
            system.reject()
        system.mc_sample()
    return system


@pytest.mark.parametrize(
    "pane_cls", [TemperaturePane, PressurePane, ForcePane, MSDPane, EnergyPane]
)
def test_series_pane_handles_empty_and_sparse_samples(drawing_display, pane_cls):
    fig, ax, handle = environment(1)
    pane = pane_cls()
    pane.setup(ax, md.initialise(4, 100, 20, "square"))
    fig.canvas.draw()
    system = sampled_md_system(steps=9, every=3)
    pane.update(ax, system)
    fig.canvas.draw()
    assert_allclose(ax.lines[0].get_xdata(), np.array([3, 6, 9]) * system.timestep_length)
    plt.close(fig)


def test_energy_pane_md_includes_kinetic_energy(drawing_display):
    system = sampled_md_system(steps=3, every=1)
    fig, ax, handle = environment(1)
    pane = EnergyPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    expected = system.energy_sample + 4 * 1.3806e-23 * system.temperature_sample
    assert_allclose(ax.lines[0].get_ydata(), expected)
    assert ax.get_xlabel() == "Time/s"
    plt.close(fig)


def test_energy_pane_mc_plots_against_step(drawing_display):
    system = sampled_mc_system(steps=3)
    fig, ax, handle = environment(1)
    pane = EnergyPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    assert_allclose(ax.lines[0].get_xdata(), [0, 1, 2, 3])
    assert_allclose(ax.lines[0].get_ydata(), system.energy_sample)
    assert ax.get_xlabel() == "Step"
    plt.close(fig)
