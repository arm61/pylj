"""Tests for the visualisation package."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pylj import mc, md
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN
from pylj.sample import environment
from pylj.sample.panes import (
    CellPane,
    CustomPane,
    EnergyPane,
    ForcePane,
    MaxwellBoltzmannPane,
    MSDPane,
    PressurePane,
    RDFPane,
    ScatteringPane,
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


def run_md_loop(system, steps: int, every: int):
    """Run an MD loop on the given system, sampling on every ``every``-th step."""
    for _ in range(steps):
        system.integrate(md.velocity_verlet)
        system.step += 1
        system.time += system.timestep_length
        if system.step % every == 0:
            system.md_sample()
    return system


def sampled_md_system(steps: int, every: int):
    """Initialise a fresh MD system and run it, sampling every ``every``-th step."""
    return run_md_loop(md.initialise(4, 100, 20, "square"), steps, every)


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
def test_time_panes_handle_empty_and_sparse_samples(drawing_display, pane_cls):
    system = md.initialise(4, 100, 20, "square")
    fig, ax, handle = environment(1)
    pane = pane_cls()
    pane.setup(ax, system)
    pane.update(ax, system)
    fig.canvas.draw()
    assert ax.lines[0].get_xdata().size == 0
    run_md_loop(system, steps=9, every=3)
    pane.update(ax, system)
    fig.canvas.draw()
    assert_allclose(ax.lines[0].get_xdata(), np.array([3, 6, 9]) * system.timestep_length)
    if pane_cls is not EnergyPane:
        assert_allclose(ax.lines[0].get_ydata(), getattr(system, pane_cls.attribute))
        assert ax.get_ylabel() == pane_cls.ylabel
    if pane_cls is MSDPane:
        assert ax.get_ylim()[0] == 0
    plt.close(fig)


def test_energy_pane_md_includes_kinetic_energy(drawing_display):
    system = sampled_md_system(steps=3, every=1)
    fig, ax, handle = environment(1)
    pane = EnergyPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    particles = system.particles
    mass_kg = system.mass * ATOMIC_MASS_UNIT
    kinetic = 0.5 * mass_kg * np.sum(particles["xvelocity"] ** 2 + particles["yvelocity"] ** 2)
    assert_allclose(ax.lines[0].get_ydata()[-1], system.energy_sample[-1] + kinetic)
    assert_allclose(
        ax.lines[0].get_ydata(),
        system.energy_sample + system.number_of_particles * BOLTZMANN * system.temperature_sample
    )
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


def test_rdf_pane_normalisation_is_unity_for_random_positions(drawing_display):
    np.random.seed(1)
    system = md.initialise(400, 100, 100, "random")
    fig, ax, handle = environment(1)
    pane = RDFPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    gr = ax.lines[0].get_ydata()
    assert abs(np.mean(gr[20:80]) - 1.0) < 0.1
    plt.close(fig)


def test_rdf_pane_average_is_mean_of_updates(drawing_display):
    fig, ax, handle = environment(1)
    pane = RDFPane()
    system = sampled_md_system(steps=1, every=1)
    pane.setup(ax, system)
    pane.update(ax, system)
    first = ax.lines[0].get_ydata().copy()
    system.integrate(md.velocity_verlet)
    pane.update(ax, system)
    second = ax.lines[0].get_ydata().copy()
    pane.average(ax)
    assert_allclose(ax.lines[0].get_ydata(), (first + second) / 2)
    plt.close(fig)


def test_scattering_pane_is_finite_and_non_negative(drawing_display):
    system = sampled_md_system(steps=1, every=1)
    fig, ax, handle = environment(1)
    pane = ScatteringPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    intensity = ax.lines[0].get_ydata()
    assert np.isfinite(intensity).all()
    assert (intensity >= 0).all()
    plt.close(fig)


def test_maxwell_boltzmann_pane_accumulates_speeds(drawing_display):
    system = md.initialise(4, 100, 20, "square")
    fig, ax, handle = environment(1)
    pane = MaxwellBoltzmannPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    pane.update(ax, system)
    assert pane.speeds.size == 8
    fig.canvas.draw()
    plt.close(fig)


def test_custom_pane_plots_supplied_data(drawing_display):
    system = md.initialise(4, 100, 20, "square")
    fig, ax, handle = environment(1)
    pane = CustomPane("x label", "y label")
    pane.setup(ax, system)
    pane.set_data([0, 1, 2], [1, 4, 9])
    pane.update(ax, system)
    assert_allclose(ax.lines[0].get_ydata(), [1, 4, 9])
    assert ax.get_xlabel() == "x label"
    plt.close(fig)
