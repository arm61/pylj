"""Tests for the visualisation package."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pylj import mc, md
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN
from pylj.sample import (
    RDF,
    CellPlus,
    Energy,
    Interactions,
    JustCell,
    MaxBolt,
    Phase,
    Scattering,
    Viewer,
    environment,
)
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
    _fit_axes,
)

NAMED_VIEWERS = [JustCell, Energy, MaxBolt, RDF, Interactions, Phase, Scattering]

TWO_TYPES = [[1.363e-134, 9.273e-78], [1.365e-130, 9.278e-77]]

SERIES_PANES = {
    TemperaturePane: ("temperature_sample", "Temperature/K"),
    PressurePane: ("pressure_sample", "Pressure/N m$^{-1}$"),
    ForcePane: ("force_sample", "Force/N"),
    MSDPane: ("msd_sample", "MSD/m$^2$"),
}


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


def test_environment_rejects_other_pane_counts():
    with pytest.raises(ValueError):
        environment(3)


def test_environment_rejects_unknown_size():
    with pytest.raises(ValueError):
        environment(1, size="huge")


@pytest.mark.parametrize("panes, shape", [(1, ()), (2, (2,)), (4, (2, 2))])
def test_environment_axes_shape(panes, shape):
    fig, axes = environment(panes)
    assert np.shape(axes) == shape
    plt.close(fig)


@pytest.mark.parametrize("size, width", [("small", 2.0), ("medium", 4.0), ("large", 8.0)])
def test_environment_figure_size(size, width):
    fig, axes = environment(1, size=size)
    assert fig.get_size_inches()[0] == width
    plt.close(fig)


def test_fit_axes_warns_on_non_finite_data_and_leaves_the_limits():
    fig, ax = environment(1)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 4)
    with pytest.warns(RuntimeWarning, match="Non-finite"):
        _fit_axes(ax, [0, 1], [0, np.nan])
    assert ax.get_xlim() == (0, 3)
    assert ax.get_ylim() == (0, 4)
    plt.close(fig)


def test_fit_axes_is_silent_on_empty_data():
    fig, ax = environment(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _fit_axes(ax, [], [])
    plt.close(fig)


def test_cell_pane_draws_each_type_separately():
    system = md.initialise(4, 100, 20, "square", constants=TWO_TYPES)
    fig, ax = environment(1)
    pane = CellPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    x = system.particles["xposition"]
    assert_allclose(ax.lines[0].get_xdata(), x[[0, 2]])
    assert_allclose(ax.lines[1].get_xdata(), x[[1, 3]])
    plt.close(fig)


@pytest.mark.parametrize("box_length", [20, 40])
def test_cell_pane_marker_matches_particle_diameter(box_length):
    system = md.initialise(4, 100, box_length, "square", diameter=4.0)
    fig, ax = environment(1)
    pane = CellPane()
    pane.setup(ax, system)
    pane.update(ax, system)

    def drawn_and_true_diameter_px():
        origin, edge = ax.transData.transform([(0, 0), (system.diameters[0], 0)])
        return ax.lines[0].get_markersize() * fig.dpi / 72, edge[0] - origin[0]

    assert_allclose(*drawn_and_true_diameter_px())
    fig.tight_layout()
    pane.update(ax, system)
    assert_allclose(*drawn_and_true_diameter_px())
    plt.close(fig)


@pytest.mark.parametrize("pane_cls", list(SERIES_PANES) + [EnergyPane])
def test_time_panes_handle_empty_and_sparse_samples(pane_cls):
    system = md.initialise(4, 100, 20, "square")
    fig, ax = environment(1)
    pane = pane_cls()
    pane.setup(ax, system)
    pane.update(ax, system)
    fig.canvas.draw()
    assert ax.lines[0].get_xdata().size == 0
    run_md_loop(system, steps=9, every=3)
    pane.update(ax, system)
    fig.canvas.draw()
    assert_allclose(ax.lines[0].get_xdata(), np.array([3, 6, 9]) * system.timestep_length)
    if pane_cls in SERIES_PANES:
        attribute, ylabel = SERIES_PANES[pane_cls]
        assert_allclose(ax.lines[0].get_ydata(), getattr(system, attribute))
        assert ax.get_ylabel() == ylabel
    if pane_cls is MSDPane:
        assert ax.get_ylim()[0] == 0
    plt.close(fig)


def test_energy_pane_md_includes_kinetic_energy():
    system = sampled_md_system(steps=3, every=1)
    fig, ax = environment(1)
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


def test_energy_pane_mc_plots_against_step():
    system = sampled_mc_system(steps=3)
    fig, ax = environment(1)
    pane = EnergyPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    assert_allclose(ax.lines[0].get_xdata(), [0, 1, 2, 3])
    assert_allclose(ax.lines[0].get_ydata(), system.energy_sample)
    assert ax.get_xlabel() == "Step"
    plt.close(fig)


def test_rdf_pane_normalisation_is_unity_for_random_positions():
    state = np.random.get_state()
    try:
        np.random.seed(1)
        system = md.initialise(400, 100, 100, "random")
    finally:
        np.random.set_state(state)
    fig, ax = environment(1)
    pane = RDFPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    gr = ax.lines[0].get_ydata()
    assert abs(np.mean(gr[20:80]) - 1.0) < 0.1
    plt.close(fig)


def test_rdf_pane_average_is_mean_of_updates():
    fig, ax = environment(1)
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


def test_rdf_pane_x_values_are_the_bin_centres():
    system = md.initialise(4, 100, 20, "square")
    fig, ax = environment(1)
    pane = RDFPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    dr = system.box_length / 2 / RDFPane.BINS
    r = ax.lines[0].get_xdata()
    assert r.size == RDFPane.BINS
    assert_allclose(r[0], dr / 2)
    assert_allclose(r, np.arange(RDFPane.BINS) * dr + dr / 2)
    plt.close(fig)


def test_scattering_pane_average_is_mean_of_updates():
    fig, ax = environment(1)
    pane = ScatteringPane()
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


def test_scattering_pane_is_finite_and_non_negative():
    system = sampled_md_system(steps=1, every=1)
    fig, ax = environment(1)
    pane = ScatteringPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    intensity = ax.lines[0].get_ydata()
    assert np.isfinite(intensity).all()
    assert (intensity >= 0).all()
    plt.close(fig)


def test_scattering_pane_matches_direct_debye_sum():
    system = md.initialise(4, 100, 20, "square")
    fig, ax = environment(1)
    pane = ScatteringPane()
    pane.setup(ax, system)
    pane.update(ax, system)

    q = np.linspace(2 * np.pi / system.box_length, ScatteringPane.Q_MAX, ScatteringPane.POINTS)
    q = q[ScatteringPane.SKIP:]
    r = system.distances
    expected = np.clip(np.array([np.sum(np.sin(qi * r) / (qi * r)) for qi in q]), 0, None)
    assert_allclose(ax.lines[0].get_ydata(), expected, rtol=1e-6)
    plt.close(fig)


def test_maxwell_boltzmann_pane_accumulates_speeds():
    system = md.initialise(4, 100, 20, "square")
    fig, ax = environment(1)
    pane = MaxwellBoltzmannPane()
    pane.setup(ax, system)

    system.particles["xvelocity"] = 100.0
    system.particles["yvelocity"] = 0.0
    pane.update(ax, system)

    system.particles["xvelocity"] = 300.0
    system.particles["yvelocity"] = 0.0
    pane.update(ax, system)

    x = ax.lines[0].get_xdata()
    assert x[0] <= 100
    assert x[-1] >= 300
    density = ax.lines[0].get_ydata()[:-1]
    bin_width = x[1] - x[0]
    assert np.sum(density * bin_width) == pytest.approx(1)
    fig.canvas.draw()
    plt.close(fig)


def test_maxwell_boltzmann_pane_draws_a_post_step_histogram():
    system = md.initialise(4, 100, 20, "square")
    system.particles["xvelocity"] = 100.0
    system.particles["yvelocity"] = 0.0
    fig, ax = environment(1)
    pane = MaxwellBoltzmannPane()
    pane.setup(ax, system)
    pane.update(ax, system)
    _, edges = np.histogram(np.full(4, 100.0), bins=MaxwellBoltzmannPane.BINS, density=True)
    assert_allclose(ax.lines[0].get_xdata()[0], edges[0])
    assert_allclose(ax.lines[0].get_xdata()[-1], edges[-1])
    assert ax.lines[0].get_drawstyle() == "steps-post"
    plt.close(fig)


def test_custom_pane_plots_supplied_data():
    system = md.initialise(4, 100, 20, "square")
    fig, ax = environment(1)
    pane = CustomPane("x label", "y label")
    pane.setup(ax, system)
    pane.set_data([0, 1, 2], [1, 4, 9])
    pane.update(ax, system)
    assert_allclose(ax.lines[0].get_ydata(), [1, 4, 9])
    assert ax.get_xlabel() == "x label"
    plt.close(fig)


def test_custom_pane_rejects_mismatched_data():
    pane = CustomPane("x label", "y label")
    with pytest.raises(ValueError, match=r"\(2, 2\) and \(4,\)"):
        pane.set_data([[0, 1], [2, 3]], [1, 4, 9, 16])


def test_custom_pane_takes_scalar_data():
    system = md.initialise(4, 100, 20, "square")
    fig, ax = environment(1)
    pane = CustomPane("x label", "y label")
    pane.setup(ax, system)
    pane.set_data(1.0, 2.0)
    pane.update(ax, system)
    fig.canvas.draw()
    assert_allclose(ax.lines[0].get_xdata(), [1.0])
    assert_allclose(ax.lines[0].get_ydata(), [2.0])
    plt.close(fig)


def test_custom_pane_rejects_non_finite_data():
    pane = CustomPane("x label", "y label")
    with pytest.raises(ValueError, match="finite"):
        pane.set_data([0, 1, 2], [1, 4, np.nan])


@pytest.mark.parametrize("viewer_cls", NAMED_VIEWERS)
def test_named_viewer_constructs_before_first_sample(drawing_display, viewer_cls):
    viewer_cls(md.initialise(4, 100, 20, "square"))
    assert drawing_display[0].updates == 0


@pytest.mark.parametrize("viewer_cls", NAMED_VIEWERS)
@pytest.mark.parametrize("every", [1, 3])
def test_named_viewer_updates_at_any_sampling_cadence(drawing_display, viewer_cls, every):
    system = md.initialise(4, 100, 20, "square")
    viewer = viewer_cls(system)
    for _ in range(6):
        system.integrate(md.velocity_verlet)
        system.step += 1
        system.time += system.timestep_length
        if system.step % every == 0:
            system.md_sample()
        viewer.update(system)
    assert drawing_display[0].updates == 6


@pytest.mark.parametrize("viewer_cls", [Interactions, Phase, Scattering])
def test_md_only_viewer_rejects_an_mc_system(drawing_display, viewer_cls):
    system = sampled_mc_system(steps=1)
    with pytest.raises(ValueError, match="Monte Carlo"):
        viewer_cls(system)


def test_speed_histogram_rejects_an_mc_system(drawing_display):
    system = sampled_mc_system(steps=1)
    with pytest.raises(ValueError, match="Monte Carlo"):
        MaxBolt(system)


def test_failed_viewer_setup_closes_its_figure(drawing_display):
    plt.close("all")
    system = sampled_mc_system(steps=1)
    with pytest.raises(ValueError):
        Interactions(system)
    assert plt.get_fignums() == []


def test_failed_first_draw_closes_the_figure_without_opening_a_display(drawing_display):
    class FailingPane(CellPane):
        def update(self, ax, system):
            raise RuntimeError("this pane cannot draw")

    plt.close("all")
    system = md.initialise(4, 100, 20, "square")
    with pytest.raises(RuntimeError, match="cannot draw"):
        Viewer(system, [FailingPane()])
    assert plt.get_fignums() == []
    assert drawing_display == []


def test_energy_viewer_on_mc_system(drawing_display):
    system = sampled_mc_system(steps=3)
    viewer = Energy(system)
    viewer.update(system)
    assert drawing_display[0].updates == 1


def test_rdf_viewer_average_shows_the_mean(drawing_display):
    system = md.initialise(20, 100, 20, "square")
    viewer = RDF(system)
    history = [viewer.axes[1].lines[0].get_ydata().copy()]
    for _ in range(3):
        system.integrate(md.velocity_verlet)
        system.md_sample()
        viewer.update(system)
        history.append(viewer.axes[1].lines[0].get_ydata().copy())
    viewer.average()
    assert_allclose(viewer.axes[1].lines[0].get_ydata(), np.mean(history, axis=0))


def test_average_is_available_before_any_update(drawing_display):
    RDF(md.initialise(20, 100, 20, "square")).average()


def test_average_rejects_viewers_without_history(drawing_display):
    with pytest.raises(ValueError):
        Energy(md.initialise(4, 100, 20, "square")).average()


def test_cell_plus_rejects_half_supplied_data(drawing_display):
    viewer = CellPlus(md.initialise(4, 100, 20, "square"), "x", "y")
    system = md.initialise(4, 100, 20, "square")
    with pytest.raises(ValueError):
        viewer.update(system, [0, 1, 2])


def test_cell_plus_rejects_y_data_without_x_data(drawing_display):
    viewer = CellPlus(md.initialise(4, 100, 20, "square"), "x", "y")
    system = md.initialise(4, 100, 20, "square")
    with pytest.raises(ValueError):
        viewer.update(system, ydata=[1, 2])


def test_cell_plus_takes_custom_data(drawing_display):
    system = md.initialise(4, 100, 20, "square")
    viewer = CellPlus(system, "x label", "y label")
    viewer.update(system, [0, 1, 2], [1, 4, 9])
    assert_allclose(viewer.axes[1].lines[0].get_ydata(), [1, 4, 9])


def test_viewer_forwards_size_to_environment(drawing_display):
    viewer = JustCell(md.initialise(4, 100, 20, "square"), size="small")
    assert viewer.fig.get_figwidth() == 2


def test_viewer_without_kernel(capsys):
    system = md.initialise(4, 100, 20, "square")
    viewer = JustCell(system)
    viewer.update(system)
    assert capsys.readouterr().out == ""
