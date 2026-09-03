"""Tests for the visualisation package."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pylj import md
from pylj.sample import environment
from pylj.sample.panes import CellPane

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
