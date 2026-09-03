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


def test_cell_pane_marker_scales_with_box(drawing_display):
    sizes = []
    for box in (20, 40):
        system = md.initialise(4, 100, box, "square", diameter=4.0)
        fig, ax, handle = environment(1)
        CellPane().setup(ax, system)
        sizes.append(ax.lines[0].get_markersize())
        plt.close(fig)
    assert_allclose(sizes[0], 2 * sizes[1])
    assert sizes[0] > 0
