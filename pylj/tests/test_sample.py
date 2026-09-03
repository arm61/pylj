"""Tests for the visualisation package."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pylj.sample import environment


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
