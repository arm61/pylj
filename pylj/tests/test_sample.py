"""Tests for the visualisation package."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pylj.sample import environment


def test_environment_rejects_other_pane_counts(drawing_display):
    with pytest.raises(ValueError):
        environment(3)


def test_environment_rejects_unknown_size(drawing_display):
    with pytest.raises(ValueError):
        environment(1, size="huge")


@pytest.mark.parametrize("panes, shape", [(1, ()), (2, (2,)), (4, (2, 2))])
def test_environment_axes_shape(drawing_display, panes, shape):
    fig, axes, handle = environment(panes)
    assert np.shape(axes) == shape
    plt.close(fig)


def test_environment_without_kernel(monkeypatch):
    monkeypatch.setattr("pylj.sample._display.display", lambda fig, display_id=True: None)
    fig, axes, handle = environment(1)
    handle.update(fig)
    plt.close(fig)
