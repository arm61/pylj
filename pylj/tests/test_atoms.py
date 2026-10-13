"""Tests for Simulation.view and Simulation.animate."""

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from numpy.testing import assert_allclose

from pylj.configuration import Configuration
from pylj.sample._atoms import _atom_colours, _atom_diameters
from pylj.tests.argon import ARGON, LARGER, MIXTURE_MODEL

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
