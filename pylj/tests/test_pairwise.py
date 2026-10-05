import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import pairwise
from pylj.constants import BOLTZMANN


class TestPairwise(unittest.TestCase):
    def test_dist_applies_the_minimum_image(self):
        # Pairs 1 Angstrom apart across the periodic boundary of a 10 Angstrom
        # box: the minimum image is 1 Angstrom, not the 9 Angstrom raw
        # separation, on either axis and in either direction. Pairs are
        # (0, 1), (0, 2), (1, 2) and separations are r_i - r_j.
        position = np.array([[0.5, 9.5], [9.5, 9.5], [0.5, 0.5]])
        dr, separation = pairwise.dist(position, 10.0)
        assert_almost_equal(dr, [1.0, 1.0, np.sqrt(2)])
        assert_almost_equal(separation[:, 0], [1.0, 0.0, -1.0])
        assert_almost_equal(separation[:, 1], [0.0, -1.0, -1.0])
        self.assertEqual(separation.shape, (3, 2))

    def test_minimum_image_wraps_each_component_to_the_nearest_copy(self):
        separation = np.array([[9.0, -6.0], [4.0, 0.0]])
        assert_almost_equal(
            pairwise.minimum_image(separation, 10.0), [[-1.0, 4.0], [4.0, 0.0]]
        )

    def test_species_pairs_yields_each_unordered_pair_once(self):
        # Atoms of species 1, 0, 1: pairs (0, 1), (0, 2), (1, 2) are
        # 1-0, 1-1 and 0-1, so the unordered pair (0, 1) covers the first
        # and the last. Species 0 has one atom, so no pair is 0-0. Starting
        # with species 1 checks that the species are taken in sorted order.
        pairs = list(pairwise.species_pairs(np.array([1, 0, 1])))
        self.assertEqual([(a, b) for _, a, b in pairs], [(0, 1), (1, 1)])
        assert_equal(pairs[0][0], [True, False, True])
        assert_equal(pairs[1][0], [False, True, False])

    def test_calculate_pressure(self):
        # The kinetic term K / L^2 plus the virial sum(f r) / (2 L^2).
        virial = -2.0
        kinetic = 2 * BOLTZMANN * 300
        p = pairwise.calculate_pressure(virial, 30.0, kinetic)
        assert_almost_equal(p, kinetic / 30.0**2 + virial / (2 * 30.0**2))

    def test_calculate_pressure_ideal_gas_limit(self):
        # With no pair forces the virial vanishes and the two-dimensional
        # pressure is the kinetic energy over the area: N k_B T / L^2 for N
        # atoms at temperature T in two dimensions.
        box = 25.0
        p = pairwise.calculate_pressure(0.0, box, 50 * BOLTZMANN * 200)
        assert_almost_equal(p, 50 * BOLTZMANN * 200 / box**2)
