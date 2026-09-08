import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import pairwise


class TestPairwise(unittest.TestCase):
    def test_dist_applies_the_minimum_image(self):
        # Pairs 1 Angstrom apart across the periodic boundary of a 10 Angstrom
        # box: the minimum image is 1 Angstrom, not the 9 Angstrom raw
        # separation, on either axis and in either direction. Pairs are
        # (0, 1), (0, 2), (1, 2) and separations are r_i - r_j.
        position = np.array([[0.5e-10, 9.5e-10], [9.5e-10, 9.5e-10], [0.5e-10, 0.5e-10]])
        dr, separation = pairwise.dist(position, 10e-10)
        assert_almost_equal(dr * 1e10, [1.0, 1.0, np.sqrt(2)])
        assert_almost_equal(separation[:, 0] * 1e10, [1.0, 0.0, -1.0])
        assert_almost_equal(separation[:, 1] * 1e10, [0.0, -1.0, -1.0])
        self.assertEqual(separation.shape, (3, 2))

    def test_minimum_image_wraps_each_component_to_the_nearest_copy(self):
        separation = np.array([[9.0e-10, -6.0e-10], [4.0e-10, 0.0]])
        assert_almost_equal(
            pairwise.minimum_image(separation, 10e-10) * 1e10, [[-1.0, 4.0], [4.0, 0.0]]
        )

    def test_species_pairs_yields_each_unordered_pair_once(self):
        # Atoms of species 0, 1, 0: pairs (0, 1), (0, 2), (1, 2) are
        # 0-1, 0-0 and 1-0, so the unordered pair (0, 1) covers the first
        # and the last.
        pairs = list(pairwise.species_pairs(np.array([0, 1, 0])))
        self.assertEqual([(a, b) for _, a, b in pairs], [(0, 0), (0, 1)])
        assert_equal(pairs[0][0], [False, True, False])
        assert_equal(pairs[1][0], [True, False, True])

    def test_calculate_pressure(self):
        # The kinetic term K / L^2 plus the virial sum(f r) / (2 L^2).
        virial = -9.5864009e-12 * 4e-10
        kinetic = 2 * 1.380649e-23 * 300
        p = pairwise.calculate_pressure(virial, 30e-10, kinetic)
        assert_almost_equal(p, kinetic / (30e-10) ** 2 + virial / (2 * (30e-10) ** 2))

    def test_calculate_pressure_ideal_gas_limit(self):
        # With no pair forces the virial vanishes and the two-dimensional
        # pressure is the kinetic energy over the area: N k_B T / L^2 for N
        # atoms at temperature T in two dimensions.
        box = 25e-10
        p = pairwise.calculate_pressure(0.0, box, 50 * 1.380649e-23 * 200)
        assert_almost_equal(p, 50 * 1.380649e-23 * 200 / box**2)
