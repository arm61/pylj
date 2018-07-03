from numpy.testing import assert_almost_equal, assert_equal
from pylj import pairwise, util
import unittest
import numpy as np

class TestPairwise(unittest.TestCase):
    def test_update_accelerations(self):
        part_dt = util.particle_dt()
        particles = np.zeros(2, dtype=part_dt)
        particles = pairwise.update_accelerations(particles, 1, 1, 1, 1, np.sqrt(2), 0, 1)
        assert_almost_equal(particles['xacceleration'][0], 0.707106781)
        assert_almost_equal(particles['yacceleration'][0], 0.707106781)
        assert_almost_equal(particles['xacceleration'][1], -0.707106781)
        assert_almost_equal(particles['yacceleration'][1], -0.707106781)

    def test_second_law(self):
        a = pairwise.second_law(1, 1, 1, np.sqrt(2))
        assert_almost_equal(a, 0.707106781)

    def test_lennard_jones_energy(self):
        a = pairwise.lennard_jones_energy(1, 1, 2.)
        assert_almost_equal(a, -0.015380859)

    def test_lennard_jones_force(self):
        a = pairwise.lennard_jones_force(1, 1, 2.)
        assert_almost_equal(a, -0.045410156)