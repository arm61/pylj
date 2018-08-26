from numpy.testing import assert_almost_equal
from pylj import pairwise, util
import unittest
import numpy as np


class TestPairwise(unittest.TestCase):
    def test_lennard_jones_energy(self):
        a = pairwise.lennard_jones_energy(1., 1., 2.)
        assert_almost_equal(a, -0.015380859)

    def test_lennard_jones_force(self):
        a = pairwise.lennard_jones_force(1., 1., 2.)
        assert_almost_equal(a, -0.045410156)

    def test_update_accelerations(self):
        part_dt = util.particle_dt()
        particles = np.zeros(2, dtype=part_dt)
        ones = np.array([1])
        dist = np.array([np.sqrt(2)])
        particles = pairwise.update_accelerations(particles, ones, 1, ones,
                                                  ones, dist)
        assert_almost_equal(particles['xacceleration'][0], 0.707106781)
        assert_almost_equal(particles['yacceleration'][0], 0.707106781)
        assert_almost_equal(particles['xacceleration'][1], -0.707106781)
        assert_almost_equal(particles['yacceleration'][1], -0.707106781)

    def test_second_law(self):
        a = pairwise.second_law(1, 1, 1, np.sqrt(2))
        assert_almost_equal(a, 0.707106781)

    def test_separation(self):
        a = pairwise.separation(1, 1)
        assert_almost_equal(a, np.sqrt(2))

    def test_compute_forces(self):
        part_dt = util.particle_dt()
        particles = np.zeros(2, dtype=part_dt)
        particles['xposition'][0] = 1e-10
        particles['xposition'][1] = 5e-10
        particles, distances, forces, energies = pairwise.compute_force(
                particles, 30, 15)
        assert_almost_equal(distances, [4e-10])
        assert_almost_equal(energies, [-1.4515047e-21])
        assert_almost_equal(forces, [-9.5864009e-12])
        assert_almost_equal(particles['yacceleration'], [0, 0])
        assert_almost_equal(particles['xacceleration'][0]/1e14, 1.4451452)
        assert_almost_equal(particles['xacceleration'][1]/1e14,  -1.4451452)

    def test_compute_energy(self):
        part_dt = util.particle_dt()
        particles = np.zeros(2, dtype=part_dt)
        particles['xposition'][0] = 1e-10
        particles['xposition'][1] = 5e-10
        distances, energies = pairwise.compute_energy(particles, 30, 15)
        assert_almost_equal(distances, [4e-10])
        assert_almost_equal(energies, [-1.4515047e-21])

    def test_calculate_pressure(self):
        part_dt = util.particle_dt()
        particles = np.zeros(2, dtype=part_dt)
        particles['xposition'][0] = 1e-10
        particles['xposition'][1] = 5e-10
        pressure = pairwise.calculate_pressure(particles, 30, 300, 15)
        assert_almost_equal(pressure*1e24, 7.07368869)

    def test_pbc_correction(self):
        a = pairwise.pbc_correction(1, 10)
        assert_almost_equal(a, 1)
        b = pairwise.pbc_correction(11, 10)
        assert_almost_equal(b, 1)
