from numpy.testing import assert_almost_equal, assert_equal
from pylj import util
try:
    from pylj import comp as heavy
except ImportError:
    print("WARNING, using slow force and energy calculations")
    from pylj import pairwise as heavy
import unittest
import numpy as np

class TestPairwise(unittest.TestCase):
    def test_compute_forces(self):
        part_dt = util.particle_dt()
        particles = np.zeros(2, dtype=part_dt)
        particles['xposition'][0] = 1e-10
        particles['xposition'][1] = 5e-10
        particles, distances, forces, energies = heavy.compute_forces(particles, 30, 15)
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
        distances, energies = heavy.compute_energy(particles, 30, 15)
        assert_almost_equal(distances, [4e-10])
        assert_almost_equal(energies, [-1.4515047e-21])

    def test_calculate_pressure(self):
        part_dt = util.particle_dt()
        particles = np.zeros(2, dtype=part_dt)
        particles['xposition'][0] = 1e-10
        particles['xposition'][1] = 5e-10
        pressure = heavy.calculate_pressure(particles, 30, 300, 15)
        assert_almost_equal(pressure*1e24, 7.07368869)