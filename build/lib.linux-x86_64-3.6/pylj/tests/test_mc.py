from numpy.testing import assert_almost_equal, assert_equal
from pylj import mc
import unittest


class TestMc(unittest.TestCase):
    def test_initialise_square(self):
        a = mc.initialise(2, 300, 8, 'square')
        assert_equal(a.number_of_particles, 2)
        assert_almost_equal(a.box_length, 8e-10)
        assert_almost_equal(a.init_temp, 300)
        assert_almost_equal(a.particles['xposition']*1e10, [2, 2])
        assert_almost_equal(a.particles['yposition']*1e10, [2, 6])

    def test_initialize_square(self):
        a = mc.initialize(2, 300, 8, 'square')
        assert_equal(a.number_of_particles, 2)
        assert_almost_equal(a.box_length, 8e-10)
        assert_almost_equal(a.init_temp, 300)
        assert_almost_equal(a.particles['xposition']*1e10, [2, 2])
        assert_almost_equal(a.particles['yposition']*1e10, [2, 6])

    def test_sample(self):
        a = mc.initialise(2, 300, 8, 'square')
        a = mc.sample(300, a)
        assert_almost_equal(a.energy_sample, [300])

    def test_select_random_particle(self):
        a = mc.initialise(2, 300, 8, 'square')
        b, c = mc.select_random_particle(a.particles)
        self.assertTrue(0 <= b < 2)
        self.assertTrue(0 <= c[0] <= 8e-10)
        self.assertTrue(0 <= c[1] <= 8e-10)

    def test_get_new_particle(self):
        a = mc.initialise(2, 300, 8, 'square')
        b, c = mc.select_random_particle(a.particles)
        d = mc.get_new_particle(a.particles, b, a.box_length)
        self.assertTrue(0 <= d['xposition'][b] <= 8e-10)
        self.assertTrue(0 <= d['yposition'][b] <= 8e-10)

    def test_accept(self):
        a = mc.accept(300)
        assert_almost_equal(a, 300)

    def test_reject(self):
        a = mc.initialise(2, 300, 8, 'square')
        b = [1e-10, 1e-10]
        c = mc.reject(b, a.particles, 1)
        assert_almost_equal(c['xposition'][1]*1e10, 1)
        assert_almost_equal(c['yposition'][1]*1e10, 1)

    def test_metropolis_energy_reduce(self):
        a = mc.metropolis(300, 100, 1)
        self.assertTrue(a)

    def test_metropolis_energy_increase_accept(self):
        a = mc.metropolis(300, 100e-20, 101e-20, n=0.01)
        self.assertTrue(a)

    def test_metropolis_energy_increase_reject(self):
        a = mc.metropolis(300, 100e-20, 101e-20, n=0.1)
        self.assertFalse(a)
