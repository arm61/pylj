from numpy.testing import assert_almost_equal, assert_equal
from pylj import util, md
import unittest

class TestUtil(unittest.TestCase):
    def test_system_square(self):
        a = util.System(2, 300, 8)
        assert_equal(a.number_of_particles, 2)
        assert_equal(a.init_temp, 300)
        assert_almost_equal(a.box_length * 1e10, 8)
        assert_almost_equal(a.timestep_length, 1e-14)
        assert_almost_equal(a.particles['xposition'] * 1e10, [2, 2])
        assert_almost_equal(a.particles['yposition'] * 1e10, [2, 6])
        assert_almost_equal(a.initial_particles['xposition'] * 1e10, [2, 2])
        assert_almost_equal(a.initial_particles['yposition'] * 1e10, [2, 6])
        assert_almost_equal(a.cut_off * 1e10, 15)
        assert_equal(a.distances.size, 1)
        assert_equal(a.forces.size, 1)
        assert_equal(a.energies.size, 1)

    def test_system_random(self):
        a = util.System(2, 300, 8, init_conf='random')
        assert_equal(a.number_of_particles, 2)
        assert_equal(a.init_temp, 300)
        assert_almost_equal(a.box_length * 1e10, 8)
        assert_almost_equal(a.timestep_length, 1e-14)
        self.assertTrue(0 <= a.particles['xposition'][0] * 1e10 <= 8)
        self.assertTrue(0 <= a.particles['yposition'][0] * 1e10 <= 8)
        self.assertTrue(0 <= a.particles['xposition'][1] * 1e10 <= 8)
        self.assertTrue(0 <= a.particles['yposition'][1] * 1e10 <= 8)
        assert_almost_equal(a.cut_off * 1e10, 15)
        assert_equal(a.distances.size, 1)
        assert_equal(a.forces.size, 1)
        assert_equal(a.energies.size, 1)

    def test_system_too_big(self):
        with self.assertRaises(AttributeError) as context:
            a = util.System(2, 300, 1000)
        self.assertTrue('With a box length of 1000 the particles are probably too small to be seen in the viewer. Try '
                        'something (much) less than 600.' in str(context.exception))

    def test_system_too_small(self):
        with self.assertRaises(AttributeError) as context:
            a = util.System(2, 300, 2)
        self.assertTrue('With a box length of 2 the cell is too small to really hold more than one '
                        'particle.' in str(context.exception))

    def test_system_init_conf(self):
        with self.assertRaises(NotImplementedError) as context:
            a = util.System(2, 300, 100, init_conf='horseradish')
        self.assertTrue('The initial configuration type horseradish is not recognised. '
                        'Available options are: square or random' in str(context.exception))

    def test_pbc_correction(self):
        a = util.pbc_correction(1, 10)
        assert_almost_equal(a, 1)
        b = util.pbc_correction(11, 10)
        assert_almost_equal(b, 1)

    def test_calculate_temperature(self):
        a = md.initialise(1, 300, 8, 'square')
        a.particles['xvelocity'] = [1e-10]
        a.particles['yvelocity'] = [1e-10]
        a.particles['xacceleration'] = [1e4]
        a.particles['yacceleration'] = [1e4]
        b = util.calculate_temperature(a.particles)
        assert_almost_equal(b * 1e23, 4.797479357)

    def test_calculate_msd(self):
        a = md.initialise(2, 300, 8, 'square')
        a.particles['xposition'] = [3e-10, 3e-10]
        a.particles['yposition'] = [3e-10, 7e-10]
        b = util.calculate_msd(a.particles, a.initial_particles, a.box_length)
        assert_almost_equal(b, 2e-20)

    def test_calculate_msd_large(self):
        a = md.initialise(2, 300, 8, 'square')
        a.particles['xposition'] = [7e-10, 3e-10]
        a.particles['yposition'] = [7e-10, 7e-10]
        b = util.calculate_msd(a.particles, a.initial_particles, a.box_length)
        assert_almost_equal(b, 10e-20)
