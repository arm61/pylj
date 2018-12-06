from numpy.testing import assert_almost_equal, assert_equal
from pylj import util
from pylj import forcefields as ff
import unittest


class TestUtil(unittest.TestCase):
    def test_system_square(self):
        a = util.System(2, 300, 8, mass=39.948,
                        constants=[1.363e-134, 9.273e-78],
                        forcefield=ff.lennard_jones)
        assert_equal(a.number_of_particles, 2)
        assert_equal(a.init_temp, 300)
        assert_almost_equal(a.box_length * 1e10, 8)
        assert_almost_equal(a.timestep_length, 1e-14)
        assert_almost_equal(a.particles['xposition'] * 1e10, [2, 2])
        assert_almost_equal(a.particles['yposition'] * 1e10, [2, 6])
        assert_almost_equal(a.initial_particles['xposition'] * 1e10, [2, 2])
        assert_almost_equal(a.initial_particles['yposition'] * 1e10, [2, 6])
        assert_almost_equal(a.cut_off * 1e10, 4.0)
        assert_equal(a.distances.size, 1)
        assert_equal(a.forces.size, 1)
        assert_equal(a.energies.size, 1)

    def test_system_random(self):
        a = util.System(2, 300, 8, init_conf='random', mass=39.948,
                        constants=[1.363e-134, 9.273e-78],
                        forcefield=ff.lennard_jones)
        assert_equal(a.number_of_particles, 2)
        assert_equal(a.init_temp, 300)
        assert_almost_equal(a.box_length * 1e10, 8)
        assert_almost_equal(a.timestep_length, 1e-14)
        self.assertTrue(0 <= a.particles['xposition'][0] * 1e10 <= 8)
        self.assertTrue(0 <= a.particles['yposition'][0] * 1e10 <= 8)
        self.assertTrue(0 <= a.particles['xposition'][1] * 1e10 <= 8)
        self.assertTrue(0 <= a.particles['yposition'][1] * 1e10 <= 8)
        assert_almost_equal(a.cut_off * 1e10, 4.0)
        assert_equal(a.distances.size, 1)
        assert_equal(a.forces.size, 1)
        assert_equal(a.energies.size, 1)

    def test_system_too_big(self):
        with self.assertRaises(AttributeError) as context:
            util.System(2, 300, 1000, mass=39.948,
                        constants=[1.363e-134, 9.273e-78],
                        forcefield=ff.lennard_jones)
        self.assertTrue('With a box length of 1000 the particles are probably '
                        'too small to be seen in the viewer. Try something '
                        '(much) less than 600.' in str(context.exception))

    def test_system_too_small(self):
        with self.assertRaises(AttributeError) as context:
            util.System(2, 300, 2, mass=39.948,
                        constants=[1.363e-134, 9.273e-78],
                        forcefield=ff.lennard_jones)
        self.assertTrue('With a box length of 2 the cell is too small to '
                        'really hold more than one particle.' in str(
                                context.exception))

    def test_system_init_conf(self):
        with self.assertRaises(NotImplementedError) as context:
            util.System(2, 300, 100, init_conf='horseradish', mass=39.948,
                        constants=[1.363e-134, 9.273e-78],
                        forcefield=ff.lennard_jones)
        self.assertTrue('The initial configuration type horseradish is not '
                        'recognised. Available options are: square or '
                        'random' in str(context.exception))
