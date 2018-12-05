from numpy.testing import assert_almost_equal, assert_equal
from pylj import md
import unittest


class TestMd(unittest.TestCase):
    def test_initialise_square(self):
        a = md.initialise(2, 300, 8, 'square')
        assert_equal(a.number_of_particles, 2)
        assert_almost_equal(a.box_length, 8e-10)
        assert_almost_equal(a.init_temp, 300)
        assert_almost_equal(a.particles['xposition']*1e10, [2, 2])
        assert_almost_equal(a.particles['yposition']*1e10, [2, 6])

    def test_velocity_verlet(self):
        a = md.initialise(2, 300, 8, 'square')
        a.particles, a.distances, a.forces, a.energies = md.velocity_verlet(
                a.particles, 1, a.box_length, a.cut_off, a.constants, a.
                forcefield, a.mass)
        assert_almost_equal(a.particles['xprevious_position']*1e10, [2, 2])
        assert_almost_equal(a.particles['yprevious_position']*1e10, [2, 6])

    def test_update_positions(self):
        a = md.initialise(2, 300, 8, 'square')
        a.particles['xvelocity'] = 1e4
        a.particles['yvelocity'] = 1e4
        a.particles['xacceleration'] = 1e4
        a.particles['yacceleration'] = 1e4
        b, c = md.update_positions([a.particles['xposition'],
                                    a.particles['yposition']],
                                   [a.particles['xprevious_position'],
                                    a.particles['yprevious_position']],
                                   [a.particles['xvelocity'],
                                    a.particles['yvelocity']],
                                   [a.particles['xacceleration'],
                                    a.particles['yacceleration']],
                                   a.timestep_length, a.box_length)
        assert_almost_equal(b[0][0]*1e10, 3)
        assert_almost_equal(b[1][0]*1e10, 3)
        assert_almost_equal(b[0][1]*1e10, 3)
        assert_almost_equal(b[1][1]*1e10, 7)

    def test_update_velocities(self):
        a = md.initialise(2, 300, 8, 'square')
        a.particles['xvelocity'] = 1e-10
        a.particles['yvelocity'] = 1e-10
        a.particles['xacceleration'] = 1e4
        a.particles['yacceleration'] = 1e4
        xacceleration_new = 2e4
        yacceleration_new = 2e4
        b = md.update_velocities([a.particles['xvelocity'],
                                  a.particles['yvelocity']],
                                 [xacceleration_new, yacceleration_new],
                                 [a.particles['xacceleration'],
                                  a.particles['yacceleration']],
                                 a.timestep_length)
        assert_almost_equal(b[0][0]*1e10, 2.5)
        assert_almost_equal(b[1][0]*1e10, 2.5)
        assert_almost_equal(b[0][1]*1e10, 2.5)
        assert_almost_equal(b[1][1]*1e10, 2.5)

    def test_calculate_temperature(self):
        a = md.initialise(1, 300, 8, 'square')
        a.particles['xvelocity'] = [1e-10]
        a.particles['yvelocity'] = [1e-10]
        a.particles['xacceleration'] = [1e4]
        a.particles['yacceleration'] = [1e4]
        b = md.calculate_temperature(a.particles, mass=39.948)
        assert_almost_equal(b * 1e23, 4.8048103702737945)

    def test_calculate_msd(self):
        a = md.initialise(2, 300, 8, 'square')
        a.particles['xposition'] = [3e-10, 3e-10]
        a.particles['yposition'] = [3e-10, 7e-10]
        b = md.calculate_msd(a.particles, a.initial_particles, a.box_length)
        assert_almost_equal(b, 2e-20)

    def test_calculate_msd_large(self):
        a = md.initialise(2, 300, 8, 'square')
        a.particles['xposition'] = [7e-10, 3e-10]
        a.particles['yposition'] = [7e-10, 7e-10]
        b = md.calculate_msd(a.particles, a.initial_particles, a.box_length)
        assert_almost_equal(b, 10e-20)
