import unittest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal

from pylj import forcefields as ff
from pylj import pairwise, util


class gaussian_core:
    r"""A pair potential that is finite and non-zero at zero separation,
    used to check that compute_force never evaluates a forcefield on a
    pair of a different type.

    .. math::
        E = a e^{-(r / b)^{2}}

        f = \frac{2 a r}{b^{2}} e^{-(r / b)^{2}}
    """

    def __init__(self, constants):
        self.a = constants[0]
        self.b = constants[1]

    def energy(self, dr):
        dr = np.asarray(dr, dtype=float)
        return self.a * np.exp(-((dr / self.b) ** 2))

    def force(self, dr):
        dr = np.asarray(dr, dtype=float)
        return 2 * self.a * dr / self.b**2 * np.exp(-((dr / self.b) ** 2))

    def mixing(self, constants2):
        self.a = (self.a + constants2[0]) / 2
        self.b = (self.b + constants2[1]) / 2

    @property
    def diameter(self):
        return self.b


class TestPairwise(unittest.TestCase):
    def test_update_accelerations(self):
        part_dt = util.particle_dt()
        particles = np.zeros(2, dtype=part_dt)
        ones = np.array([1])
        dist = np.array([np.sqrt(2)])
        particles = pairwise.update_accelerations(particles, ones, 1, ones, ones, dist)
        assert_almost_equal(particles["xacceleration"][0], 0.707106781)
        assert_almost_equal(particles["yacceleration"][0], 0.707106781)
        assert_almost_equal(particles["xacceleration"][1], -0.707106781)
        assert_almost_equal(particles["yacceleration"][1], -0.707106781)

    def test_second_law(self):
        a = pairwise.second_law(1, 1, 1, np.sqrt(2))
        assert_almost_equal(a, 0.707106781)

    def test_separation(self):
        a = pairwise.separation(1, 1)
        assert_almost_equal(a, np.sqrt(2))

    def test_compute_forces(self):
        part_dt = util.particle_dt()
        particles = np.zeros(2, dtype=part_dt)
        particles["xposition"][0] = 1e-10
        particles["xposition"][1] = 5e-10
        particles['types'] = ['0','0']
        particles, distances, forces, energies = pairwise.compute_force(
            particles,
            30,
            15,
            constants=[[1.363e-134, 9.273e-78]],
            forcefield=ff.lennard_jones,
            mass=39.948
        )
        assert_almost_equal(distances, [4e-10])
        assert_almost_equal(energies, [-1.4515047e-21])
        assert_almost_equal(forces, [-9.5864009e-12])
        assert_almost_equal(particles["yacceleration"], [0, 0])
        assert_almost_equal(particles["xacceleration"][0] / 1e14, 1.4451452)
        assert_almost_equal(particles["xacceleration"][1] / 1e14, -1.4451452)

    def test_calculate_pressure(self):
        part_dt = util.particle_dt()
        particles = np.zeros(2, dtype=part_dt)
        particles["xposition"][0] = 1e-10
        particles["xposition"][1] = 5e-10
        particles['types'] = ['0','0']
        p = pairwise.calculate_pressure(
            particles,
            30,
            300,
            15,
            constants=[[1.363e-134, 9.273e-78]],
            forcefield=ff.lennard_jones,
            mass = 39.948
        )
        # Only the ideal-gas term N k_B T / L^2 depends on the Boltzmann
        # constant. Moving from 1.3806e-23 to the CODATA 1.380649e-23 raises
        # it by 3.267e-28 Pa, which is the whole of the change from the
        # previous expectation of 7.07368867.
        assert_almost_equal(p * 1e24, 7.07401534)

    def test_pbc_correction(self):
        a = pairwise.pbc_correction(1, 10)
        assert_almost_equal(a, 1)
        b = pairwise.pbc_correction(11, 10)
        assert_almost_equal(b, 1)
    
    def test_multiple_particles(self):
        part_dt = util.particle_dt()
        particles = np.zeros(3, dtype=part_dt)
        particles["xposition"][0] = 1e-10
        particles["xposition"][1] = 5e-10
        particles["yposition"][2] = 5e-10
        particles['types'] = ['0','1','0']
        constants = [[1.363e-134, 9.273e-78], [1.363e-133, 9.273e-77]]
        particles, distances, forces, energies = pairwise.compute_force(
            particles,
            30,
            15,
            constants=constants,
            forcefield=ff.lennard_jones,
            mass=39.948
        )
        assert_almost_equal(distances, [4.0000000e-10, 5.0990195e-10, 7.0710678e-10])
        # Each pair contributes its energy once, evaluated with the forcefield
        # for its own pair of types. Particles 0 and 2 are type 0 and particle
        # 1 is type 1, so the pairs are (0, 1), (0, 0) and (1, 0) in the order
        # the distances come back. Unlike pairs mix the two sets of constants,
        # as compute_force does.
        expected = []
        for distance, pair in zip(distances, [(0, 1), (0, 0), (1, 0)], strict=True):
            forcefield = ff.lennard_jones(constants[pair[0]])
            if pair[0] != pair[1]:
                forcefield.mixing(constants[pair[1]])
            expected.append(forcefield.energy(distance))
        assert_almost_equal(np.array(energies) * 1e20, np.array(expected) * 1e20)
        assert_almost_equal(forces, [-9.6342138e-11, -5.1698213e-12, -6.1773405e-12])
        assert_almost_equal(particles["yacceleration"],
                            [7.6421357e+13, 2.07196175e+13, -9.71409740e+13],
                            decimal=-7)
        # The accelerations go as 1 / mass, so they shift by 42 parts in 1e9
        # with the CODATA atomic mass unit, from 4.4171075 and -4.7771464.
        assert_almost_equal(particles["xacceleration"][0] / 1e14, 4.4171073)
        assert_almost_equal(particles["xacceleration"][1] / 1e14, -4.7771462)

    def test_compute_force_zeroes_pairs_beyond_the_cut_off(self):
        # cut_off is compared against the pair distances directly, so it is in
        # the same units as the positions (metres here). At 6e-10 m it sits
        # between the (1, 2) pair at 5e-10 and the (0, 2) pair at 8e-10, so one
        # pair is inside the cut-off and one is outside.
        part_dt = util.particle_dt()
        particles = np.zeros(3, dtype=part_dt)
        particles["xposition"] = [0.0, 3e-10, 8e-10]
        particles["types"] = ["0", "0", "0"]
        constants = [[1.363e-134, 9.273e-78]]
        particles, distances, forces, energies = pairwise.compute_force(
            particles,
            30,
            6e-10,
            constants=constants,
            forcefield=ff.lennard_jones,
            mass=39.948,
        )
        assert_almost_equal(distances, [3e-10, 8e-10, 5e-10])
        # The pair beyond the cut-off is zeroed; the two inside keep the values
        # the forcefield gives, so the mask is neither dropped nor inverted.
        lj = ff.lennard_jones(np.array(constants[0]))
        self.assertEqual(energies[1], 0.0)
        self.assertEqual(forces[1], 0.0)
        assert_almost_equal(energies[0] * 1e20, lj.energy(distances[0]) * 1e20)
        assert_almost_equal(energies[2] * 1e20, lj.energy(distances[2]) * 1e20)
        assert_almost_equal(forces[0] * 1e12, lj.force(distances[0]) * 1e12)
        assert_almost_equal(forces[2] * 1e12, lj.force(distances[2]) * 1e12)

    def test_compute_force_does_not_warn_on_a_two_type_system(self):
        part_dt = util.particle_dt()
        particles = np.zeros(3, dtype=part_dt)
        particles["xposition"][0] = 1e-10
        particles["xposition"][1] = 5e-10
        particles["yposition"][2] = 5e-10
        particles['types'] = ['0', '1', '0']
        constants = [[1.363e-134, 9.273e-78], [1.363e-133, 9.273e-77]]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pairwise.compute_force(
                particles,
                30,
                15,
                constants=constants,
                forcefield=ff.lennard_jones,
                mass=39.948,
            )

    def test_compute_force_evaluates_each_pair_on_its_own_distance(self):
        # gaussian_core is finite and non-zero at zero separation, so this
        # would fail if compute_force still passed a forcefield distances
        # belonging to other type pairs, zeroed out.
        part_dt = util.particle_dt()
        particles = np.zeros(3, dtype=part_dt)
        particles["xposition"][0] = 1e-10
        particles["xposition"][1] = 5e-10
        particles["yposition"][2] = 5e-10
        particles['types'] = ['0', '1', '0']
        constants = [[1.0, 2.0], [3.0, 4.0]]
        particles, distances, forces, energies = pairwise.compute_force(
            particles,
            30,
            15,
            constants=constants,
            forcefield=gaussian_core,
            mass=39.948,
        )
        expected_energies = []
        expected_forces = []
        for distance, pair in zip(distances, [(0, 1), (0, 0), (1, 0)], strict=True):
            forcefield = gaussian_core(constants[pair[0]])
            if pair[0] != pair[1]:
                forcefield.mixing(constants[pair[1]])
            expected_energies.append(forcefield.energy(distance))
            expected_forces.append(forcefield.force(distance))
        assert_almost_equal(energies, expected_energies)
        assert_almost_equal(forces, expected_forces)
