import unittest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal

from pylj import forcefields as ff
from pylj import pairwise, util
from pylj.constants import ATOMIC_MASS_UNIT


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

    def test_compute_force_matches_a_reference_loop(self):
        # An independent double loop over pairs is the oracle for the pairwise
        # distances, energies, forces and accelerations. Full-box positions
        # exercise the minimum image. cut_off is large so no pair is zeroed,
        # keeping the oracle to the pair maths that the other tests do not pin.
        rng = np.random.default_rng(0)
        n = 8
        box_length = 30e-10
        part_dt = util.particle_dt()
        particles = np.zeros(n, dtype=part_dt)
        particles["xposition"] = rng.uniform(0, box_length, n)
        particles["yposition"] = rng.uniform(0, box_length, n)
        types = ["0", "0", "1", "1", "0", "1", "0", "1"]
        particles["types"] = types
        constants = [[1.363e-134, 9.273e-78], [1.5e-134, 1.1e-77]]
        mass = 39.948
        cut_off = 1e-8
        mass_kg = mass * ATOMIC_MASS_UNIT

        ref_dr, ref_energy, ref_force = [], [], []
        ref_ax = np.zeros(n)
        ref_ay = np.zeros(n)
        for a in range(n - 1):
            for b in range(a + 1, n):
                dx = particles["xposition"][a] - particles["xposition"][b]
                dy = particles["yposition"][a] - particles["yposition"][b]
                dx -= box_length * np.round(dx / box_length)
                dy -= box_length * np.round(dy / box_length)
                dr = np.hypot(dx, dy)
                low, high = sorted((int(types[a]), int(types[b])))
                pair = ff.lennard_jones(np.array(constants[low]))
                if low != high:
                    pair.mixing(np.array(constants[high]))
                force = pair.force(dr)
                ref_dr.append(dr)
                ref_energy.append(pair.energy(dr))
                ref_force.append(force)
                ax = force * dx / dr / mass_kg
                ay = force * dy / dr / mass_kg
                ref_ax[a] += ax
                ref_ax[b] -= ax
                ref_ay[a] += ay
                ref_ay[b] -= ay

        particles, distances, forces, energies = pairwise.compute_force(
            particles,
            box_length,
            cut_off,
            constants=constants,
            forcefield=ff.lennard_jones,
            mass=mass,
        )
        np.testing.assert_allclose(distances, ref_dr, rtol=1e-12)
        np.testing.assert_allclose(energies, ref_energy, rtol=1e-12)
        np.testing.assert_allclose(forces, ref_force, rtol=1e-12)
        np.testing.assert_allclose(particles["xacceleration"], ref_ax, rtol=1e-12)
        np.testing.assert_allclose(particles["yacceleration"], ref_ay, rtol=1e-12)
