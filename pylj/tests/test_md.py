import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import forcefields as ff
from pylj import md, pairwise
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN


class TestMd(unittest.TestCase):
    def test_initialise_square(self):
        a = md.initialise(2, 300, 8, "square")
        assert_equal(a.number_of_particles, 2)
        assert_almost_equal(a.box_length, 8e-10)
        assert_almost_equal(a.init_temp, 300)
        assert_almost_equal(a.particles["xposition"] * 1e10, [2, 2])
        assert_almost_equal(a.particles["yposition"] * 1e10, [2, 6])
        assert_equal(a.simulation, "md")

    def test_initialise_computes_initial_forces(self):
        a = md.initialise(2, 300, 8, "square")
        assert_almost_equal(a.distances * 1e10, [4.0])
        self.assertTrue(np.any(a.particles["yacceleration"] != 0))

    def test_initialize_passes_keyword_arguments_through(self):
        constants = [[3.4e-10, 1.65e-21]]
        a = md.initialize(
            2,
            300,
            8,
            "square",
            mass=20.0,
            constants=constants,
            forcefield=ff.lennard_jones_sigma_epsilon,
            diameter=3.0,
        )
        assert_equal(a.mass, 20.0)
        assert_equal(a.constants, constants)
        assert_equal(a.forcefield, ff.lennard_jones_sigma_epsilon)
        assert_almost_equal(a.diameters, [3e-10])

    def test_sample_records_step_and_thermodynamics(self):
        a = md.initialise(2, 300, 8, "square")
        a.step = 3
        a.md_sample()
        assert_equal(a.step_sample, [3])
        assert_equal(a.temperature_sample.size, 1)
        assert_equal(a.pressure_sample.size, 1)
        assert_equal(a.energy_sample.size, 1)
        assert_equal(a.force_sample.size, 1)
        assert_equal(a.msd_sample.size, 1)
        a.step = 7
        a.md_sample()
        assert_equal(a.step_sample, [3, 7])
        assert_equal(a.energy_sample.size, 2)

    def test_sample_pressure_uses_the_stored_pair_forces(self):
        system = md.initialise(20, 300, 20, "square")
        system.integrate(md.velocity_verlet)
        # Sentinels the force loop would never produce: the sampled pressure
        # reflects them only if sample reuses the stored data instead of
        # recomputing it from the particle positions.
        system.distances = np.full_like(system.distances, 3e-10)
        system.forces = np.full_like(system.forces, 1e-12)
        system.md_sample()
        temperature = md.calculate_temperature(system.particles, system.mass)
        expected = pairwise.calculate_pressure(
            system.distances,
            system.forces,
            system.box_length,
            system.particles.size,
            temperature,
        )
        assert_almost_equal(system.pressure_sample[-1], expected)

    def test_velocity_verlet(self):
        a = md.initialise(2, 300, 8, "square")
        a.particles, a.distances, a.forces, a.energies = md.velocity_verlet(
            a.particles, 1, a.box_length, a.cut_off, a.constants, a.forcefield, a.mass
        )
        assert_almost_equal(a.particles["xprevious_position"] * 1e10, [2, 2])
        assert_almost_equal(a.particles["yprevious_position"] * 1e10, [2, 6])

    def test_update_positions(self):
        a = md.initialise(2, 300, 8, "square")
        a.particles["xvelocity"] = 1e4
        a.particles["yvelocity"] = 1e4
        a.particles["xacceleration"] = 1e4
        a.particles["yacceleration"] = 1e4
        b, c = md.update_positions(
            [a.particles["xposition"], a.particles["yposition"]],
            [a.particles["xprevious_position"], a.particles["yprevious_position"]],
            [a.particles["xvelocity"], a.particles["yvelocity"]],
            [a.particles["xacceleration"], a.particles["yacceleration"]],
            a.timestep_length,
            a.box_length,
        )
        assert_almost_equal(b[0][0] * 1e10, 3)
        assert_almost_equal(b[1][0] * 1e10, 3)
        assert_almost_equal(b[0][1] * 1e10, 3)
        assert_almost_equal(b[1][1] * 1e10, 7)

    def test_update_velocities(self):
        a = md.initialise(2, 300, 8, "square")
        a.particles["xvelocity"] = 1e-10
        a.particles["yvelocity"] = 1e-10
        a.particles["xacceleration"] = 1e4
        a.particles["yacceleration"] = 1e4
        xacceleration_new = 2e4
        yacceleration_new = 2e4
        b = md.update_velocities(
            [a.particles["xvelocity"], a.particles["yvelocity"]],
            [xacceleration_new, yacceleration_new],
            [a.particles["xacceleration"], a.particles["yacceleration"]],
            a.timestep_length,
        )
        assert_almost_equal(b[0][0] * 1e10, 2.5)
        assert_almost_equal(b[1][0] * 1e10, 2.5)
        assert_almost_equal(b[0][1] * 1e10, 2.5)
        assert_almost_equal(b[1][1] * 1e10, 2.5)

    def test_calculate_temperature(self):
        a = md.initialise(1, 300, 8, "square")
        a.particles["xvelocity"] = [1e-10]
        a.particles["yvelocity"] = [1e-10]
        a.particles["xacceleration"] = [1e4]
        a.particles["yacceleration"] = [1e4]
        b = md.calculate_temperature(a.particles, mass=39.948)
        # T = m (vx^2 + vy^2) / (2 N k_B) for the one particle in the cell.
        expected = 0.5 * 39.948 * ATOMIC_MASS_UNIT * 2e-20 / BOLTZMANN
        assert_almost_equal(b * 1e23, expected * 1e23)

    def test_calculate_msd(self):
        a = md.initialise(2, 300, 8, "square")
        a.particles["xposition"] = [3e-10, 3e-10]
        a.particles["yposition"] = [3e-10, 7e-10]
        b = md.calculate_msd(a.particles, a.initial_particles, a.box_length)
        assert_almost_equal(b, 2e-20)

    def test_calculate_msd_large(self):
        a = md.initialise(2, 300, 8, "square")
        a.particles["xposition"] = [7e-10, 3e-10]
        a.particles["yposition"] = [7e-10, 7e-10]
        b = md.calculate_msd(a.particles, a.initial_particles, a.box_length)
        assert_almost_equal(b, 10e-20)

    def test_initialise_accepts_diameter(self):
        a = md.initialise(2, 300, 8, "square", diameter=3.0)
        assert_almost_equal(a.diameters, [3e-10])

    def test_heat_bath_rescales_to_bath_temperature(self):
        a = md.initialise(10, 300, 20, "square")
        a.particles = md.heat_bath(a.particles, a.mass, 250.0)
        t = md.calculate_temperature(a.particles, a.mass)
        assert_almost_equal(t / 250.0, 1.0)

    def test_heat_bath_preserves_velocity_directions(self):
        a = md.initialise(10, 300, 20, "square")
        old_x = np.array(a.particles["xvelocity"])
        old_y = np.array(a.particles["yvelocity"])
        a.particles = md.heat_bath(a.particles, a.mass, 250.0)
        x_ratio = a.particles["xvelocity"] / old_x
        y_ratio = a.particles["yvelocity"] / old_y
        assert_almost_equal(x_ratio, np.full(x_ratio.shape, x_ratio[0]))
        assert_almost_equal(y_ratio, np.full(y_ratio.shape, x_ratio[0]))

    def test_heat_bath_ignores_the_temperature_sample_record(self):
        for history in ([], [1000.0, 1000.0, 1000.0]):
            a = md.initialise(10, 300, 20, "square")
            a.temperature_sample = np.array(history)
            a.heat_bath(50.0)
            t = md.calculate_temperature(a.particles, a.mass)
            assert_almost_equal(t, 50.0)

    def test_heat_bath_two_calls_each_hit_their_own_target(self):
        a = md.initialise(10, 300, 20, "square")
        a.particles = md.heat_bath(a.particles, a.mass, 250.0)
        a.particles = md.heat_bath(a.particles, a.mass, 100.0)
        t = md.calculate_temperature(a.particles, a.mass)
        assert_almost_equal(t / 100.0, 1.0)

    def test_heat_bath_raises_when_the_particles_are_at_rest(self):
        a = md.initialise(10, 300, 20, "square")
        a.particles["xvelocity"] = 0.0
        a.particles["yvelocity"] = 0.0
        with self.assertRaises(ValueError) as context:
            md.heat_bath(a.particles, a.mass, 250.0)
        self.assertIn("at rest", str(context.exception))

    def test_heat_bath_raises_when_the_temperature_is_not_finite(self):
        for bad in (np.inf, np.nan):
            a = md.initialise(10, 300, 20, "square")
            a.particles["xvelocity"][0] = bad
            with self.assertRaises(ValueError) as context:
                md.heat_bath(a.particles, a.mass, 250.0)
            self.assertIn("diverged", str(context.exception))

    def test_heat_bath_raises_for_a_non_positive_bath_temperature(self):
        for bad in (0.0, -5.0, np.nan):
            a = md.initialise(10, 300, 20, "square")
            with self.assertRaises(ValueError):
                md.heat_bath(a.particles, a.mass, bad)
