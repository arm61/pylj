import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from pylj import md, pairwise, placement
from pylj.configuration import MDConfiguration
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN
from pylj.md import MDSimulation
from pylj.potentials import LennardJones
from pylj.tests.argon import ARGON, ARGON_MODEL, MIXTURE_MODEL, WELL_MODEL


def two_argon(velocity, box=8e-10):
    """Two argon particles at (2, 2) and (2, 6) Angstrom with the given velocities."""
    position = np.array([[2e-10, 2e-10], [2e-10, 6e-10]])
    return MDConfiguration(
        position=position,
        species=(ARGON,),
        species_index=np.zeros(2, dtype=np.int64),
        box=box,
        velocity=np.asarray(velocity, dtype=float),
        unwrapped=position.copy(),
    )


def kinetic_plus_potential(sim):
    """The total energy of a simulation, from its configuration and stored cut-off."""
    c = sim.configuration
    return c.kinetic_energy() + c.potential_energy(sim.pair_potentials, sim.cut_off)


class TestInitialise(unittest.TestCase):
    def test_square_lattice_in_a_converted_box(self):
        a = MDSimulation.initialise(2, 300, 8, **ARGON_MODEL)
        c = a.configuration
        self.assertIsInstance(c, MDConfiguration)
        self.assertEqual(c.number_of_particles, 2)
        assert_almost_equal(c.box, 8e-10)
        assert_almost_equal(c.position * 1e10, [[2, 2], [2, 6]])
        assert_almost_equal(c.unwrapped, c.position)
        assert_almost_equal(a.cut_off * 1e10, 4.0)
        assert_almost_equal(a.timestep, 1e-14)
        self.assertEqual(a.steps, 0)
        self.assertEqual(a.time, 0.0)

    def test_forces_are_valid_after_construction(self):
        a = MDSimulation.initialise(2, 300, 8, **ARGON_MODEL)
        assert_allclose(a.forces, a.configuration.forces(a.pair_potentials, a.cut_off))
        self.assertNotEqual(a.forces[0, 1], 0.0)

    def test_velocities_have_no_net_momentum(self):
        a = MDSimulation.initialise(25, 100, 40, **ARGON_MODEL)
        thermal_speed = np.sqrt(BOLTZMANN * 100 / (ARGON.mass * ATOMIC_MASS_UNIT))
        momentum = a.configuration.velocity.sum(axis=0)
        self.assertLess(abs(momentum[0]), 1e-12 * thermal_speed)
        self.assertLess(abs(momentum[1]), 1e-12 * thermal_speed)

    def test_velocities_match_the_requested_temperature(self):
        a = MDSimulation.initialise(25, 100, 40, **ARGON_MODEL)
        assert_almost_equal(a.configuration.temperature(), 100)

    def test_two_species_have_no_net_momentum_at_the_temperature(self):
        # The centre-of-mass velocity is mass weighted, so with unequal
        # masses the total momentum is zero and the temperature exact.
        a = MDSimulation.initialise(24, 100, 60, **MIXTURE_MODEL)
        c = a.configuration
        assert_allclose(c.masses[:4], np.array([39.948, 80.0, 39.948, 80.0]) * ATOMIC_MASS_UNIT)
        momentum_scale = ARGON.mass * np.sqrt(BOLTZMANN * 100 / (ARGON.mass * ATOMIC_MASS_UNIT))
        momentum = (c.masses[:, None] * c.velocity).sum(axis=0)
        self.assertLess(abs(momentum[0]), 1e-12 * momentum_scale)
        self.assertLess(abs(momentum[1]), 1e-12 * momentum_scale)
        assert_almost_equal(c.temperature(), 100)

    def test_refuses_a_potential_with_no_force(self):
        with self.assertRaisesRegex(ValueError, "Monte Carlo"):
            MDSimulation.initialise(2, 300, 8, **WELL_MODEL)

    def test_is_reproducible_with_a_seed(self):
        def build(seed):
            return MDSimulation.initialise(
                10, 100, 40, init_conf="metropolis", seed=seed, **ARGON_MODEL
            )

        first, second, other = build(3), build(3), build(4)
        assert_equal(first.configuration.position, second.configuration.position)
        assert_equal(first.configuration.velocity, second.configuration.velocity)
        self.assertFalse(np.array_equal(first.configuration.velocity, other.configuration.velocity))
        # The generator continues from the placement and velocity draws.
        self.assertEqual(first.rng.random(), second.rng.random())

    def test_passes_the_placement_temperature_through(self):
        # 50 argon in a 27 Angstrom box cannot be placed at 1 K.
        with self.assertRaisesRegex(ValueError, "Could not place"):
            MDSimulation.initialise(
                50,
                100,
                27,
                init_conf="metropolis",
                placement_temperature=1.0,
                seed=0,
                **ARGON_MODEL,
            )

    def test_passes_the_timestep_and_cut_off_through(self):
        a = MDSimulation.initialise(2, 300, 40, timestep=2e-15, cut_off=10, **ARGON_MODEL)
        assert_almost_equal(a.timestep, 2e-15)
        assert_almost_equal(a.cut_off * 1e10, 10)

    def test_one_particle_raises(self):
        with self.assertRaisesRegex(ValueError, "at least two particles"):
            MDSimulation.initialise(1, 300, 8, **ARGON_MODEL)

    def test_rejects_a_non_positive_or_infinite_temperature(self):
        for temperature in (0, -10, np.inf):
            with self.assertRaisesRegex(ValueError, "temperature must be positive"):
                MDSimulation.initialise(2, temperature, 8, **ARGON_MODEL)

    def test_refuses_an_overlapping_lattice(self):
        with self.assertRaisesRegex(ValueError, "k_B T of potential energy"):
            MDSimulation.initialise(16, 300, 10, **ARGON_MODEL)


class TestConstructor(unittest.TestCase):
    def test_refuses_a_configuration_without_velocities(self):
        c = placement.place_square(2, (ARGON,), 8e-10)
        self.assertNotIsInstance(c, MDConfiguration)
        with self.assertRaisesRegex(TypeError, "MDConfiguration"):
            MDSimulation(c, ARGON_MODEL["pair_potentials"])

    def test_refuses_a_configuration_at_rest(self):
        with self.assertRaisesRegex(ValueError, "at rest"):
            MDSimulation(two_argon(np.zeros((2, 2))), ARGON_MODEL["pair_potentials"])

    def test_checks_the_model_at_the_measured_temperature(self):
        # Sigma in Angstrom is caught against the kinetic temperature of the
        # configuration given.
        wrong = {(ARGON, ARGON): LennardJones(epsilon=1.577e-21, sigma=3.372)}
        with self.assertRaisesRegex(ValueError, "at the cut-off"):
            MDSimulation(two_argon([[3e2, 0.0], [-3e2, 0.0]]), wrong)

    def test_takes_a_ready_configuration(self):
        c = two_argon([[3e2, 0.0], [-3e2, 0.0]])
        a = MDSimulation(c, ARGON_MODEL["pair_potentials"], timestep=2e-15, seed=5)
        self.assertIs(a.configuration, c)
        self.assertIs(a.initial_configuration, c)
        assert_almost_equal(a.timestep, 2e-15)
        assert_allclose(a.forces, c.forces(a.pair_potentials, a.cut_off))


class TestStep(unittest.TestCase):
    def test_step_advances_the_clock(self):
        a = MDSimulation.initialise(4, 100, 20, **ARGON_MODEL)
        a.step()
        a.step()
        self.assertEqual(a.steps, 2)
        assert_almost_equal(a.time, 2 * a.timestep)

    def test_step_uses_the_integrate_method(self):
        class Frozen(MDSimulation):
            def integrate(self):
                pass

        a = Frozen.initialise(4, 100, 20, **ARGON_MODEL)
        before = a.configuration
        a.step()
        self.assertIs(a.configuration, before)
        self.assertEqual(a.steps, 1)

    def test_integrate_replaces_the_configuration_and_the_forces(self):
        a = MDSimulation.initialise(4, 100, 20, **ARGON_MODEL)
        before, forces_before = a.configuration, a.forces
        a.integrate()
        self.assertIsNot(a.configuration, before)
        self.assertFalse(np.array_equal(a.configuration.position, before.position))
        assert_allclose(a.forces, a.configuration.forces(a.pair_potentials, a.cut_off))
        self.assertFalse(np.array_equal(a.forces, forces_before))
        self.assertEqual(a.steps, 0)


class TestVelocityVerlet(unittest.TestCase):
    def test_advances_the_unwrapped_positions(self):
        # A y velocity of 3e4 m/s moves each particle by 3 Angstrom in one
        # 1e-14 s step, so the second particle crosses the boundary of the 8
        # Angstrom box: its wrapped position comes back in and its unwrapped
        # one does not. The forces are zeroed so the motion is the drift.
        c = two_argon([[0.0, 3e4], [0.0, 3e4]])
        moved, forces = md.velocity_verlet(
            c, np.zeros((2, 2)), 1e-14, ARGON_MODEL["pair_potentials"], 15e-10
        )
        assert_almost_equal(moved.unwrapped * 1e10, [[2, 5], [2, 9]])
        assert_almost_equal(moved.position * 1e10, [[2, 5], [2, 1]])
        assert_allclose(forces, moved.forces(ARGON_MODEL["pair_potentials"], 15e-10))

    def test_matches_the_hand_computed_step(self):
        # One particle drifting in x, the other at rest, no forces: the
        # velocities are unchanged and the drift is v dt.
        c = two_argon([[1e3, 0.0], [0.0, 0.0]], box=40e-10)
        cut_off = 15e-10
        forces = c.forces(ARGON_MODEL["pair_potentials"], cut_off)
        moved, next_forces = md.velocity_verlet(
            c, forces, 1e-14, ARGON_MODEL["pair_potentials"], cut_off
        )
        accelerations = forces / c.masses[:, None]
        expected_position = c.position + c.velocity * 1e-14 + 0.5 * accelerations * 1e-28
        assert_allclose(moved.position, expected_position)
        next_accelerations = next_forces / c.masses[:, None]
        expected_velocity = c.velocity + 0.5 * (accelerations + next_accelerations) * 1e-14
        assert_allclose(moved.velocity, expected_velocity)

    def test_update_positions_wraps_the_position_and_not_the_unwrapped_one(self):
        c = two_argon([[1e4, 3e4], [1e4, 3e4]])
        position, unwrapped = md.update_positions(c, np.zeros((2, 2)), 1e-14)
        assert_almost_equal(position * 1e10, [[3, 5], [3, 1]])
        assert_almost_equal(unwrapped * 1e10, [[3, 5], [3, 9]])

    def test_update_velocities_uses_the_mean_acceleration(self):
        velocity = np.full((2, 2), 1e-10)
        updated = md.update_velocities(velocity, np.full((2, 2), 1e4), np.full((2, 2), 2e4), 1e-14)
        assert_almost_equal(updated * 1e10, np.full((2, 2), 2.5))
        assert_almost_equal(velocity * 1e10, np.full((2, 2), 1.0))

    def test_conserves_energy_to_second_order(self):
        # The cut-off is moved beyond every minimum-image separation so that
        # truncation adds no energy jumps, leaving only the integrator's
        # error, which is second order in the timestep: halving the
        # timestep over the same simulated time cuts the drift by about
        # four. Measured: 1.6e-4 at 1e-14 s, 3.9e-5 at 5e-15 s.
        def worst_drift(timestep, steps):
            a = MDSimulation.initialise(25, 100, 20, timestep=timestep, seed=0, **ARGON_MODEL)
            a.cut_off = 1e-8
            a.forces = a.configuration.forces(a.pair_potentials, a.cut_off)
            initial = kinetic_plus_potential(a)
            drift = 0.0
            for _ in range(steps):
                a.step()
                drift = max(drift, abs(kinetic_plus_potential(a) - initial) / abs(initial))
            return drift

        coarse = worst_drift(1e-14, 200)
        fine = worst_drift(5e-15, 400)
        self.assertLess(coarse, 5e-4)
        self.assertLess(fine, coarse / 3)

    def test_conserves_momentum(self):
        # The pair forces are equal and opposite, so the total momentum,
        # zero after initialisation, stays zero to rounding.
        a = MDSimulation.initialise(25, 100, 20, seed=0, **ARGON_MODEL)
        thermal_speed = np.sqrt(BOLTZMANN * 100 / (ARGON.mass * ATOMIC_MASS_UNIT))
        for _ in range(200):
            a.step()
        momentum = a.configuration.velocity.sum(axis=0)
        self.assertLess(abs(momentum[0]), 1e-12 * thermal_speed)
        self.assertLess(abs(momentum[1]), 1e-12 * thermal_speed)


class TestMSD(unittest.TestCase):
    def test_is_zero_before_the_first_step(self):
        a = MDSimulation.initialise(16, 300, 20, **ARGON_MODEL)
        self.assertEqual(a.configuration.msd(a.initial_configuration), 0.0)

    def test_with_sparse_sampling(self):
        # Both particles are driven in -x at 1e4 m/s, 1 Angstrom per step, so
        # each crosses the periodic boundary of the 8 Angstrom box several
        # times in 60 steps with no sampling in between. Both share the same
        # x, so the pair force acts only along y and the x motion is the
        # imposed drift. The oracle accumulates the minimum-image
        # displacement between consecutive steps, which is exact while a
        # particle moves less than half a box per step.
        a = MDSimulation(two_argon([[-1e4, 0.0], [-1e4, 0.0]]), ARGON_MODEL["pair_potentials"])
        box = a.configuration.box
        total = np.zeros((2, 2))
        for _ in range(60):
            before = a.configuration.position
            a.step()
            displacement = a.configuration.position - before
            total += displacement - box * np.round(displacement / box)
        self.assertTrue(np.all(total[:, 0] < -5e-10))
        expected = np.mean(np.sum(total**2, axis=1))
        assert_almost_equal(a.configuration.msd(a.initial_configuration) * 1e20, expected * 1e20)


class TestHeatBath(unittest.TestCase):
    def test_rescales_to_the_bath_temperature(self):
        a = MDSimulation.initialise(10, 300, 20, **ARGON_MODEL)
        a.heat_bath(250.0)
        assert_almost_equal(a.configuration.temperature() / 250.0, 1.0)

    def test_preserves_velocity_directions_and_the_forces(self):
        a = MDSimulation.initialise(10, 300, 20, **ARGON_MODEL)
        old = a.configuration.velocity
        forces = a.forces
        a.heat_bath(250.0)
        ratio = a.configuration.velocity / old
        assert_almost_equal(ratio, np.full(ratio.shape, ratio[0, 0]))
        self.assertIs(a.forces, forces)

    def test_two_calls_each_hit_their_own_target(self):
        c = MDSimulation.initialise(10, 300, 20, **ARGON_MODEL).configuration
        c = md.heat_bath(md.heat_bath(c, 250.0), 100.0)
        assert_almost_equal(c.temperature() / 100.0, 1.0)

    def test_raises_when_the_particles_are_at_rest(self):
        with self.assertRaisesRegex(ValueError, "at rest"):
            md.heat_bath(two_argon(np.zeros((2, 2))), 250.0)

    def test_raises_when_the_temperature_is_not_finite(self):
        for bad in (np.inf, np.nan):
            with self.assertRaisesRegex(ValueError, "diverged"):
                md.heat_bath(two_argon([[bad, 0.0], [1.0, 0.0]]), 250.0)

    def test_raises_for_a_non_positive_bath_temperature(self):
        c = two_argon([[3e2, 0.0], [-3e2, 0.0]])
        for bad in (0.0, -5.0, np.nan):
            with self.assertRaises(ValueError):
                md.heat_bath(c, bad)


class TestSample(unittest.TestCase):
    def test_records_the_step_and_the_thermodynamics(self):
        a = MDSimulation.initialise(2, 300, 8, **ARGON_MODEL)
        for _ in range(3):
            a.step()
        a.sample()
        assert_equal(a.samples.step, [3])
        for name in ("temperature", "pressure", "potential_energy", "kinetic_energy", "msd"):
            self.assertEqual(getattr(a.samples, name).size, 1)
        for _ in range(4):
            a.step()
        a.sample()
        assert_equal(a.samples.step, [3, 7])
        self.assertEqual(a.samples.total_energy.size, 2)

    def test_measures_the_current_configuration(self):
        a = MDSimulation.initialise(20, 300, 20, seed=0, **ARGON_MODEL)
        for _ in range(5):
            a.step()
        a.sample()
        c = a.configuration
        temperature = c.temperature()
        virial = c.virial(a.pair_potentials, a.cut_off)
        samples = a.samples
        assert_almost_equal(samples.temperature[-1], temperature)
        assert_almost_equal(
            samples.pressure[-1],
            pairwise.calculate_pressure(virial, c.box, c.number_of_particles, temperature),
        )
        potential = c.potential_energy(a.pair_potentials, a.cut_off)
        assert_almost_equal(samples.potential_energy[-1], potential)
        assert_almost_equal(samples.kinetic_energy[-1], c.kinetic_energy())
        assert_almost_equal(samples.total_energy[-1], potential + c.kinetic_energy())
        self.assertGreater(samples.msd[-1], 0.0)


class TestRestart(unittest.TestCase):
    def run_and_sample(self, a, steps):
        for _ in range(steps):
            a.step()
            a.sample()

    def test_starts_a_fresh_record_from_the_current_state(self):
        a = MDSimulation.initialise(4, 300, 12, **ARGON_MODEL)
        self.run_and_sample(a, 5)
        production = a.restart()
        self.assertIsNot(production, a)
        self.assertIsInstance(production, MDSimulation)
        self.assertEqual(production.steps, 0)
        self.assertEqual(production.time, 0.0)
        self.assertIsInstance(production.samples, md.MDSamples)
        self.assertEqual(production.samples.step.size, 0)
        assert_equal(production.configuration.position, a.configuration.position)
        assert_equal(production.configuration.unwrapped, a.configuration.position)
        self.assertIs(production.initial_configuration, production.configuration)
        self.assertEqual(production.configuration.msd(production.initial_configuration), 0.0)
        assert_equal(production.forces, a.forces)
        # The restarted simulation follows the same trajectory as the source
        # while its displacement is measured from the restart.
        a.step()
        production.step()
        assert_equal(production.configuration.position, a.configuration.position)
        production.sample()
        self.assertGreater(production.samples.msd[-1], 0.0)

    def test_leaves_the_source_alone(self):
        a = MDSimulation.initialise(4, 300, 12, **ARGON_MODEL)
        self.run_and_sample(a, 3)
        source = a.configuration
        production = a.restart()
        production.step()
        self.assertIs(a.configuration, source)
        self.assertEqual(a.steps, 3)
        self.assertEqual(a.samples.msd.size, 3)
