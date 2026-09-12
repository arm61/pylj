import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from pylj import md, pairwise, placement
from pylj.configuration import MDConfiguration
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN
from pylj.md import MDSimulation
from pylj.tests.argon import ARGON, ARGON_MODEL, LARGER, MIXTURE_MODEL, WELL_MODEL


def two_argon(velocity, box=8e-10):
    """Two argon atoms at (2, 2) and (2, 6) Angstrom with the given velocities."""
    position = np.array([[2e-10, 2e-10], [2e-10, 6e-10]])
    return MDConfiguration(
        positions=position,
        species=(ARGON,),
        species_index=np.zeros(2, dtype=np.int64),
        box=box,
        velocities=np.asarray(velocity, dtype=float),
        unwrapped=position.copy(),
    )


def drift_speed(configuration):
    """The speed of the centre of mass, in m/s."""
    masses = configuration.masses[:, None]
    return float(
        np.linalg.norm((masses * configuration.velocities).sum(axis=0) / masses.sum())
    )


def kinetic_plus_potential(sim):
    """The total energy of a simulation, from its configuration and stored cut-off."""
    c = sim.configuration
    return c.kinetic_energy() + c.potential_energy(sim.model, sim.cut_off)


class TestInitialise(unittest.TestCase):
    def test_square_lattice_in_a_converted_box(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=2, temperature=300, box=8)
        c = a.configuration
        self.assertIsInstance(c, MDConfiguration)
        self.assertEqual(c.number_of_atoms, 2)
        assert_almost_equal(c.box, 8e-10)
        assert_almost_equal(c.positions * 1e10, [[2, 2], [2, 6]])
        assert_almost_equal(c.unwrapped, c.positions)
        assert_almost_equal(a.cut_off * 1e10, 4.0)
        assert_almost_equal(a.timestep, 1e-14)
        self.assertEqual(a.steps, 0)
        self.assertEqual(a.time, 0.0)

    def test_forces_are_valid_after_construction(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=2, temperature=300, box=8)
        assert_allclose(a.forces, a.configuration.forces(a.model, a.cut_off))
        self.assertNotEqual(a.forces[0, 1], 0.0)

    def test_velocities_have_no_net_momentum(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=25, temperature=100, box=40)
        thermal_speed = np.sqrt(BOLTZMANN * 100 / (ARGON.mass * ATOMIC_MASS_UNIT))
        momentum = a.configuration.velocities.sum(axis=0)
        self.assertLess(abs(momentum[0]), 1e-12 * thermal_speed)
        self.assertLess(abs(momentum[1]), 1e-12 * thermal_speed)

    def test_velocities_match_the_requested_temperature(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=25, temperature=100, box=40)
        assert_almost_equal(a.configuration.temperature(), 100)

    def test_two_species_have_no_net_momentum_at_the_temperature(self):
        # The centre-of-mass velocity is mass weighted, so with unequal
        # masses the total momentum is zero and the temperature exact.
        a = MDSimulation.initialise(MIXTURE_MODEL, number_of_atoms=24, temperature=100, box=60)
        c = a.configuration
        assert_allclose(c.masses[:4], np.array([39.948, 80.0, 39.948, 80.0]) * ATOMIC_MASS_UNIT)
        momentum_scale = ARGON.mass * np.sqrt(BOLTZMANN * 100 / (ARGON.mass * ATOMIC_MASS_UNIT))
        momentum = (c.masses[:, None] * c.velocities).sum(axis=0)
        self.assertLess(abs(momentum[0]), 1e-12 * momentum_scale)
        self.assertLess(abs(momentum[1]), 1e-12 * momentum_scale)
        assert_almost_equal(c.temperature(), 100)

    def test_heavier_atoms_move_more_slowly(self):
        # Each species gets its own thermal width: the mean square speed of
        # a species is 2 k_B T / m, so the argon atoms move faster than
        # the heavier ones. A thousand atoms keep the sampling noise
        # to a few per cent.
        c = MDSimulation.initialise(
            MIXTURE_MODEL, number_of_atoms=1000, temperature=100, box=320, seed=0
        ).configuration
        for index, species in enumerate((ARGON, LARGER)):
            speeds_squared = np.sum(c.velocities[c.species_index == index] ** 2, axis=1)
            expected = 2 * BOLTZMANN * 100 / (species.mass * ATOMIC_MASS_UNIT)
            assert_allclose(speeds_squared.mean(), expected, rtol=0.1)

    def test_refuses_a_potential_with_no_force(self):
        with self.assertRaisesRegex(ValueError, "Monte Carlo"):
            MDSimulation.initialise(WELL_MODEL, number_of_atoms=2, temperature=300, box=8)

    def test_is_reproducible_with_a_seed(self):
        def build(seed):
            return MDSimulation.initialise(
                ARGON_MODEL,
                number_of_atoms=10,
                temperature=100,
                box=40,
                init_conf="metropolis",
                seed=seed,
            )

        first, second, other = build(3), build(3), build(4)
        assert_equal(first.configuration.positions, second.configuration.positions)
        assert_equal(first.configuration.velocities, second.configuration.velocities)
        self.assertFalse(
            np.array_equal(first.configuration.velocities, other.configuration.velocities)
        )
        # The generator continues from the placement and velocity draws.
        self.assertEqual(first.rng.random(), second.rng.random())

    def test_passes_the_placement_temperature_through(self):
        # 50 argon in a 27 Angstrom box cannot be placed at 1 K.
        with self.assertRaisesRegex(ValueError, "Could not place"):
            MDSimulation.initialise(
                ARGON_MODEL,
                number_of_atoms=50,
                temperature=100,
                box=27,
                init_conf="metropolis",
                placement_temperature=1.0,
                seed=0,
            )

    def test_passes_the_timestep_and_cut_off_through(self):
        a = MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=2, temperature=300, box=40, timestep=2e-15, cut_off=10
        )
        assert_almost_equal(a.timestep, 2e-15)
        assert_almost_equal(a.cut_off * 1e10, 10)

    def test_one_atom_raises(self):
        with self.assertRaisesRegex(ValueError, "at least two atoms"):
            MDSimulation.initialise(ARGON_MODEL, number_of_atoms=1, temperature=300, box=8)

    def test_rejects_a_non_positive_or_infinite_temperature(self):
        for temperature in (0, -10, np.inf):
            with self.assertRaisesRegex(ValueError, "temperature must be positive"):
                MDSimulation.initialise(
                    ARGON_MODEL, number_of_atoms=2, temperature=temperature, box=8
                )

    def test_initialise_on_a_triangular_lattice(self):
        simulation = MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=56, temperature=100, box=60, init_conf="triangular"
        )
        self.assertEqual(simulation.configuration.number_of_atoms, 56)

    def test_initialise_passes_max_strain_to_the_placement(self):
        with self.assertRaisesRegex(ValueError, "max_strain"):
            MDSimulation.initialise(
                ARGON_MODEL,
                number_of_atoms=100,
                temperature=100,
                box=60,
                init_conf="triangular",
            )
        simulation = MDSimulation.initialise(
            ARGON_MODEL,
            number_of_atoms=100,
            temperature=100,
            box=60,
            init_conf="triangular",
            max_strain=0.2,
        )
        self.assertEqual(simulation.configuration.number_of_atoms, 100)


class TestConstructor(unittest.TestCase):
    def test_refuses_a_configuration_without_velocities(self):
        c = placement.place_square(2, (ARGON,), 8e-10)
        self.assertNotIsInstance(c, MDConfiguration)
        with self.assertRaisesRegex(TypeError, "MDConfiguration"):
            MDSimulation(c, ARGON_MODEL)

    def test_refuses_a_bad_timestep(self):
        c = two_argon([[3e2, 0.0], [-3e2, 0.0]])
        for bad in (0.0, -1e-14, np.nan):
            with self.assertRaisesRegex(ValueError, "timestep must be positive"):
                MDSimulation(c, ARGON_MODEL, timestep=bad)

    def test_builds_from_a_configuration_at_rest(self):
        a = MDSimulation(two_argon(np.zeros((2, 2))), ARGON_MODEL)
        a.step()
        self.assertEqual(a.steps, 1)

    def test_takes_a_ready_configuration(self):
        # Already at rest, so the constructor's at_rest call leaves its
        # velocities and positions unchanged.
        c = two_argon([[3e2, 0.0], [-3e2, 0.0]])
        a = MDSimulation(c, ARGON_MODEL, timestep=2e-15, seed=5)
        assert_allclose(a.configuration.velocities, c.velocities)
        assert_allclose(a.configuration.positions, c.positions)
        self.assertIs(a.configuration, a.initial_configuration)
        assert_almost_equal(a.timestep, 2e-15)
        assert_allclose(a.forces, c.forces(a.model, a.cut_off))


class TestStep(unittest.TestCase):
    def test_step_advances_the_clock(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        a.step()
        a.step()
        self.assertEqual(a.steps, 2)
        assert_almost_equal(a.time, 2 * a.timestep)

    def test_step_uses_the_integrate_method(self):
        class Frozen(MDSimulation):
            def integrate(self):
                pass

        a = Frozen.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        before = a.configuration
        a.step()
        self.assertIs(a.configuration, before)
        self.assertEqual(a.steps, 1)

    def test_integrate_replaces_the_configuration_and_the_forces(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        before, forces_before = a.configuration, a.forces
        a.integrate()
        self.assertIsNot(a.configuration, before)
        self.assertFalse(np.array_equal(a.configuration.positions, before.positions))
        assert_allclose(a.forces, a.configuration.forces(a.model, a.cut_off))
        self.assertFalse(np.array_equal(a.forces, forces_before))
        self.assertEqual(a.steps, 0)


class TestVelocityVerlet(unittest.TestCase):
    def test_advances_the_unwrapped_positions(self):
        # A y velocity of 3e4 m/s moves each atom by 3 Angstrom in one
        # 1e-14 s step, so the second atom crosses the boundary of the 8
        # Angstrom box: its wrapped position comes back in and its unwrapped
        # one does not. The forces are zeroed so the motion is the drift.
        c = two_argon([[0.0, 3e4], [0.0, 3e4]])
        moved, forces = md.velocity_verlet(c, np.zeros((2, 2)), 1e-14, ARGON_MODEL, 15e-10)
        assert_almost_equal(moved.unwrapped * 1e10, [[2, 5], [2, 9]])
        assert_almost_equal(moved.positions * 1e10, [[2, 5], [2, 1]])
        assert_allclose(forces, moved.forces(ARGON_MODEL, 15e-10))

    def test_matches_the_hand_computed_step(self):
        # A pair 4 Angstrom apart, inside the cut-off, one atom drifting
        # in x: the attraction acts along y, so the positions advance by
        # v dt + a dt^2 / 2 and the velocities by the mean acceleration
        # times dt, with the forces at the new positions evaluated afresh.
        c = two_argon([[1e3, 0.0], [0.0, 0.0]], box=40e-10)
        cut_off = 15e-10
        forces = c.forces(ARGON_MODEL, cut_off)
        moved, next_forces = md.velocity_verlet(c, forces, 1e-14, ARGON_MODEL, cut_off)
        accelerations = forces / c.masses[:, None]
        expected_position = c.positions + c.velocities * 1e-14 + 0.5 * accelerations * 1e-28
        assert_allclose(moved.positions, expected_position)
        expected_forces = c.replace(positions=expected_position).forces(ARGON_MODEL, cut_off)
        assert_allclose(next_forces, expected_forces)
        next_accelerations = expected_forces / c.masses[:, None]
        expected_velocity = c.velocities + 0.5 * (accelerations + next_accelerations) * 1e-14
        assert_allclose(moved.velocities, expected_velocity)
        self.assertNotEqual(moved.velocities[0, 1], 0.0)

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
        # four. The mixture checks that each species is moved with its own
        # mass. Measured for argon: 1.6e-4 at 1e-14 s, 3.9e-5 at 5e-15 s;
        # for the mixture: 3.0e-3 and 7.6e-4.
        def worst_drift(model, box, timestep, steps):
            a = MDSimulation.initialise(
                model, number_of_atoms=25, temperature=100, box=box, timestep=timestep, seed=0
            )
            a.cut_off = 1e-8
            a.forces = a.configuration.forces(a.model, a.cut_off)
            initial = kinetic_plus_potential(a)
            drift = 0.0
            for _ in range(steps):
                a.step()
                drift = max(drift, abs(kinetic_plus_potential(a) - initial) / abs(initial))
            return drift

        for model, box, limit in ((ARGON_MODEL, 20, 5e-4), (MIXTURE_MODEL, 30, 5e-3)):
            coarse = worst_drift(model, box, 1e-14, 200)
            fine = worst_drift(model, box, 5e-15, 400)
            self.assertLess(coarse, limit)
            self.assertLess(fine, coarse / 3)

    def test_conserves_momentum(self):
        # The pair forces are equal and opposite, so the total momentum,
        # zero after initialisation, stays zero to rounding. With two
        # masses only the mass-weighted sum is conserved.
        momentum_scale = (
            ARGON.mass
            * ATOMIC_MASS_UNIT
            * np.sqrt(BOLTZMANN * 100 / (ARGON.mass * ATOMIC_MASS_UNIT))
        )
        for model, box in ((ARGON_MODEL, 20), (MIXTURE_MODEL, 30)):
            a = MDSimulation.initialise(model, number_of_atoms=25, temperature=100, box=box, seed=0)
            for _ in range(200):
                a.step()
            c = a.configuration
            momentum = (c.masses[:, None] * c.velocities).sum(axis=0)
            self.assertLess(abs(momentum[0]), 1e-12 * momentum_scale)
            self.assertLess(abs(momentum[1]), 1e-12 * momentum_scale)

    def test_refuses_a_step_that_moves_a_atom_past_half_the_cut_off(self):
        # A timestep a thousand times too long carries an atom tens of
        # Angstrom in one step; the integrator refuses rather than continue
        # from a configuration that is no longer meaningful.
        a = MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=25, temperature=100, box=20, timestep=1e-11, seed=0
        )
        with self.assertRaisesRegex(ValueError, "half the cut-off"):
            a.step()
        self.assertTrue(np.isfinite(a.configuration.positions).all())
        self.assertTrue(np.isfinite(a.configuration.velocities).all())


class TestMSD(unittest.TestCase):
    def test_is_zero_before_the_first_step(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=16, temperature=300, box=20)
        self.assertEqual(a.configuration.msd(a.initial_configuration), 0.0)

    def test_with_sparse_sampling(self):
        # The atoms move in opposite directions at 1e4 m/s, 1 Angstrom per
        # step, so each crosses the periodic boundary of the 8 Angstrom box
        # several times in 60 steps with no sampling in between. Their y
        # separation is 4 Angstrom, which is exactly the cut-off, so the
        # pair is inside it only when their minimum-image x separation
        # returns to zero, every fourth step, and the force is then purely
        # along y. The x motion is the imposed velocity throughout. The
        # oracle accumulates the minimum-image displacement between
        # consecutive steps, which is exact while an atom moves less than
        # half a box per step.
        a = MDSimulation(two_argon([[1e4, 0.0], [-1e4, 0.0]]), ARGON_MODEL)
        box = a.configuration.box
        total = np.zeros((2, 2))
        for _ in range(60):
            before = a.configuration.positions
            a.step()
            displacement = a.configuration.positions - before
            total += displacement - box * np.round(displacement / box)
        self.assertGreater(total[0, 0], 5e-10)
        self.assertLess(total[1, 0], -5e-10)
        expected = np.mean(np.sum(total**2, axis=1))
        assert_almost_equal(a.configuration.msd(a.initial_configuration) * 1e20, expected * 1e20)


class TestHeatBath(unittest.TestCase):
    def test_rescales_to_the_bath_temperature(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=10, temperature=300, box=20)
        a.heat_bath(250.0)
        assert_almost_equal(a.configuration.temperature() / 250.0, 1.0)

    def test_preserves_velocity_directions_and_the_forces(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=10, temperature=300, box=20)
        old = a.configuration.velocities
        forces = a.forces
        a.heat_bath(250.0)
        ratio = a.configuration.velocities / old
        assert_almost_equal(ratio, np.full(ratio.shape, ratio[0, 0]))
        self.assertIs(a.forces, forces)

    def test_two_calls_each_hit_their_own_target(self):
        c = MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=10, temperature=300, box=20
        ).configuration
        c = md.heat_bath(md.heat_bath(c, 250.0), 100.0)
        assert_almost_equal(c.temperature() / 100.0, 1.0)

    def test_raises_when_the_atoms_are_at_rest(self):
        with self.assertRaisesRegex(ValueError, "at rest"):
            md.heat_bath(two_argon(np.zeros((2, 2))), 250.0)

    def test_raises_when_the_temperature_is_not_finite(self):
        for bad in (np.inf, np.nan):
            with self.assertRaisesRegex(ValueError, "diverged"):
                md.heat_bath(two_argon([[bad, 0.0], [1.0, 0.0]]), 250.0)

    def test_raises_for_a_non_positive_bath_temperature(self):
        c = two_argon([[3e2, 0.0], [-3e2, 0.0]])
        for bad in (0.0, -5.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                md.heat_bath(c, bad)


class TestSample(unittest.TestCase):
    def test_records_the_step_and_the_thermodynamics(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=2, temperature=300, box=8)
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
        a = MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=20, temperature=300, box=20, seed=0
        )
        for _ in range(5):
            a.step()
        a.sample()
        c = a.configuration
        temperature = c.temperature()
        virial = c.virial(a.model, a.cut_off)
        samples = a.samples
        assert_almost_equal(samples.temperature[-1], temperature)
        assert_almost_equal(
            samples.pressure[-1],
            pairwise.calculate_pressure(virial, c.box, c.kinetic_energy()),
        )
        potential = c.potential_energy(a.model, a.cut_off)
        assert_almost_equal(samples.potential_energy[-1], potential)
        assert_almost_equal(samples.kinetic_energy[-1], c.kinetic_energy())
        assert_almost_equal(samples.total_energy[-1], potential + c.kinetic_energy())
        self.assertGreater(samples.msd[-1], 0.0)

    def test_appends_the_configuration_to_the_trajectory(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        self.assertEqual(len(a.trajectory), 0)
        a.sample()
        a.step()
        a.sample()
        self.assertEqual(len(a.trajectory), 2)
        self.assertIs(a.trajectory[1], a.configuration)


class TestRestart(unittest.TestCase):
    def run_and_sample(self, a, steps):
        for _ in range(steps):
            a.step()
            a.sample()

    def test_starts_a_fresh_record_from_the_current_state(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=300, box=12)
        self.run_and_sample(a, 5)
        production = a.restart()
        self.assertIsNot(production, a)
        self.assertIsInstance(production, MDSimulation)
        self.assertEqual(production.steps, 0)
        self.assertEqual(production.time, 0.0)
        self.assertIsInstance(production.samples, md.MDSamples)
        self.assertEqual(production.samples.step.size, 0)
        assert_equal(production.configuration.positions, a.configuration.positions)
        assert_equal(production.configuration.unwrapped, a.configuration.positions)
        self.assertIs(production.initial_configuration, production.configuration)
        self.assertEqual(production.configuration.msd(production.initial_configuration), 0.0)
        assert_equal(production.forces, a.forces)
        # The restarted simulation follows the same trajectory as the source
        # while its displacement is measured from the restart.
        a.step()
        production.step()
        assert_equal(production.configuration.positions, a.configuration.positions)
        production.sample()
        self.assertGreater(production.samples.msd[-1], 0.0)

    def test_leaves_the_source_alone(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=300, box=12)
        self.run_and_sample(a, 3)
        source = a.configuration
        production = a.restart()
        production.step()
        self.assertIs(a.configuration, source)
        self.assertEqual(a.steps, 3)
        self.assertEqual(a.samples.msd.size, 3)

    def test_starts_an_empty_trajectory(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        a.sample()
        production = a.restart()
        self.assertEqual(len(production.trajectory), 0)
        self.assertEqual(len(a.trajectory), 1)


class TestAtRest(unittest.TestCase):
    def drifting(self):
        # A mixture: with one species the plain mean of the velocities
        # equals the mass-weighted mean, so an at_rest using the plain mean
        # would pass every test here.
        configuration = MDSimulation.initialise(
            MIXTURE_MODEL, number_of_atoms=8, temperature=100, box=20, seed=1
        ).configuration
        return configuration.replace(velocities=configuration.velocities + [10.0, -4.0])

    def test_removes_the_drift(self):
        moving = self.drifting()
        self.assertGreater(drift_speed(moving), 1.0)
        self.assertLess(drift_speed(md.at_rest(moving)), 1e-9)

    def test_leaves_the_positions_alone(self):
        moving = self.drifting()
        assert_allclose(md.at_rest(moving).positions, moving.positions)

    def test_leaves_every_relative_velocity_alone(self):
        moving = self.drifting()
        rested = md.at_rest(moving)
        assert_allclose(
            rested.velocities - rested.velocities[0], moving.velocities - moving.velocities[0]
        )

    def test_the_kinetic_energy_falls_by_the_drift_energy(self):
        moving = self.drifting()
        speed = drift_speed(moving)
        total_mass = moving.masses.sum()
        assert_allclose(
            moving.kinetic_energy() - md.at_rest(moving).kinetic_energy(),
            0.5 * total_mass * speed**2,
        )

    def test_is_idempotent(self):
        moving = self.drifting()
        once = md.at_rest(moving)
        assert_allclose(md.at_rest(once).velocities, once.velocities)

    def test_the_constructor_sets_the_centre_of_mass_at_rest(self):
        moving = self.drifting()
        simulation = md.MDSimulation(moving, MIXTURE_MODEL, timestep=1e-14)
        self.assertLess(drift_speed(simulation.configuration), 1e-9)

    def test_dropping_an_atom_leaves_the_simulation_at_rest(self):
        # Removing an atom takes its momentum with it, so the rest drift.
        started = MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=16, temperature=100, box=25, seed=1
        )
        vacancy = started.configuration.without(0)
        self.assertGreater(drift_speed(vacancy), 0.5)
        simulation = md.MDSimulation(vacancy, ARGON_MODEL, timestep=1e-14)
        self.assertLess(drift_speed(simulation.configuration), 1e-9)

    def test_initialise_still_reports_its_target_temperature(self):
        simulation = MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=16, temperature=137.0, box=25, seed=1
        )
        assert_allclose(simulation.configuration.temperature(), 137.0)
