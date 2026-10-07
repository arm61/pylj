import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from pylj import md, pairwise, placement
from pylj.configuration import MDConfiguration
from pylj.constants import BOLTZMANN, KJ_PER_MOL
from pylj.md import MDSimulation
from pylj.tests.argon import ARGON, ARGON_MODEL, LARGER, MIXTURE_MODEL, WELL_MODEL


def two_argon(velocity, box=8.0):
    """Two argon atoms at (2, 2) and (2, 6) Angstrom with the given velocities."""
    position = np.array([[2.0, 2.0], [2.0, 6.0]])
    return MDConfiguration(
        positions=position,
        species=(ARGON,),
        species_index=np.zeros(2, dtype=np.int64),
        box=box,
        velocities=velocity,
    )


def drift_speed(configuration):
    """The speed of the centre of mass, in Angstrom per picosecond."""
    masses = configuration.masses[:, None]
    return float(
        np.linalg.norm((masses * configuration.velocities).sum(axis=0) / masses.sum())
    )


def kinetic_plus_potential(sim):
    """The total energy of a simulation, from its configuration and stored cut-off."""
    c = sim.configuration
    return c.kinetic_energy() + c.potential_energy(sim.model, sim.cut_off)


class TestInitialise(unittest.TestCase):
    def test_square_lattice_in_the_box(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=2, temperature=300, box=8)
        c = a.configuration
        self.assertIsInstance(c, MDConfiguration)
        self.assertEqual(c.number_of_atoms, 2)
        assert_almost_equal(c.box, 8.0)
        assert_almost_equal(c.positions, [[2, 4], [6, 4]])
        assert_almost_equal(c.unwrapped, c.positions)
        assert_almost_equal(a.cut_off, 4.0)
        assert_almost_equal(a.timestep, 0.01)
        self.assertEqual(a.steps, 0)
        self.assertEqual(a.time, 0.0)

    def test_forces_are_valid_after_construction(self):
        a = MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=4, temperature=300, box=12, init_conf="metropolis", seed=0
        )
        assert_allclose(a.forces, a.configuration.forces(a.model, a.cut_off))
        self.assertTrue(np.all(a.forces != 0.0))

    def test_velocities_have_no_net_momentum(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=25, temperature=100, box=40)
        thermal_speed = np.sqrt(BOLTZMANN * 100 * KJ_PER_MOL / ARGON.mass)
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
        assert_allclose(c.masses[:4], [39.948, 80.0, 39.948, 80.0])
        momentum_scale = ARGON.mass * np.sqrt(BOLTZMANN * 100 * KJ_PER_MOL / ARGON.mass)
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
            expected = 2 * BOLTZMANN * 100 * KJ_PER_MOL / species.mass
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
            ARGON_MODEL, number_of_atoms=2, temperature=300, box=40, timestep=0.002, cut_off=10
        )
        assert_almost_equal(a.timestep, 0.002)
        assert_almost_equal(a.cut_off, 10)

    def test_one_atom_raises(self):
        with self.assertRaisesRegex(ValueError, "at least two atoms"):
            MDSimulation.initialise(ARGON_MODEL, number_of_atoms=1, temperature=300, box=8)

    def test_starts_at_rest_at_zero_temperature(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=0, box=12)
        assert_equal(a.configuration.velocities, 0.0)

    def test_rejects_a_negative_or_infinite_temperature(self):
        for temperature in (-10, np.inf):
            with self.assertRaisesRegex(ValueError, "temperature must be non-negative"):
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
        c = placement.place_square(2, (ARGON,), 8.0)
        self.assertNotIsInstance(c, MDConfiguration)
        with self.assertRaisesRegex(TypeError, "MDConfiguration"):
            MDSimulation(c, ARGON_MODEL)

    def test_refuses_a_bad_timestep(self):
        c = two_argon([[3.0, 0.0], [-3.0, 0.0]])
        for bad in (0.0, -0.01, np.nan):
            with self.assertRaisesRegex(ValueError, "timestep must be positive"):
                MDSimulation(c, ARGON_MODEL, timestep=bad)

    def test_builds_from_a_configuration_at_rest(self):
        a = MDSimulation(two_argon(np.zeros((2, 2))), ARGON_MODEL)
        a.step()
        self.assertTrue((a.configuration.velocities != 0).any())

    def test_takes_a_ready_configuration(self):
        # Already at rest, so removing the drift leaves its velocities and
        # positions unchanged.
        c = two_argon([[3.0, 0.0], [-3.0, 0.0]])
        a = MDSimulation(c, ARGON_MODEL, timestep=0.002, seed=5)
        assert_allclose(a.configuration.velocities, c.velocities)
        assert_allclose(a.configuration.positions, c.positions)
        assert_almost_equal(a.timestep, 0.002)
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
        before = a.configuration.positions.copy()
        a.step()
        assert_equal(a.configuration.positions, before)
        self.assertEqual(a.steps, 1)

    def test_integrate_moves_the_atoms_and_updates_the_forces(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        positions_before, forces_before = a.configuration.positions, a.forces
        a.integrate()
        self.assertFalse(np.array_equal(a.configuration.positions, positions_before))
        assert_allclose(a.forces, a.configuration.forces(a.model, a.cut_off))
        self.assertFalse(np.array_equal(a.forces, forces_before))
        self.assertEqual(a.steps, 0)

    def test_computes_the_forces_once_a_step(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        with mock.patch.object(
            MDConfiguration, "forces", autospec=True, side_effect=MDConfiguration.forces
        ) as forces:
            for _ in range(3):
                a.step()
        self.assertEqual(forces.call_count, 3)

    def test_leaves_an_array_taken_before_the_step_alone(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        positions, velocities = a.configuration.positions, a.configuration.velocities
        held = positions.copy(), velocities.copy()
        a.step()
        assert_equal(positions, held[0])
        assert_equal(velocities, held[1])


class TestHandChanges(unittest.TestCase):
    def build(self):
        return MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=16, temperature=90, box=20, seed=1
        )

    def test_a_hand_change_runs_as_a_new_simulation_from_the_changed_state(self):
        a = self.build()
        for _ in range(10):
            a.step()
        nudge = np.random.default_rng(0).uniform(-0.3, 0.3, size=(16, 2))
        a.configuration.positions = (a.configuration.positions + nudge) % 20
        fresh = MDSimulation(a.configuration, ARGON_MODEL, timestep=a.timestep)
        for _ in range(20):
            a.step()
            fresh.step()
        assert_allclose(a.configuration.positions, fresh.configuration.positions, atol=1e-12)
        assert_allclose(a.configuration.velocities, fresh.configuration.velocities, atol=1e-12)

    def test_reading_the_forces_after_a_hand_change_recomputes_them(self):
        a = self.build()
        a.configuration.positions[0] += [0.3, -0.2]
        assert_allclose(a.forces, a.configuration.forces(a.model, a.cut_off))

    def test_assigning_a_configuration_recomputes_the_forces(self):
        a = self.build()
        moved = a.configuration.copy()
        moved.positions = (moved.positions + 0.25) % 20
        a.configuration = moved
        assert_allclose(a.forces, moved.forces(a.model, a.cut_off))

    def test_a_changed_number_of_atoms_or_box_raises_at_the_next_step(self):
        a = self.build()
        a.configuration = a.configuration.without(0)
        with self.assertRaisesRegex(ValueError, "number of atoms has changed from 16 to 15"):
            a.step()
        a = self.build()
        a.configuration.box = 14.0
        a.configuration.positions = a.configuration.positions * 0.7
        with self.assertRaisesRegex(ValueError, "box has changed from 20 to 14 Angstrom"):
            a.step()

    def test_two_simulations_from_one_configuration_run_independently(self):
        c = self.build().configuration
        first = MDSimulation(c, ARGON_MODEL)
        second = MDSimulation(c, ARGON_MODEL)
        first.step()
        assert_equal(second.configuration.positions, c.positions)


class TestVelocityVerlet(unittest.TestCase):
    def test_wraps_the_positions_and_counts_the_crossings(self):
        # Two atoms half an Angstrom inside the top and bottom edges of an
        # 8 Angstrom box move 1 Angstrom in one step, outwards, so each
        # crosses an edge: its position comes back in at the opposite
        # edge, its images count the crossing, and its unwrapped position
        # does not come back. The pair is 4.1 Angstrom apart before and
        # after the step, beyond the 3.9 Angstrom cut-off, so there is no
        # force and the atoms move at their own velocities. The velocities
        # are equal and opposite, so the constructor leaves them alone.
        c = MDConfiguration(
            positions=[[2.0, 7.5], [6.0, 0.5]],
            species=(ARGON,),
            species_index=[0, 0],
            box=8.0,
            velocities=[[0.0, 100.0], [0.0, -100.0]],
        )
        a = MDSimulation(c, ARGON_MODEL, cut_off=3.9)
        a.step()
        assert_almost_equal(a.configuration.positions, [[2.0, 0.5], [6.0, 7.5]])
        assert_equal(a.configuration.images, [[0, 1], [0, -1]])
        assert_almost_equal(a.configuration.unwrapped, [[2.0, 8.5], [6.0, -0.5]])

    def test_matches_the_hand_computed_step(self):
        # A pair 4 Angstrom apart, inside the cut-off, one atom drifting
        # in x: the attraction acts along y, so the positions advance by
        # v dt + a dt^2 / 2 and the velocities by the mean acceleration
        # times dt, with the forces at the new positions evaluated afresh.
        # A force in kJ/mol/Angstrom over a mass in amu, times KJ_PER_MOL,
        # is an acceleration in Angstrom/ps^2.
        c = two_argon([[10.0, 0.0], [0.0, 0.0]], box=40.0)
        cut_off, dt = 15.0, 0.01
        a = MDSimulation(c, ARGON_MODEL, cut_off=cut_off, timestep=dt)
        start = a.configuration.copy()
        forces = start.forces(ARGON_MODEL, cut_off)
        a.step()
        accelerations = forces / start.masses[:, None] * KJ_PER_MOL
        expected_position = start.positions + start.velocities * dt + 0.5 * accelerations * dt**2
        assert_allclose(a.configuration.positions, expected_position)
        moved = start.copy()
        moved.positions = expected_position
        expected_forces = moved.forces(ARGON_MODEL, cut_off)
        assert_allclose(a.forces, expected_forces)
        next_accelerations = expected_forces / start.masses[:, None] * KJ_PER_MOL
        expected_velocity = start.velocities + 0.5 * (accelerations + next_accelerations) * dt
        assert_allclose(a.configuration.velocities, expected_velocity)
        self.assertNotEqual(a.configuration.velocities[0, 1], 0.0)

    def test_conserves_energy_to_second_order(self):
        # The cut-off is moved beyond every minimum-image separation so that
        # truncation adds no energy jumps, leaving only the integrator's
        # error, which is second order in the timestep: halving the
        # timestep over the same simulated time cuts the drift by about
        # four. The mixture checks that each species is moved with its own
        # mass. The worst drift at 0.01 ps over 20 seeds is 6.1e-4 for argon
        # and 4.0e-3 for the mixture, and the finer timestep cuts it to at
        # most 0.26 of that.
        def worst_drift(model, box, timestep, steps):
            a = MDSimulation.initialise(
                model, number_of_atoms=25, temperature=100, box=box, timestep=timestep, seed=0
            )
            a.cut_off = 100.0
            initial = kinetic_plus_potential(a)
            drift = 0.0
            for _ in range(steps):
                a.step()
                drift = max(drift, abs(kinetic_plus_potential(a) - initial) / abs(initial))
            return drift

        for model, box, limit in ((ARGON_MODEL, 20, 1e-3), (MIXTURE_MODEL, 30, 1e-2)):
            coarse = worst_drift(model, box, 0.01, 200)
            fine = worst_drift(model, box, 0.005, 400)
            self.assertLess(coarse, limit)
            self.assertLess(fine, coarse / 3)

    def test_conserves_momentum(self):
        # The pair forces are equal and opposite, so the total momentum,
        # zero after initialisation, stays zero to rounding. With two
        # masses only the mass-weighted sum is conserved.
        momentum_scale = ARGON.mass * np.sqrt(BOLTZMANN * 100 * KJ_PER_MOL / ARGON.mass)
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
            ARGON_MODEL, number_of_atoms=25, temperature=100, box=20, timestep=10, seed=0
        )
        before = a.configuration.positions.copy()
        with self.assertRaisesRegex(ValueError, "half the cut-off"):
            a.step()
        assert_equal(a.configuration.positions, before)
        self.assertEqual(a.steps, 0)

    def test_refuses_a_step_whose_displacement_is_not_a_number(self):
        # A non-finite velocity gives a nan displacement, which the guard
        # catches only because it is spelled ``not furthest < cut_off / 2``.
        a = MDSimulation(two_argon([[np.nan, 0.0], [1.0, 0.0]]), ARGON_MODEL)
        with self.assertRaisesRegex(ValueError, "half the cut-off"):
            a.step()


class TestUnwrapped(unittest.TestCase):
    def test_follows_atoms_across_the_boundary(self):
        # The atoms move in opposite directions at 100 Angstrom/ps, 1 Angstrom
        # per step, so each crosses the periodic boundary of the 8 Angstrom box
        # several times in 60 steps with no sampling in between. Their y
        # separation is 4 Angstrom, which is exactly the cut-off, so the
        # pair is inside it only when their minimum-image x separation
        # returns to zero, every fourth step, and the force is then purely
        # along y. The x motion is the imposed velocity throughout. The
        # oracle accumulates the minimum-image displacement between
        # consecutive steps, which is exact while an atom moves less than
        # half a box per step.
        a = MDSimulation(two_argon([[100.0, 0.0], [-100.0, 0.0]]), ARGON_MODEL)
        start = a.configuration.unwrapped
        box = a.configuration.box
        total = np.zeros((2, 2))
        for _ in range(60):
            before = a.configuration.positions
            a.step()
            displacement = a.configuration.positions - before
            total += displacement - box * np.round(displacement / box)
        self.assertGreater(total[0, 0], 5.0)
        self.assertLess(total[1, 0], -5.0)
        assert_allclose(a.configuration.unwrapped - start, total, atol=1e-9)


class TestRescaleVelocities(unittest.TestCase):
    def test_rescales_to_the_temperature(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=10, temperature=300, box=20)
        a.rescale_velocities(250.0)
        assert_almost_equal(a.configuration.temperature() / 250.0, 1.0)

    def test_preserves_velocity_directions_and_the_forces(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=10, temperature=300, box=20)
        old = a.configuration.velocities
        forces = a.forces
        a.rescale_velocities(250.0)
        ratio = a.configuration.velocities / old
        assert_almost_equal(ratio, np.full(ratio.shape, ratio[0, 0]))
        self.assertIs(a.forces, forces)

    def test_two_calls_each_hit_their_own_target(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=10, temperature=300, box=20)
        a.rescale_velocities(250.0)
        a.rescale_velocities(100.0)
        assert_almost_equal(a.configuration.temperature() / 100.0, 1.0)

    def test_raises_when_the_atoms_are_at_rest(self):
        a = MDSimulation(two_argon(np.zeros((2, 2))), ARGON_MODEL)
        with self.assertRaisesRegex(ValueError, "at rest"):
            a.rescale_velocities(250.0)

    def test_raises_when_the_temperature_is_not_finite(self):
        for bad in (np.inf, np.nan):
            a = MDSimulation(two_argon([[3.0, 0.0], [-3.0, 0.0]]), ARGON_MODEL)
            a.configuration.velocities[0, 0] = bad
            with self.assertRaisesRegex(ValueError, "diverged"):
                a.rescale_velocities(250.0)

    def test_zero_stops_the_atoms(self):
        a = MDSimulation(two_argon([[3.0, 0.0], [-3.0, 0.0]]), ARGON_MODEL)
        a.rescale_velocities(0.0)
        assert_equal(a.configuration.velocities, 0.0)

    def test_zero_leaves_atoms_at_rest(self):
        a = MDSimulation(two_argon(np.zeros((2, 2))), ARGON_MODEL)
        a.rescale_velocities(0.0)
        assert_equal(a.configuration.velocities, 0.0)

    def test_raises_for_a_negative_or_non_finite_temperature(self):
        a = MDSimulation(two_argon([[3.0, 0.0], [-3.0, 0.0]]), ARGON_MODEL)
        for bad in (-5.0, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "temperature must be non-negative"):
                a.rescale_velocities(bad)


class TestSample(unittest.TestCase):
    def test_records_the_step_and_the_thermodynamics(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=2, temperature=300, box=8)
        for _ in range(3):
            a.step()
        a.sample()
        assert_equal(a.samples.step, [3])
        for name in ("temperature", "pressure", "potential_energy", "kinetic_energy"):
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

    def test_appends_the_configuration_to_the_trajectory(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        self.assertEqual(len(a.trajectory), 0)
        a.sample()
        a.step()
        a.sample()
        self.assertEqual(len(a.trajectory), 2)
        self.assertIsNot(a.trajectory[1], a.configuration)
        assert_equal(a.trajectory[1].positions, a.configuration.positions)
        a.step()
        self.assertFalse(np.array_equal(a.trajectory[1].positions, a.configuration.positions))

    def test_frames_carry_the_time_of_each_sample(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        assert_equal(a.trajectory.times, [])
        for _ in range(3):
            a.step()
            a.step()
            a.sample()
        assert_allclose(a.trajectory.times, a.samples.step * a.timestep)
        assert_allclose(a.trajectory.times, np.array([2, 4, 6]) * a.timestep)
        assert_equal(a.restart().trajectory.times, [])


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
        assert_equal(production.forces, a.forces)
        # The restarted simulation follows the same trajectory as the source.
        a.step()
        production.step()
        assert_equal(production.configuration.positions, a.configuration.positions)

    def test_leaves_the_source_alone(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=300, box=12)
        self.run_and_sample(a, 3)
        source = a.configuration.positions.copy()
        production = a.restart()
        production.step()
        production.configuration.velocities *= 2
        assert_equal(a.configuration.positions, source)
        assert_equal(a.trajectory[-1].positions, source)
        self.assertEqual(a.steps, 3)
        self.assertEqual(a.samples.step.size, 3)

    def test_starts_an_empty_trajectory(self):
        a = MDSimulation.initialise(ARGON_MODEL, number_of_atoms=4, temperature=100, box=20)
        a.sample()
        production = a.restart()
        self.assertEqual(len(production.trajectory), 0)
        self.assertEqual(len(a.trajectory), 1)


class TestDriftRemoval(unittest.TestCase):
    def drifting(self):
        # A mixture: with one species the plain mean of the velocities
        # equals the mass-weighted mean, so removing the plain mean would
        # pass every test here.
        configuration = MDSimulation.initialise(
            MIXTURE_MODEL, number_of_atoms=8, temperature=100, box=20, seed=1
        ).configuration
        configuration.velocities = configuration.velocities + [0.1, -0.04]
        return configuration

    def test_the_constructor_sets_the_centre_of_mass_at_rest(self):
        moving = self.drifting()
        self.assertGreater(drift_speed(moving), 0.01)
        simulation = MDSimulation(moving, MIXTURE_MODEL)
        self.assertLess(drift_speed(simulation.configuration), 1e-11)

    def test_leaves_the_positions_and_every_relative_velocity_alone(self):
        moving = self.drifting()
        rested = MDSimulation(moving, MIXTURE_MODEL).configuration
        assert_allclose(rested.positions, moving.positions)
        assert_allclose(
            rested.velocities - rested.velocities[0], moving.velocities - moving.velocities[0]
        )

    def test_the_kinetic_energy_falls_by_the_drift_energy(self):
        moving = self.drifting()
        speed = drift_speed(moving)
        rested = MDSimulation(moving, MIXTURE_MODEL).configuration
        assert_allclose(
            moving.kinetic_energy() - rested.kinetic_energy(),
            0.5 * moving.masses.sum() * speed**2 / KJ_PER_MOL,
        )

    def test_dropping_an_atom_leaves_the_simulation_at_rest(self):
        # Removing an atom takes its momentum with it, so the rest drift.
        started = MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=16, temperature=100, box=25, seed=1
        )
        vacancy = started.configuration.without(0)
        self.assertGreater(drift_speed(vacancy), 0.005)
        simulation = MDSimulation(vacancy, ARGON_MODEL)
        self.assertLess(drift_speed(simulation.configuration), 1e-11)

    def test_initialise_still_reports_its_target_temperature(self):
        simulation = MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=16, temperature=137.0, box=25, seed=1
        )
        assert_allclose(simulation.configuration.temperature(), 137.0)
