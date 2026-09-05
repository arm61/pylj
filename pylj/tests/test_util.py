import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import mc, md, pairwise, util
from pylj.constants import BOLTZMANN
from pylj.potentials import LennardJones, SquareWell
from pylj.tests.argon import ARGON, ARGON_MODEL, LARGER, LJ_ARGON, MIXTURE_MODEL


class TestUtil(unittest.TestCase):
    def test_system_square(self):
        a = util.System(2, 300, 8, simulation="md", **ARGON_MODEL)
        assert_equal(a.number_of_particles, 2)
        assert_equal(a.temperature, 300)
        assert_almost_equal(a.box_length * 1e10, 8)
        assert_almost_equal(a.timestep_length, 1e-14)
        assert_almost_equal(a.particles["xposition"] * 1e10, [2, 2])
        assert_almost_equal(a.particles["yposition"] * 1e10, [2, 6])
        assert_almost_equal(a.initial_particles["xposition"] * 1e10, [2, 2])
        assert_almost_equal(a.initial_particles["yposition"] * 1e10, [2, 6])
        assert_almost_equal(a.cut_off * 1e10, 4.0)
        assert_equal(a.distances.size, 1)
        assert_equal(a.forces.size, 1)
        assert_equal(a.energies.size, 1)

    def test_system_metropolis(self):
        a = util.System(2, 300, 8, init_conf="metropolis", simulation="md", **ARGON_MODEL)
        assert_equal(a.number_of_particles, 2)
        for axis in ("xposition", "yposition"):
            self.assertTrue(np.all((0 <= a.particles[axis]) & (a.particles[axis] < a.box_length)))
        assert_almost_equal(a.cut_off * 1e10, 4.0)
        assert_equal(a.distances.size, 1)

    def test_system_metropolis_places_a_hard_core_outside_its_diameter(self):
        # A trial inside the square well's core costs infinite energy and is
        # always rejected, so no pair is closer than sigma.
        well = SquareWell(epsilon=1.5e-21, sigma=3e-10, lambda_=1.5)
        a = util.System(
            50, 300, 30, init_conf="metropolis", simulation="mc", seed=1,
            species=[ARGON], pair_potentials={(ARGON, ARGON): well},
        )
        distances, _ = pairwise.compute_energy(
            a.particles, a.box_length, a.cut_off, a.pair_potentials, a.species
        )
        self.assertGreaterEqual(distances.min(), well.sigma)

    def test_system_metropolis_places_a_mixture_with_each_pairs_own_potential(self):
        # Under its own potential no placed pair can sit at a repulsive energy
        # the acceptance could not have passed: a single pair above 20 k_B T
        # cannot be offset by a few attractive neighbours. Evaluating every
        # pair with its own potential catches a placement that used the
        # wrong one, whatever the seed.
        for seed in range(5):
            a = util.System(
                20, 100, 60, init_conf="metropolis", simulation="md", seed=seed, **MIXTURE_MODEL
            )
            _, energies = pairwise.compute_energy(
                a.particles, a.box_length, a.cut_off, a.pair_potentials, a.species
            )
            self.assertLess(energies.max(), 20 * BOLTZMANN * 100)

    def test_system_refuses_a_potential_still_repulsive_at_the_cut_off(self):
        # Sigma given in Angstrom: the pair energy is astronomically positive
        # at the cut-off, where a sensible potential has died away.
        in_angstrom = LennardJones(epsilon=1.577e-21, sigma=3.372)
        with self.assertRaisesRegex(ValueError, "at the cut-off"):
            util.System(
                2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): in_angstrom},
                simulation="md",
            )

    def test_system_refuses_a_cross_potential_still_repulsive_at_the_cut_off(self):
        mistyped = dict(MIXTURE_MODEL["pair_potentials"])
        mistyped[(ARGON, LARGER)] = LennardJones(epsilon=1.577e-21, sigma=4.186)
        with self.assertRaisesRegex(ValueError, "between argon and larger"):
            util.System(
                4, 100, 60, species=[ARGON, LARGER], pair_potentials=mistyped, simulation="md"
            )

    def test_system_metropolis_configuration_has_no_overlap(self):
        # At 100 K a pair inside 0.8 sigma costs over 40 well depths and is
        # never accepted, so the closest pair sits outside the core.
        a = util.System(30, 100, 40, init_conf="metropolis", simulation="md", seed=0, **ARGON_MODEL)
        distances, _ = pairwise.compute_energy(
            a.particles, a.box_length, a.cut_off, a.pair_potentials, a.species
        )
        self.assertGreater(distances.min(), 0.8 * LJ_ARGON.sigma)

    def test_system_metropolis_too_dense_raises(self):
        with self.assertRaisesRegex(ValueError, f"after {util.PLACEMENT_ATTEMPTS} attempts"):
            util.System(200, 100, 20, init_conf="metropolis", simulation="md", **ARGON_MODEL)

    def test_system_metropolis_seed_reproduces_placement(self):
        def build(seed):
            return util.System(
                10, 100, 40, init_conf="metropolis", simulation="md", seed=seed, **ARGON_MODEL
            )

        first = build(3)
        second = build(3)
        other = build(4)
        assert_equal(first.particles["xposition"], second.particles["xposition"])
        assert_equal(first.particles["yposition"], second.particles["yposition"])
        self.assertFalse(
            np.array_equal(first.particles["xposition"], other.particles["xposition"])
        )

    def test_system_metropolis_seed_is_not_taken_from_the_global_state(self):
        def build():
            return util.System(
                10, 100, 40, init_conf="metropolis", simulation="md", seed=3, **ARGON_MODEL
            )

        first = build()
        state = np.random.get_state()
        try:
            np.random.seed(99)
            second = build()
        finally:
            np.random.set_state(state)
        assert_equal(first.particles["xposition"], second.particles["xposition"])

    def test_system_metropolis_defaults_placement_temperature_to_the_run_temperature(self):
        default = util.System(2, 300, 8, simulation="md", **ARGON_MODEL)
        explicit = util.System(
            2, 300, 8, simulation="md", placement_temperature=1000, **ARGON_MODEL
        )
        self.assertEqual(default.placement_temperature, 300)
        self.assertEqual(explicit.placement_temperature, 1000)

    def test_system_rejects_a_bad_placement_temperature(self):
        for bad in (0, -1, np.inf):
            with self.assertRaisesRegex(ValueError, "placement_temperature must be positive"):
                util.System(2, 300, 8, simulation="md", placement_temperature=bad, **ARGON_MODEL)

    def test_system_metropolis_placement_temperature_governs_success(self):
        # 50 argon particles in a 25 Angstrom box: as near-hard discs of
        # diameter sigma (placement at 1 K) they exceed the packing that
        # sequential insertion reaches, but at 1e5 K overlaps are tolerated.
        with self.assertRaises(ValueError):
            util.System(
                50, 100, 25, init_conf="metropolis", simulation="md", seed=0,
                placement_temperature=1.0, **ARGON_MODEL,
            )
        hot = util.System(
            50, 100, 25, init_conf="metropolis", simulation="md", seed=0,
            placement_temperature=1e5, **ARGON_MODEL,
        )
        assert_equal(hot.number_of_particles, 50)

    def test_system_too_big(self):
        with self.assertRaises(AttributeError) as context:
            util.System(2, 300, 1000, simulation="md", **ARGON_MODEL)
        self.assertTrue(
            "With a box length of 1000 the particles are probably "
            "too small to be seen in the viewer. Try something "
            "(much) less than 600." in str(context.exception)
        )

    def test_system_too_small(self):
        with self.assertRaises(AttributeError) as context:
            util.System(2, 300, 2, simulation="md", **ARGON_MODEL)
        self.assertTrue(
            "With a box length of 2 the cell is too small to "
            "really hold more than one particle." in str(context.exception)
        )

    def test_system_init_conf(self):
        with self.assertRaises(NotImplementedError) as context:
            util.System(2, 300, 100, init_conf="horseradish", simulation="md", **ARGON_MODEL)
        self.assertTrue(
            "The initial configuration type horseradish is not "
            "recognised. Available options are: square or "
            "metropolis" in str(context.exception)
        )

    def test_system_records_simulation_kind(self):
        a = util.System(2, 300, 8, simulation="mc", **ARGON_MODEL)
        assert_equal(a.simulation, "mc")

    def test_system_rejects_unknown_simulation_kind(self):
        with self.assertRaises(ValueError):
            util.System(2, 300, 8, simulation="dft", **ARGON_MODEL)

    def test_system_assigns_species_in_turn_and_masses_from_them(self):
        a = util.System(5, 100, 30, simulation="md", **MIXTURE_MODEL)
        assert_equal(a.particles["types"], [0, 1, 0, 1, 0])
        assert_almost_equal(a.masses, [39.948, 80.0, 39.948, 80.0, 39.948])

    def test_system_rejects_a_missing_pair_potential(self):
        incomplete = {(ARGON, ARGON): LJ_ARGON, (LARGER, LARGER): LJ_ARGON}
        with self.assertRaisesRegex(ValueError, "no entry for the pair .*larger"):
            util.System(
                2, 300, 12, species=[ARGON, LARGER], pair_potentials=incomplete, simulation="md"
            )

    def test_system_accepts_a_pair_potential_keyed_in_either_order(self):
        reversed_cross = dict(MIXTURE_MODEL["pair_potentials"])
        reversed_cross[(LARGER, ARGON)] = reversed_cross.pop((ARGON, LARGER))
        a = util.System(
            2, 300, 12, species=[ARGON, LARGER], pair_potentials=reversed_cross, simulation="md"
        )
        assert_equal(a.particles["types"], [0, 1])

    def test_system_rejects_a_potential_class_in_place_of_an_instance(self):
        with self.assertRaisesRegex(TypeError, "PairPotential instance"):
            util.System(
                2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): LennardJones},
                simulation="md",
            )

    def test_system_rejects_a_cross_pair_given_in_both_orders(self):
        both_orders = dict(MIXTURE_MODEL["pair_potentials"])
        both_orders[(LARGER, ARGON)] = LJ_ARGON
        with self.assertRaisesRegex(ValueError, "in both orders"):
            util.System(
                2, 300, 12, species=[ARGON, LARGER], pair_potentials=both_orders, simulation="md"
            )

    def test_system_rejects_no_species(self):
        with self.assertRaisesRegex(ValueError, "at least one Species"):
            util.System(2, 300, 8, species=[], pair_potentials={}, simulation="md")

    def test_system_rejects_a_non_positive_or_infinite_temperature(self):
        for temperature in (0, -10, np.inf):
            with self.assertRaisesRegex(ValueError, "temperature must be positive"):
                util.System(2, temperature, 8, simulation="md", **ARGON_MODEL)

    def test_restart_shares_the_model(self):
        system = md.initialise(4, 300, 12, "square", **ARGON_MODEL)
        production = system.restart()
        self.assertIs(production.species, system.species)
        self.assertIs(production.pair_potentials, system.pair_potentials)
        assert_equal(production.masses, system.masses)

    def test_restart_starts_a_fresh_record_from_the_current_state(self):
        system = md.initialise(4, 300, 12, "square", **ARGON_MODEL)
        for _ in range(5):
            system.integrate(md.velocity_verlet)
            system.step += 1
            system.time += system.timestep_length
            system.md_sample()
        production = system.restart()
        self.assertIsNot(production, system)
        self.assertEqual(production.simulation, "md")
        self.assertEqual(production.step, 0)
        self.assertEqual(production.time, 0.0)
        for name in (
            "temperature_sample",
            "pressure_sample",
            "force_sample",
            "msd_sample",
            "energy_sample",
            "step_sample",
        ):
            self.assertEqual(getattr(production, name).size, 0)
        self.assertEqual(
            md.calculate_msd(production.particles, production.initial_particles), 0.0
        )
        # Sampling straight after the restart uses the copied pair arrays.
        system.md_sample()
        production.md_sample()
        self.assertEqual(production.pressure_sample[0], system.pressure_sample[-1])
        self.assertEqual(production.energy_sample[0], system.energy_sample[-1])
        # The restarted system follows the same trajectory as the source
        # while its displacement is measured from the restart.
        system.integrate(md.velocity_verlet)
        production.integrate(md.velocity_verlet)
        assert_equal(production.particles["xposition"], system.particles["xposition"])
        assert_equal(production.particles["yposition"], system.particles["yposition"])
        production.md_sample()
        self.assertGreater(production.msd_sample[-1], 0.0)

    def test_restart_leaves_the_source_alone(self):
        system = md.initialise(4, 300, 12, "square", **ARGON_MODEL)
        for _ in range(3):
            system.integrate(md.velocity_verlet)
            system.step += 1
            system.md_sample()
        positions = system.particles["xposition"].copy()
        origin = system.initial_particles["xunwrapped"].copy()
        production = system.restart()
        production.particles["xposition"] = 0.0
        production.initial_particles["xunwrapped"] = 0.0
        self.assertEqual(system.step, 3)
        self.assertEqual(system.msd_sample.size, 3)
        assert_equal(system.particles["xposition"], positions)
        assert_equal(system.initial_particles["xunwrapped"], origin)

    def test_restart_carries_the_energy_for_monte_carlo(self):
        system = mc.initialise(4, 300, 12, "square", **ARGON_MODEL)
        system.mc_sample()
        production = system.restart()
        self.assertEqual(production.simulation, "mc")
        self.assertEqual(production.energy, system.energy)
        self.assertEqual(production.energy_sample.size, 0)
        self.assertEqual(system.energy_sample.size, 1)

    def test_restart_copies_the_generator_state(self):
        system = md.initialise(4, 300, 12, "square", seed=1, **ARGON_MODEL)
        production = system.restart()
        self.assertIsNot(production.rng, system.rng)
        # Equal state: the next draw agrees. Independent: drawing from one
        # leaves the other where it was.
        self.assertEqual(production.rng.random(), system.rng.random())
        system.rng.random()
        self.assertNotEqual(production.rng.random(), system.rng.random())
