import itertools
import re
import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import mc, md, util
from pylj.potentials import Buckingham, LennardJones, PairPotential, SquareWell
from pylj.tests.argon import ARGON, ARGON_MODEL, LARGER, LJ_ARGON, MIXTURE_MODEL


def minimum_image_distances(system):
    """Return the minimum-image separation, in metres, for every pair of
    particles in ``system``."""
    box_length = system.box_length
    x = system.particles["xposition"]
    y = system.particles["yposition"]
    distances = []
    for i in range(system.number_of_particles):
        for j in range(i + 1, system.number_of_particles):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dx -= box_length * np.round(dx / box_length)
            dy -= box_length * np.round(dy / box_length)
            distances.append(np.sqrt(dx**2 + dy**2))
    return distances


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

    def test_system_random(self):
        a = util.System(2, 300, 8, init_conf="random", simulation="md", **ARGON_MODEL)
        assert_equal(a.number_of_particles, 2)
        assert_equal(a.temperature, 300)
        assert_almost_equal(a.box_length * 1e10, 8)
        assert_almost_equal(a.timestep_length, 1e-14)
        self.assertTrue(0 <= a.particles["xposition"][0] * 1e10 <= 8)
        self.assertTrue(0 <= a.particles["yposition"][0] * 1e10 <= 8)
        self.assertTrue(0 <= a.particles["xposition"][1] * 1e10 <= 8)
        self.assertTrue(0 <= a.particles["yposition"][1] * 1e10 <= 8)
        assert_almost_equal(a.cut_off * 1e10, 4.0)
        assert_equal(a.distances.size, 1)
        assert_equal(a.forces.size, 1)
        assert_equal(a.energies.size, 1)

    def test_system_random_no_overlap(self):
        a = util.System(30, 100, 40, init_conf="random", simulation="md", seed=0, **ARGON_MODEL)
        for distance in minimum_image_distances(a):
            self.assertTrue(distance >= a.cores[0])

    def test_system_random_too_dense_raises(self):
        with self.assertRaisesRegex(ValueError, "after 1000 attempts"):
            util.System(200, 100, 20, init_conf="random", simulation="md", **ARGON_MODEL)

    def test_system_square_overlap_message_for_one_particle(self):
        with self.assertRaisesRegex(ValueError, "1 particle fits"):
            util.System(5, 100, 4, simulation="md", **ARGON_MODEL)

    def test_system_square_overlap_raises(self):
        # 50 argon particles in a 20 Angstrom box: at most 25 fit on a
        # square lattice without overlap.
        with self.assertRaisesRegex(ValueError, "at most 25 particles"):
            util.System(50, 100, 20, simulation="md", **ARGON_MODEL)

    def test_system_square_between_core_and_diameter_is_accepted(self):
        # 101 argon particles in a 40 Angstrom box space 3.64 Angstrom apart,
        # above the 3.37 Angstrom repulsive core.
        a = util.System(101, 100, 40, simulation="md", **ARGON_MODEL)
        assert_equal(a.number_of_particles, 101)

    def test_system_random_threshold_is_the_core(self):
        # With seed 4 particles land closer together than 8 Angstrom but
        # still respect the 3.37 Angstrom repulsive core.
        a = util.System(10, 100, 30, init_conf="random", simulation="md", seed=4, **ARGON_MODEL)
        distances = minimum_image_distances(a)
        self.assertTrue(all(distance >= a.cores[0] for distance in distances))
        self.assertTrue(any(distance < 8e-10 for distance in distances))

    def test_system_random_two_types_uses_the_mean_of_the_pair_cores(self):
        a = util.System(12, 100, 60, init_conf="random", simulation="md", seed=5, **MIXTURE_MODEL)
        types = a.particles["types"]
        distances = minimum_image_distances(a)
        pairs = itertools.combinations(range(a.number_of_particles), 2)
        for distance, (i, j) in zip(distances, pairs, strict=True):
            type_i, type_j = int(types[i]), int(types[j])
            min_separation = (a.cores[type_i] + a.cores[type_j]) / 2
            self.assertTrue(distance >= min_separation)

    def test_system_argon_core_equals_sigma(self):
        a = util.System(2, 300, 8, simulation="md", **ARGON_MODEL)
        assert_almost_equal(a.cores[0] * 1e10, LJ_ARGON.sigma * 1e10, decimal=3)

    def test_system_square_well_core_is_at_least_sigma(self):
        sigma = 1.5e-10
        well = SquareWell(epsilon=1.0, sigma=sigma, lambda_=2.0)
        b = util.System(
            2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): well}, simulation="md"
        )
        self.assertTrue(sigma <= b.cores[0] <= sigma * 1.002)

    def test_system_potential_never_positive_raises(self):
        class NeverPositive(PairPotential):
            def energies(self, dr):
                return -np.ones_like(np.asarray(dr, dtype=float))

            def forces(self, dr):
                return np.zeros_like(np.asarray(dr, dtype=float))

        with self.assertRaisesRegex(ValueError, "repulsive core"):
            util.System(
                2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): NeverPositive()},
                simulation="md",
            )

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
            "random" in str(context.exception)
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

    def test_system_square_too_small_box_suggests_a_box_that_fits(self):
        for number_of_particles in (2, 7, 50):
            with self.assertRaises(ValueError) as context:
                util.System(number_of_particles, 100, 4, simulation="md", **ARGON_MODEL)
            match = re.search(r"at least ([0-9.]+) Angstrom fits", str(context.exception))
            self.assertIsNotNone(match)
            suggested_box = float(match.group(1))
            a = util.System(
                number_of_particles, 100, suggested_box, simulation="md", **ARGON_MODEL
            )
            self.assertEqual(a.number_of_particles, number_of_particles)

    def test_system_potential_energies_returning_a_scalar_is_rejected(self):
        class ScalarEnergy(PairPotential):
            def energies(self, dr):
                return 1.0

            def forces(self, dr):
                return 0.0

        with self.assertRaisesRegex(ValueError, "one value per separation"):
            util.System(
                2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): ScalarEnergy()},
                simulation="md",
            )

    def test_system_potential_energies_returning_the_wrong_shape_is_rejected(self):
        class ShortEnergy(PairPotential):
            def energies(self, dr):
                return np.ones(3)

            def forces(self, dr):
                return np.zeros(3)

        with self.assertRaisesRegex(ValueError, "one value per separation"):
            util.System(
                2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): ShortEnergy()},
                simulation="md",
            )

    def test_system_potential_positive_at_50_angstrom_names_units(self):
        # Sigma given in Angstrom rather than metres: the pair energy never
        # falls to zero on the 0.1 to 50 Angstrom grid.
        in_angstrom = LennardJones(epsilon=1.58e-21, sigma=3.4)
        with self.assertRaisesRegex(ValueError, "still positive at 50 Angstrom"):
            util.System(
                2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): in_angstrom},
                simulation="md",
            )

    def test_system_random_buckingham_core_energy_is_near_zero(self):
        potential = Buckingham(a=1.69e-15, b=3.66e10, c=1.01e-77)
        a = util.System(
            10, 100, 40, init_conf="random", species=[ARGON],
            pair_potentials={(ARGON, ARGON): potential}, simulation="md", seed=0,
        )
        self.assertTrue(a.cores[0] > 3e-10)
        r = np.logspace(-11, np.log10(5e-9), 4000)
        well_depth = potential.energies(r).min()
        core_energy = potential.energies(np.array([a.cores[0]]))[0]
        self.assertTrue(abs(core_energy) < 1e-3 * abs(well_depth))

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

    def test_system_seed_reproduces_random_placement(self):
        def build(seed):
            return util.System(
                10, 100, 40, init_conf="random", simulation="md", seed=seed, **ARGON_MODEL
            )

        first = build(3)
        second = build(3)
        other = build(4)
        assert_equal(first.particles["xposition"], second.particles["xposition"])
        assert_equal(first.particles["yposition"], second.particles["yposition"])
        self.assertFalse(
            np.array_equal(first.particles["xposition"], other.particles["xposition"])
        )

    def test_system_seed_is_not_taken_from_the_global_state(self):
        # Two systems built with the same seed agree even when the global
        # NumPy state differs between them.
        def build():
            return util.System(
                10, 100, 40, init_conf="random", simulation="md", seed=3, **ARGON_MODEL
            )

        first = build()
        state = np.random.get_state()
        try:
            np.random.seed(99)
            second = build()
        finally:
            np.random.set_state(state)
        assert_equal(first.particles["xposition"], second.particles["xposition"])

    def test_restart_copies_the_generator_state(self):
        system = md.initialise(4, 300, 12, "square", seed=1, **ARGON_MODEL)
        production = system.restart()
        self.assertIsNot(production.rng, system.rng)
        # Equal state: the next draw agrees. Independent: drawing from one
        # leaves the other where it was.
        self.assertEqual(production.rng.random(), system.rng.random())
        system.rng.random()
        self.assertNotEqual(production.rng.random(), system.rng.random())
