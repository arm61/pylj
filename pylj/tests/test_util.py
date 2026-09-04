import itertools
import re
import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import forcefields as ff
from pylj import util

# Sigma and epsilon in metres and joules for two Lennard-Jones species: argon,
# and a larger particle with a 5 Angstrom core and the same well depth.
ARGON = [3.37e-10, 1.58e-21]
LARGER = [5.0e-10, 1.58e-21]


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
        a = util.System(
            2,
            300,
            8,
            mass=39.948,
            constants=[[1.363e-134, 9.273e-78]],
            forcefield=ff.lennard_jones,
            simulation="md",
        )
        assert_equal(a.number_of_particles, 2)
        assert_equal(a.init_temp, 300)
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
        a = util.System(
            2,
            300,
            8,
            init_conf="random",
            mass=39.948,
            constants=[[1.363e-134, 9.273e-78]],
            forcefield=ff.lennard_jones,
            simulation="md",
        )
        assert_equal(a.number_of_particles, 2)
        assert_equal(a.init_temp, 300)
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
        state = np.random.get_state()
        np.random.seed(0)
        try:
            a = util.System(
                30,
                100,
                40,
                init_conf="random",
                mass=39.948,
                constants=[[1.363e-134, 9.273e-78]],
                forcefield=ff.lennard_jones,
                simulation="md",
            )
        finally:
            np.random.set_state(state)
        for distance in minimum_image_distances(a):
            self.assertTrue(distance >= a.cores[0])

    def test_system_random_too_dense_raises(self):
        with self.assertRaisesRegex(ValueError, "after 1000 attempts"):
            util.System(
                200,
                100,
                20,
                init_conf="random",
                mass=39.948,
                constants=[[1.363e-134, 9.273e-78]],
                forcefield=ff.lennard_jones,
                simulation="md",
            )

    def test_system_square_overlap_message_for_one_particle(self):
        with self.assertRaisesRegex(ValueError, "1 particle fits"):
            util.System(
                5, 100, 4, [[1.363e-134, 9.273e-78]], ff.lennard_jones, 39.948,
                simulation="md",
            )

    def test_system_square_overlap_raises(self):
        # 50 argon particles in a 20 Angstrom box: at most 25 fit on a
        # square lattice without overlap.
        with self.assertRaisesRegex(ValueError, "at most 25 particles"):
            util.System(
                50,
                100,
                20,
                mass=39.948,
                constants=[[1.363e-134, 9.273e-78]],
                forcefield=ff.lennard_jones,
                simulation="md",
            )

    def test_system_square_between_core_and_diameter_is_accepted(self):
        # 101 argon particles in a 40 Angstrom box space 3.64 Angstrom apart:
        # above the 3.37 Angstrom repulsive core, but below the 3.78 Angstrom
        # drawn diameter, which used to be the (overly strict) threshold.
        a = util.System(
            101,
            100,
            40,
            mass=39.948,
            constants=[[1.363e-134, 9.273e-78]],
            forcefield=ff.lennard_jones,
            simulation="md",
        )
        assert_equal(a.number_of_particles, 101)

    def test_system_random_threshold_is_the_core_not_the_drawn_diameter(self):
        # A diameter=8.0 override only changes how particles are drawn, not
        # how far apart random() places them: with seed 4, particles land
        # closer together than 8 Angstrom but still respect the 3.37
        # Angstrom repulsive core.
        state = np.random.get_state()
        np.random.seed(4)
        try:
            a = util.System(
                10,
                100,
                30,
                init_conf="random",
                mass=39.948,
                constants=[[1.363e-134, 9.273e-78]],
                forcefield=ff.lennard_jones,
                simulation="md",
                diameter=8.0,
            )
        finally:
            np.random.set_state(state)
        distances = minimum_image_distances(a)
        self.assertTrue(all(distance >= a.cores[0] for distance in distances))
        self.assertTrue(any(distance < 8e-10 for distance in distances))

    def test_system_random_two_types_uses_the_mean_of_the_pair_cores(self):
        constants = [ARGON, LARGER]
        state = np.random.get_state()
        np.random.seed(5)
        try:
            a = util.System(
                12,
                100,
                60,
                init_conf="random",
                mass=39.948,
                constants=constants,
                forcefield=ff.lennard_jones_sigma_epsilon,
                simulation="md",
            )
        finally:
            np.random.set_state(state)
        types = a.particles["types"]
        distances = minimum_image_distances(a)
        pairs = itertools.combinations(range(a.number_of_particles), 2)
        for distance, (i, j) in zip(distances, pairs, strict=True):
            type_i, type_j = int(types[i]), int(types[j])
            min_separation = (a.cores[type_i] + a.cores[type_j]) / 2
            self.assertTrue(distance >= min_separation)

    def test_system_argon_core_equals_sigma(self):
        a = util.System(
            2, 300, 8, [[1.363e-134, 9.273e-78]], ff.lennard_jones, 39.948, simulation="md"
        )
        expected_sigma = (1.363e-134 / 9.273e-78) ** (1 / 6)
        assert_almost_equal(a.cores[0] * 1e10, expected_sigma * 1e10, decimal=3)

    def test_system_square_well_core_is_at_least_sigma(self):
        sigma = 1.5e-10
        b = util.System(
            2, 300, 8, [[1.0, sigma, 2.0]], ff.square_well, 39.948, simulation="md"
        )
        self.assertTrue(sigma <= b.cores[0] <= sigma * 1.002)

    def test_system_forcefield_never_positive_raises(self):
        class NeverPositive:
            def __init__(self, constants):
                self.constants = constants

            @property
            def diameter(self):
                return 1e-10

            def energy(self, dr):
                return -np.ones_like(np.asarray(dr, dtype=float))

        with self.assertRaisesRegex(ValueError, "repulsive core"):
            util.System(2, 300, 8, [[1.0]], NeverPositive, 39.948, simulation="md")

    def test_system_too_big(self):
        with self.assertRaises(AttributeError) as context:
            util.System(
                2,
                300,
                1000,
                mass=39.948,
                constants=[[1.363e-134, 9.273e-78]],
                forcefield=ff.lennard_jones,
                simulation="md",
            )
        self.assertTrue(
            "With a box length of 1000 the particles are probably "
            "too small to be seen in the viewer. Try something "
            "(much) less than 600." in str(context.exception)
        )

    def test_system_too_small(self):
        with self.assertRaises(AttributeError) as context:
            util.System(
                2,
                300,
                2,
                mass=39.948,
                constants=[[1.363e-134, 9.273e-78]],
                forcefield=ff.lennard_jones,
                simulation="md",
            )
        self.assertTrue(
            "With a box length of 2 the cell is too small to "
            "really hold more than one particle." in str(context.exception)
        )

    def test_system_init_conf(self):
        with self.assertRaises(NotImplementedError) as context:
            util.System(
                2,
                300,
                100,
                init_conf="horseradish",
                mass=39.948,
                constants=[[1.363e-134, 9.273e-78]],
                forcefield=ff.lennard_jones,
                simulation="md",
            )
        self.assertTrue(
            "The initial configuration type horseradish is not "
            "recognised. Available options are: square or "
            "random" in str(context.exception)
        )

    def test_system_records_simulation_kind(self):
        a = util.System(
            2, 300, 8, [[1.363e-134, 9.273e-78]], ff.lennard_jones, 39.948, simulation="mc"
        )
        assert_equal(a.simulation, "mc")

    def test_system_rejects_unknown_simulation_kind(self):
        with self.assertRaises(ValueError):
            util.System(
                2, 300, 8, [[1.363e-134, 9.273e-78]], ff.lennard_jones, 39.948, simulation="dft"
            )

    def test_system_diameters_default_to_forcefield(self):
        a = util.System(
            2, 300, 8, [[1.363e-134, 9.273e-78]], ff.lennard_jones, 39.948, simulation="md"
        )
        assert_almost_equal(a.diameters[0] * 1e10, 3.78, decimal=2)

    def test_system_single_diameter_applies_to_every_type(self):
        constants = [ARGON, LARGER]
        a = util.System(
            2, 300, 12, constants, ff.lennard_jones_sigma_epsilon, 39.948,
            simulation="md", diameter=3.0,
        )
        assert_almost_equal(a.diameters, [3e-10, 3e-10])

    def test_system_diameter_list_is_per_type(self):
        constants = [ARGON, LARGER]
        a = util.System(
            2, 300, 12, constants, ff.lennard_jones_sigma_epsilon, 39.948,
            simulation="md", diameter=[3.0, 5.0],
        )
        assert_almost_equal(a.diameters, [3e-10, 5e-10])

    def test_system_diameter_array_is_per_type(self):
        constants = [ARGON, LARGER]
        a = util.System(
            2, 300, 12, constants, ff.lennard_jones_sigma_epsilon, 39.948, simulation="md",
            diameter=np.array([3.0, 5.0]),
        )
        assert_almost_equal(a.diameters, [3e-10, 5e-10])

    def test_system_diameter_list_must_match_types(self):
        constants = [ARGON, LARGER]
        with self.assertRaises(ValueError):
            util.System(
                2, 300, 8, constants, ff.lennard_jones_sigma_epsilon, 39.948,
                simulation="md", diameter=[3.0],
            )

    def test_system_diameter_must_be_positive(self):
        constants = [[1.363e-134, 9.273e-78]]
        with self.assertRaisesRegex(ValueError, "positive"):
            util.System(
                2, 300, 8, constants, ff.lennard_jones, 39.948, simulation="md", diameter=0.0
            )

    def test_system_diameter_in_metres_is_rejected(self):
        constants = [[1.363e-134, 9.273e-78]]
        with self.assertRaisesRegex(ValueError, "Angstrom"):
            util.System(
                2, 300, 8, constants, ff.lennard_jones, 39.948, simulation="md", diameter=3.4e-10
            )

    def test_system_forcefield_whose_diameter_raises_keeps_the_cause(self):
        class BrokenDiameter:
            def __init__(self, constants):
                self.constants = constants

            @property
            def diameter(self):
                raise AttributeError("oops")

        constants = [[1.363e-134, 9.273e-78]]
        with self.assertRaises(ValueError) as caught:
            util.System(2, 300, 8, constants, BrokenDiameter, 39.948, simulation="md")
        self.assertIn("oops", str(caught.exception.__cause__))

    def test_system_forcefield_without_a_diameter_is_rejected(self):
        class NoDiameter:
            def __init__(self, constants):
                self.constants = constants

        constants = [[1.363e-134, 9.273e-78]]
        with self.assertRaisesRegex(ValueError, "diameter"):
            util.System(2, 300, 8, constants, NoDiameter, 39.948, simulation="md")

    def test_system_forcefield_with_a_zero_diameter_is_rejected(self):
        class ZeroDiameter:
            def __init__(self, constants):
                self.constants = constants

            @property
            def diameter(self):
                return 0.0

            def energy(self, dr):
                return np.ones_like(np.asarray(dr, dtype=float))

        with self.assertRaisesRegex(ValueError, "ZeroDiameter"):
            util.System(2, 300, 8, [[1.0]], ZeroDiameter, 39.948, simulation="md")

    def test_system_diameter_must_be_finite(self):
        constants = [[1.363e-134, 9.273e-78]]
        with self.assertRaisesRegex(ValueError, "nan"):
            util.System(
                2, 300, 8, constants, ff.lennard_jones, 39.948, simulation="md",
                diameter=float("nan"),
            )

    def test_system_square_too_small_box_suggests_a_box_that_fits(self):
        for number_of_particles in (2, 7, 50):
            with self.assertRaises(ValueError) as context:
                util.System(
                    number_of_particles,
                    100,
                    4,
                    mass=39.948,
                    constants=[[1.363e-134, 9.273e-78]],
                    forcefield=ff.lennard_jones,
                    simulation="md",
                )
            match = re.search(r"at least ([0-9.]+) Angstrom fits", str(context.exception))
            self.assertIsNotNone(match)
            suggested_box = float(match.group(1))
            a = util.System(
                number_of_particles,
                100,
                suggested_box,
                mass=39.948,
                constants=[[1.363e-134, 9.273e-78]],
                forcefield=ff.lennard_jones,
                simulation="md",
            )
            self.assertEqual(a.number_of_particles, number_of_particles)

    def test_system_forcefield_energy_returning_a_scalar_is_rejected(self):
        class ScalarEnergy:
            def __init__(self, constants):
                self.constants = constants

            @property
            def diameter(self):
                return 1e-10

            def energy(self, dr):
                return 1.0

        with self.assertRaisesRegex(ValueError, "one value per separation"):
            util.System(2, 300, 8, [[1.0]], ScalarEnergy, 39.948, simulation="md")

    def test_system_forcefield_energy_returning_the_wrong_shape_is_rejected(self):
        class ShortEnergy:
            def __init__(self, constants):
                self.constants = constants

            @property
            def diameter(self):
                return 1e-10

            def energy(self, dr):
                return np.ones(3)

        with self.assertRaisesRegex(ValueError, "one value per separation"):
            util.System(2, 300, 8, [[1.0]], ShortEnergy, 39.948, simulation="md")

    def test_system_forcefield_energy_positive_at_50_angstrom_names_units(self):
        # Sigma given in Angstrom rather than metres: the pair energy never
        # falls to zero on the 0.1 to 50 Angstrom grid.
        with self.assertRaisesRegex(ValueError, "still positive at 50 Angstrom"):
            util.System(
                2, 300, 8, [[3.4, 1.58e-21]], ff.lennard_jones_sigma_epsilon, 39.948,
                simulation="md",
            )

    def test_system_random_buckingham_core_energy_is_near_zero(self):
        state = np.random.get_state()
        np.random.seed(0)
        try:
            a = util.System(
                10,
                100,
                40,
                init_conf="random",
                mass=39.948,
                constants=[[1.69e-15, 3.66e10, 1.01e-77]],
                forcefield=ff.buckingham,
                simulation="md",
            )
        finally:
            np.random.set_state(state)
        self.assertTrue(a.cores[0] > 3e-10)
        r = np.logspace(-11, np.log10(5e-9), 4000)
        well_depth = ff.buckingham([1.69e-15, 3.66e10, 1.01e-77]).energy(r).min()
        core_energy = ff.buckingham(
            [1.69e-15, 3.66e10, 1.01e-77]
        ).energy(np.array([a.cores[0]]))[0]
        self.assertTrue(abs(core_energy) < 1e-3 * abs(well_depth))
