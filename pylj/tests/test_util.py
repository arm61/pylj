import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import forcefields as ff
from pylj import util

# Sigma and epsilon in metres and joules for two Lennard-Jones species: argon,
# and a larger particle with a 5 Angstrom core and the same well depth.
ARGON = [3.37e-10, 1.58e-21]
LARGER = [5.0e-10, 1.58e-21]


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
        box_length = a.box_length
        x = a.particles["xposition"]
        y = a.particles["yposition"]
        for i in range(a.number_of_particles):
            for j in range(i + 1, a.number_of_particles):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dx -= box_length * np.round(dx / box_length)
                dy -= box_length * np.round(dy / box_length)
                distance = np.sqrt(dx**2 + dy**2)
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

    def test_system_square_overlap_raises(self):
        with self.assertRaises(ValueError) as context:
            util.System(
                50,
                100,
                20,
                mass=39.948,
                constants=[[1.363e-134, 9.273e-78]],
                forcefield=ff.lennard_jones,
                simulation="md",
            )
        message = str(context.exception)
        self.assertTrue("repulsive core" in message)
        self.assertTrue("fit" in message)

    def test_system_square_between_core_and_diameter_is_accepted(self):
        # 100 argon particles in a 40 Angstrom box space 4.0 Angstrom apart:
        # above the 3.4 Angstrom repulsive core, but below the 3.78 Angstrom
        # drawn diameter, which used to be the (overly strict) threshold.
        a = util.System(
            100,
            100,
            40,
            mass=39.948,
            constants=[[1.363e-134, 9.273e-78]],
            forcefield=ff.lennard_jones,
            simulation="md",
        )
        assert_equal(a.number_of_particles, 100)

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
        box_length = a.box_length
        x = a.particles["xposition"]
        y = a.particles["yposition"]
        distances = []
        for i in range(a.number_of_particles):
            for j in range(i + 1, a.number_of_particles):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dx -= box_length * np.round(dx / box_length)
                dy -= box_length * np.round(dy / box_length)
                distances.append(np.sqrt(dx**2 + dy**2))
        self.assertTrue(all(distance >= a.cores[0] for distance in distances))
        self.assertTrue(any(distance < 8e-10 for distance in distances))

    def test_system_random_two_types_uses_the_mean_of_the_pair_cores(self):
        constants = [[1.363e-134, 9.273e-78], [1.365e-130, 9.278e-77]]
        state = np.random.get_state()
        np.random.seed(0)
        try:
            a = util.System(
                12,
                100,
                60,
                init_conf="random",
                mass=39.948,
                constants=constants,
                forcefield=ff.lennard_jones,
                simulation="md",
            )
        finally:
            np.random.set_state(state)
        box_length = a.box_length
        x = a.particles["xposition"]
        y = a.particles["yposition"]
        types = a.particles["types"]
        for i in range(a.number_of_particles):
            for j in range(i + 1, a.number_of_particles):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dx -= box_length * np.round(dx / box_length)
                dy -= box_length * np.round(dy / box_length)
                distance = np.sqrt(dx**2 + dy**2)
                type_i, type_j = int(types[i]), int(types[j])
                min_separation = (a.cores[type_i] + a.cores[type_j]) / 2
                self.assertTrue(distance >= min_separation)

    def test_system_cores_default_to_the_energy_zero(self):
        a = util.System(
            2, 300, 8, [[1.363e-134, 9.273e-78]], ff.lennard_jones, 39.948, simulation="md"
        )
        expected_sigma = (1.363e-134 / 9.273e-78) ** (1 / 6)
        assert_almost_equal(a.cores[0] * 1e10, expected_sigma * 1e10, decimal=3)

        b = util.System(
            2, 300, 8, [[1.0, 1.5e-10, 2.0]], ff.square_well, 39.948, simulation="md"
        )
        assert_almost_equal(b.cores[0], 1.5e-10, decimal=12)

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
        constants = [[1.363e-134, 9.273e-78], [1.365e-130, 9.278e-77]]
        with self.assertRaises(ValueError):
            util.System(
                2, 300, 8, constants, ff.lennard_jones, 39.948, simulation="md", diameter=[3.0]
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
