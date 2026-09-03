import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import forcefields as ff
from pylj import util


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
        constants = [[1.363e-134, 9.273e-78], [1.365e-130, 9.278e-77]]
        a = util.System(
            2, 300, 8, constants, ff.lennard_jones, 39.948, simulation="md", diameter=3.0
        )
        assert_almost_equal(a.diameters, [3e-10, 3e-10])

    def test_system_diameter_list_is_per_type(self):
        constants = [[1.363e-134, 9.273e-78], [1.365e-130, 9.278e-77]]
        a = util.System(
            2, 300, 8, constants, ff.lennard_jones, 39.948, simulation="md", diameter=[3.0, 5.0]
        )
        assert_almost_equal(a.diameters, [3e-10, 5e-10])

    def test_system_diameter_array_is_per_type(self):
        constants = [[1.363e-134, 9.273e-78], [1.365e-130, 9.278e-77]]
        a = util.System(
            2, 300, 8, constants, ff.lennard_jones, 39.948, simulation="md",
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

    def test_system_forcefield_without_a_diameter_is_rejected(self):
        class NoDiameter:
            def __init__(self, constants):
                self.constants = constants

        constants = [[1.363e-134, 9.273e-78]]
        with self.assertRaisesRegex(ValueError, "diameter"):
            util.System(2, 300, 8, constants, NoDiameter, 39.948, simulation="md")
