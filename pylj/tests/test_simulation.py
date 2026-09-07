import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import md, placement, simulation
from pylj.tests.argon import ARGON, ARGON_MODEL, WELL_MODEL


class TestInitialEnergyCheck(unittest.TestCase):
    def test_refuses_an_overlapping_lattice(self):
        # 16 argon on a 4 by 4 lattice in a 10 Angstrom box are 2.5 Angstrom
        # apart, inside sigma: about 90 k_B T of potential energy per atom.
        c = placement.place_square(16, (ARGON,), 10e-10)
        energy = c.potential_energy(ARGON_MODEL["pair_potentials"], 5e-10)
        with self.assertRaisesRegex(ValueError, "k_B T of potential energy"):
            simulation._check_initial_energy(energy, 16, 300)

    def test_refuses_a_lattice_inside_a_hard_core(self):
        c = placement.place_square(16, (ARGON,), 10e-10)
        energy = c.potential_energy(WELL_MODEL["pair_potentials"], 5e-10)
        with self.assertRaisesRegex(ValueError, "not finite"):
            simulation._check_initial_energy(energy, 16, 300)

    def test_accepts_a_lattice_below_the_limit(self):
        # 16 argon in a 12 Angstrom box store about 5.6 k_B T per atom
        # at 300 K, under the limit of 10.
        c = placement.place_square(16, (ARGON,), 12e-10)
        energy = c.potential_energy(ARGON_MODEL["pair_potentials"], 6e-10)
        simulation._check_initial_energy(energy, 16, 300)


class Counting(simulation.Simulation):
    """A simulation whose step and sample only count, for the base class tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampled = 0

    def step(self):
        self.steps += 1

    def sample(self):
        self.sampled += 1
        self.samples.add(step=self.steps)


class TestSimulation(unittest.TestCase):
    def build(self, box=40e-10, **kwargs):
        c = placement.place_square(4, (ARGON,), box)
        return Counting(c, ARGON_MODEL["pair_potentials"], **kwargs)

    def test_holds_the_configuration_and_the_model(self):
        s = self.build(seed=1)
        self.assertEqual(s.configuration.number_of_atoms, 4)
        self.assertEqual(s.pair_potentials, ARGON_MODEL["pair_potentials"])
        self.assertEqual(s.steps, 0)
        self.assertIsInstance(s.samples, simulation.Samples)
        self.assertEqual(s.samples.step.size, 0)
        self.assertEqual(s.rng.random(), np.random.default_rng(1).random())

    def test_samples_add_appends_one_value_to_each_array(self):
        samples = simulation.Samples()
        samples.add(step=3)
        samples.add(step=7)
        assert_equal(samples.step, [3, 7])

    def test_samples_add_refuses_a_partial_or_unknown_sample(self):
        # A missing or unknown name would leave the arrays out of step, so
        # both are refused and the record is left as it was.
        samples = md.MDSamples()
        with self.assertRaisesRegex(ValueError, "one value for each"):
            samples.add(step=3)
        with self.assertRaisesRegex(ValueError, "one value for each"):
            samples.add(step=3, pressure=1.0, mass=2.0)
        self.assertEqual(samples.step.size, 0)

    def test_cut_off_defaults_to_15_angstrom_or_half_the_box(self):
        assert_almost_equal(self.build().cut_off * 1e10, 15)
        assert_almost_equal(self.build(box=20e-10).cut_off * 1e10, 10)
        assert_almost_equal(self.build(cut_off=12e-10).cut_off * 1e10, 12)

    def test_refuses_a_cut_off_beyond_half_the_box(self):
        with self.assertRaisesRegex(ValueError, "exceeds half the box"):
            self.build(cut_off=25e-10)

    def test_validates_the_model(self):
        c = placement.place_square(4, (ARGON,), 40e-10)
        with self.assertRaisesRegex(ValueError, "no entry for the pair"):
            Counting(c, {})

    def test_step_and_sample_are_abstract(self):
        class Stepless(simulation.Simulation):
            def sample(self):
                pass

        c = placement.place_square(4, (ARGON,), 40e-10)
        with self.assertRaisesRegex(TypeError, "step"):
            Stepless(c, ARGON_MODEL["pair_potentials"])

    def test_restart_starts_a_fresh_record_and_shares_the_model(self):
        s = self.build(seed=1)
        for _ in range(3):
            s.step()
            s.sample()
        production = s.restart()
        self.assertIsNot(production, s)
        self.assertIsInstance(production, Counting)
        self.assertIs(production.configuration, s.configuration)
        self.assertIs(production.pair_potentials, s.pair_potentials)
        self.assertEqual(production.steps, 0)
        self.assertIsNot(production.samples, s.samples)
        self.assertEqual(production.samples.step.size, 0)
        self.assertEqual(s.steps, 3)
        assert_equal(s.samples.step, [1, 2, 3])

    def test_restart_shares_state_a_subclass_adds(self):
        # A shallow copy: a subclass holding other per-run state extends
        # restart to reset it.
        s = self.build()
        s.sample()
        self.assertEqual(s.restart().sampled, 1)

    def test_restart_copies_the_generator_state(self):
        s = self.build(seed=1)
        production = s.restart()
        self.assertIsNot(production.rng, s.rng)
        # Equal state: the next draw agrees. Independent: drawing from one
        # leaves the other where it was.
        self.assertEqual(production.rng.random(), s.rng.random())
        s.rng.random()
        self.assertNotEqual(production.rng.random(), s.rng.random())
