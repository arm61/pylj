import pickle
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from pylj import md, placement, simulation
from pylj.tests.argon import ARGON, ARGON_MODEL, LARGER, MIXTURE_MODEL


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
        return Counting(c, ARGON_MODEL, **kwargs)

    def test_holds_the_configuration_and_the_model(self):
        s = self.build(seed=1)
        self.assertEqual(s.configuration.number_of_atoms, 4)
        self.assertIs(s.model, ARGON_MODEL)
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

    def test_refuses_a_configuration_species_outside_the_model(self):
        c = placement.place_square(4, (LARGER,), 40e-10)
        with self.assertRaisesRegex(ValueError, "larger.*not in the model"):
            Counting(c, ARGON_MODEL)

    def test_accepts_a_configuration_using_some_of_the_model_species(self):
        c = placement.place_square(4, (ARGON,), 40e-10)
        self.assertIs(Counting(c, MIXTURE_MODEL).model, MIXTURE_MODEL)

    def test_step_and_sample_are_abstract(self):
        class Stepless(simulation.Simulation):
            def sample(self):
                pass

        c = placement.place_square(4, (ARGON,), 40e-10)
        with self.assertRaisesRegex(TypeError, "step"):
            Stepless(c, ARGON_MODEL)

    def test_restart_starts_a_fresh_record_and_shares_the_model(self):
        s = self.build(seed=1)
        for _ in range(3):
            s.step()
            s.sample()
        production = s.restart()
        self.assertIsNot(production, s)
        self.assertIsInstance(production, Counting)
        self.assertIs(production.configuration, s.configuration)
        self.assertIs(production.model, s.model)
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


class TestPickle(unittest.TestCase):
    def test_a_simulation_survives_a_round_trip(self):
        s = md.MDSimulation.initialise(
            ARGON_MODEL, number_of_atoms=4, temperature=100, box=20, seed=1
        )
        s.step()
        s.sample()
        copied = pickle.loads(pickle.dumps(s))
        # Stepping the copy needs its model, generator, cut-off and
        # timestep, so one comparison covers all four.
        s.step()
        copied.step()
        assert_allclose(copied.configuration.positions, s.configuration.positions)
