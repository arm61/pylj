import unittest

from pylj.model import Model
from pylj.potentials import LennardJones
from pylj.tests.argon import ARGON, LARGER, LJ_ARGON, LJ_ARGON_LARGER, LJ_LARGER


class TestModel(unittest.TestCase):
    def test_single_builds_one_species_and_its_own_pair(self):
        model = Model.single(ARGON, LJ_ARGON)
        self.assertEqual(model.species, (ARGON,))
        self.assertEqual(model.pair_potentials, {(ARGON, ARGON): LJ_ARGON})

    def test_potential_is_found_in_either_order(self):
        model = Model(
            (ARGON, LARGER),
            {
                (ARGON, ARGON): LJ_ARGON,
                (LARGER, LARGER): LJ_LARGER,
                (LARGER, ARGON): LJ_ARGON_LARGER,
            },
        )
        self.assertIs(model.potential(ARGON, LARGER), LJ_ARGON_LARGER)
        self.assertIs(model.potential(LARGER, ARGON), LJ_ARGON_LARGER)

    def test_potential_names_a_pair_outside_the_model(self):
        model = Model.single(ARGON, LJ_ARGON)
        with self.assertRaisesRegex(KeyError, "argon and larger"):
            model.potential(ARGON, LARGER)

    def test_rejects_a_missing_pair(self):
        with self.assertRaisesRegex(ValueError, "no entry for the pair .*larger"):
            Model((ARGON, LARGER), {(ARGON, ARGON): LJ_ARGON, (LARGER, LARGER): LJ_LARGER})

    def test_rejects_a_cross_pair_given_in_both_orders(self):
        with self.assertRaisesRegex(ValueError, "in both orders"):
            Model(
                (ARGON, LARGER),
                {
                    (ARGON, ARGON): LJ_ARGON,
                    (LARGER, LARGER): LJ_LARGER,
                    (ARGON, LARGER): LJ_ARGON_LARGER,
                    (LARGER, ARGON): LJ_ARGON_LARGER,
                },
            )

    def test_rejects_a_potential_class_in_place_of_an_instance(self):
        with self.assertRaisesRegex(TypeError, "PairPotential instance"):
            Model((ARGON,), {(ARGON, ARGON): LennardJones})

    def test_rejects_no_species(self):
        with self.assertRaisesRegex(ValueError, "at least one Species"):
            Model((), {})

    def test_is_frozen(self):
        model = Model.single(ARGON, LJ_ARGON)
        with self.assertRaises(AttributeError):
            model.species = ()
