import copy
import pickle
import unittest

from pylj.model import Model
from pylj.potentials import LennardJones, Species
from pylj.tests.argon import ARGON, LARGER, LJ_ARGON, LJ_ARGON_LARGER, LJ_LARGER, MIXTURE_MODEL


class TestModel(unittest.TestCase):
    def test_single_builds_one_species_and_its_own_pair(self):
        model = Model.single(ARGON, LJ_ARGON)
        self.assertEqual(model.species, (ARGON,))
        self.assertEqual(model.pair_potentials, {(ARGON, ARGON): LJ_ARGON})

    def test_potential_is_found_in_either_order(self):
        self.assertIs(MIXTURE_MODEL.potential(ARGON, LARGER), LJ_ARGON_LARGER)
        self.assertIs(MIXTURE_MODEL.potential(LARGER, ARGON), LJ_ARGON_LARGER)

    def test_potential_shows_the_species_that_differs(self):
        twin = Species(mass=40.0, name="argon")
        with self.assertRaisesRegex(KeyError, "mass=40.0"):
            Model.single(ARGON, LJ_ARGON).potential(twin, twin)

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

    def test_rejects_a_repeated_species(self):
        twin = Species(mass=ARGON.mass, name=ARGON.name)
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            Model((ARGON, twin), {(ARGON, ARGON): LJ_ARGON})

    def test_accepts_any_sequence_of_species(self):
        model = Model([ARGON], {(ARGON, ARGON): LJ_ARGON})
        self.assertEqual(model.species, (ARGON,))

    def test_names_single_for_an_argument_that_is_not_a_sequence_or_mapping(self):
        with self.assertRaisesRegex(TypeError, "Model.single"):
            Model(ARGON, {(ARGON, ARGON): LJ_ARGON})
        with self.assertRaisesRegex(TypeError, "Model.single"):
            Model("argon", {(ARGON, ARGON): LJ_ARGON})
        with self.assertRaisesRegex(TypeError, "Model.single"):
            Model((ARGON,), LJ_ARGON)

    def test_rejects_a_string_as_a_species(self):
        with self.assertRaisesRegex(TypeError, r"Species\(mass="):
            Model(("argon",), {("argon", "argon"): LJ_ARGON})

    def test_rejects_a_key_that_is_not_a_pair(self):
        with self.assertRaisesRegex(ValueError, "must be a pair of species"):
            Model((ARGON,), {ARGON: LJ_ARGON})
        with self.assertRaisesRegex(ValueError, "must be a pair of species"):
            Model((ARGON,), {(ARGON, ARGON): LJ_ARGON, (ARGON, ARGON, ARGON): LJ_ARGON})

    def test_rejects_an_entry_naming_something_outside_the_species(self):
        with self.assertRaisesRegex(ValueError, "not one of the species"):
            Model((ARGON,), {(ARGON, ARGON): LJ_ARGON, (ARGON, LARGER): LJ_ARGON_LARGER})
        with self.assertRaisesRegex(ValueError, "'argon' is not one of the species"):
            Model((ARGON,), {("argon", "argon"): LJ_ARGON})

    def test_is_frozen(self):
        model = Model.single(ARGON, LJ_ARGON)
        with self.assertRaises(AttributeError):
            model.species = ()

    def test_copies_the_pair_potentials(self):
        pair_potentials = {(ARGON, ARGON): LJ_ARGON}
        model = Model((ARGON,), pair_potentials)
        pair_potentials[(ARGON, ARGON)] = LJ_LARGER
        self.assertIs(model.potential(ARGON, ARGON), LJ_ARGON)

    def test_pair_potentials_are_read_only(self):
        model = Model.single(ARGON, LJ_ARGON)
        with self.assertRaises(TypeError):
            model.pair_potentials[(ARGON, ARGON)] = LJ_LARGER

    def test_repr_names_the_species_and_potentials(self):
        text = repr(Model.single(ARGON, LJ_ARGON))
        self.assertTrue(text.startswith("Model(species=("))
        self.assertIn("LennardJones(epsilon=", text)


class TestPickle(unittest.TestCase):
    def test_a_model_survives_a_round_trip(self):
        # A potential compares by identity, so the copies are compared by
        # what they are made of, which their repr gives.
        model = pickle.loads(pickle.dumps(MIXTURE_MODEL))
        self.assertEqual(model.species, MIXTURE_MODEL.species)
        self.assertEqual(
            {pair: repr(one) for pair, one in model.pair_potentials.items()},
            {pair: repr(one) for pair, one in MIXTURE_MODEL.pair_potentials.items()},
        )
        self.assertEqual(repr(model.potential(ARGON, LARGER)), repr(LJ_ARGON_LARGER))

    def test_a_model_survives_a_deep_copy(self):
        # The copy rebuilds the mapping keys as new Species, so looking a
        # pair up works only because a Species is hashed by its values.
        model = copy.deepcopy(MIXTURE_MODEL)
        self.assertEqual(model.species, MIXTURE_MODEL.species)
        self.assertEqual(repr(model.potential(ARGON, LARGER)), repr(LJ_ARGON_LARGER))
