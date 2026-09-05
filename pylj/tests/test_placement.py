import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import pairwise, placement
from pylj.constants import ATOMIC_MASS_UNIT
from pylj.potentials import LennardJones, SquareWell
from pylj.tests.argon import (
    ARGON,
    ARGON_MODEL,
    BUCKINGHAM_ARGON,
    BUCKINGHAM_MODEL,
    LARGER,
    LJ_ARGON,
    MIXTURE_MODEL,
    WELL,
    WELL_MIXTURE_MODEL,
    WELL_MODEL,
)


def place(
    n,
    temperature,
    box,
    *,
    init_conf="square",
    placement_temperature=None,
    cut_off=None,
    seed=None,
    model=ARGON_MODEL,
    **overrides,
):
    """simulation.place with the argon model and a seeded generator."""
    kwargs = dict(model)
    kwargs.update(overrides)
    return placement.place(
        n,
        temperature,
        box,
        init_conf=init_conf,
        placement_temperature=placement_temperature,
        cut_off=cut_off,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


class TestPlacement(unittest.TestCase):
    def test_square_fills_the_lattice_in_order(self):
        c = placement.place_square(2, (ARGON,), 8e-10)
        assert_almost_equal(c.position * 1e10, [[2, 2], [2, 6]])
        assert_equal(c.species_index, [0, 0])
        self.assertEqual(c.box, 8e-10)

    def test_square_assigns_species_in_turn(self):
        c = placement.place_square(5, (ARGON, LARGER), 30e-10)
        assert_equal(c.species_index, [0, 1, 0, 1, 0])
        np.testing.assert_allclose(
            c.masses, np.array([39.948, 80.0, 39.948, 80.0, 39.948]) * ATOMIC_MASS_UNIT
        )

    def test_metropolis_places_inside_the_box(self):
        c, _ = place(10, 300, 20, init_conf="metropolis", seed=0)
        self.assertTrue(np.all((0 <= c.position) & (c.position < c.box)))
        self.assertEqual(c.number_of_particles, 10)

    def test_metropolis_places_a_hard_core_outside_its_diameter(self):
        # A trial inside the square well's core costs infinite energy and is
        # always rejected, so no pair is closer than sigma.
        c, cut_off = place(50, 300, 30, init_conf="metropolis", seed=1, model=WELL_MODEL)
        distance = c.pairs(WELL_MODEL["pair_potentials"], cut_off).distance
        self.assertGreaterEqual(distance.min(), WELL.sigma)

    def test_metropolis_places_a_mixture_with_each_pairs_own_potential(self):
        # Hard cores of three different diameters: no pair may sit inside the
        # core of its own potential. A placement using the wrong potential
        # for a pair lets it inside the true core.
        potentials = WELL_MIXTURE_MODEL["pair_potentials"]
        c, cut_off = place(30, 300, 40, init_conf="metropolis", seed=0, model=WELL_MIXTURE_MODEL)
        distance = c.pairs(potentials, cut_off).distance
        for mask, type_1, type_2 in pairwise.species_pairs(c.species_index):
            potential = pairwise.pair_potential(potentials, c.species[type_1], c.species[type_2])
            core = potential.sigma
            self.assertGreaterEqual(distance[mask].min(), core)

    def test_metropolis_places_buckingham_outside_its_turnover(self):
        # The Buckingham form falls to minus infinity inside its barrier, so
        # a trial there would be accepted as downhill; the wall inside the
        # turnover keeps every pair outside it.
        c, cut_off = place(30, 300, 40, init_conf="metropolis", seed=0, model=BUCKINGHAM_MODEL)
        pairs = c.pairs(BUCKINGHAM_MODEL["pair_potentials"], cut_off)
        self.assertGreater(pairs.distance.min(), BUCKINGHAM_ARGON.turnover)
        self.assertTrue(np.isfinite(pairs.energy).all())

    def test_metropolis_places_a_soft_potential_outside_its_core(self):
        # Lennard-Jones has no hard core, but at 100 K a pair inside 0.8
        # sigma costs over 40 well depths and is never accepted.
        c, cut_off = place(30, 100, 40, init_conf="metropolis", seed=0)
        distance = c.pairs(ARGON_MODEL["pair_potentials"], cut_off).distance
        self.assertGreater(distance.min(), 0.8 * LJ_ARGON.sigma)

    def test_metropolis_too_dense_raises(self):
        with self.assertRaisesRegex(ValueError, f"after {placement.PLACEMENT_ATTEMPTS} attempts"):
            place(200, 100, 20, init_conf="metropolis", seed=0)

    def test_metropolis_seed_reproduces_placement(self):
        first, _ = place(10, 100, 40, init_conf="metropolis", seed=3)
        second, _ = place(10, 100, 40, init_conf="metropolis", seed=3)
        other, _ = place(10, 100, 40, init_conf="metropolis", seed=4)
        assert_equal(first.position, second.position)
        self.assertFalse(np.array_equal(first.position, other.position))

    def test_metropolis_placement_temperature_governs_success(self):
        # 50 argon particles in a 27 Angstrom box: as near-hard discs of
        # diameter sigma (placement at 1 K) they exceed the packing that
        # sequential insertion reaches (the placement fails for every seed
        # tried), but at 1000 K closer contacts are tolerated.
        with self.assertRaisesRegex(ValueError, "Could not place"):
            place(50, 100, 27, init_conf="metropolis", seed=0, placement_temperature=1.0)
        hot, _ = place(50, 100, 27, init_conf="metropolis", seed=0, placement_temperature=1000)
        self.assertEqual(hot.number_of_particles, 50)

    def test_placement_temperature_defaults_to_the_run_temperature(self):
        # The same seed at the default and at an explicit placement
        # temperature equal to the run temperature gives the same positions.
        default, _ = place(10, 300, 40, init_conf="metropolis", seed=2)
        explicit, _ = place(10, 300, 40, init_conf="metropolis", seed=2, placement_temperature=300)
        assert_equal(default.position, explicit.position)

    def test_rejects_a_bad_placement_temperature(self):
        for bad in (0, -1, np.inf):
            with self.assertRaisesRegex(ValueError, "placement_temperature must be positive"):
                place(2, 300, 8, placement_temperature=bad)


class TestPlace(unittest.TestCase):
    def test_converts_the_box_and_cut_off_from_angstrom(self):
        c, cut_off = place(2, 300, 8)
        assert_almost_equal(c.box * 1e10, 8)
        assert_almost_equal(cut_off * 1e10, 4.0)
        assert_almost_equal(c.position * 1e10, [[2, 2], [2, 6]])
        _, given = place(2, 300, 40, cut_off=10)
        assert_almost_equal(given * 1e10, 10)

    def test_cut_off_defaults_to_15_angstrom_or_half_the_box(self):
        _, large = place(2, 300, 40)
        _, small = place(2, 300, 20)
        assert_almost_equal(large * 1e10, 15)
        assert_almost_equal(small * 1e10, 10)

    def test_refuses_a_cut_off_beyond_half_the_box(self):
        with self.assertRaisesRegex(ValueError, "exceeds half the box"):
            place(2, 300, 40, cut_off=25)

    def test_refuses_a_box_outside_the_viewer_range(self):
        for box in (2, 1000):
            with self.assertRaisesRegex(ValueError, "between 4 and 600 Angstrom"):
                place(2, 300, box)

    def test_refuses_an_unknown_init_conf(self):
        with self.assertRaisesRegex(ValueError, "'square' or 'metropolis'"):
            place(2, 300, 100, init_conf="horseradish")

    def test_refuses_fewer_than_one_particle(self):
        with self.assertRaisesRegex(ValueError, "at least one particle"):
            place(0, 300, 20)

    def test_rejects_a_non_positive_or_infinite_temperature(self):
        for temperature in (0, -10, np.inf):
            with self.assertRaisesRegex(ValueError, "temperature must be positive"):
                place(2, temperature, 8)

    def test_rejects_a_missing_pair_potential(self):
        incomplete = {(ARGON, ARGON): LJ_ARGON, (LARGER, LARGER): LJ_ARGON}
        with self.assertRaisesRegex(ValueError, "no entry for the pair .*larger"):
            place(2, 300, 12, species=[ARGON, LARGER], pair_potentials=incomplete)

    def test_accepts_a_pair_potential_keyed_in_either_order(self):
        reversed_cross = dict(MIXTURE_MODEL["pair_potentials"])
        reversed_cross[(LARGER, ARGON)] = reversed_cross.pop((ARGON, LARGER))
        c, _ = place(2, 300, 12, species=[ARGON, LARGER], pair_potentials=reversed_cross)
        assert_equal(c.species_index, [0, 1])

    def test_rejects_a_potential_class_in_place_of_an_instance(self):
        with self.assertRaisesRegex(TypeError, "PairPotential instance"):
            place(2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): LennardJones})

    def test_rejects_a_cross_pair_given_in_both_orders(self):
        both_orders = dict(MIXTURE_MODEL["pair_potentials"])
        both_orders[(LARGER, ARGON)] = LJ_ARGON
        with self.assertRaisesRegex(ValueError, "in both orders"):
            place(2, 300, 12, species=[ARGON, LARGER], pair_potentials=both_orders)

    def test_rejects_no_species(self):
        with self.assertRaisesRegex(ValueError, "at least one Species"):
            place(2, 300, 8, species=[], pair_potentials={})

    def test_refuses_a_potential_still_repulsive_at_the_cut_off(self):
        # Sigma given in Angstrom: the pair energy is astronomically positive
        # at the cut-off, where a sensible potential has died away.
        in_angstrom = LennardJones(epsilon=1.577e-21, sigma=3.372)
        with self.assertRaisesRegex(ValueError, "at the cut-off"):
            place(2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): in_angstrom})

    def test_refuses_a_potential_still_attractive_at_the_cut_off(self):
        # Epsilon typed in kJ/mol: a well about 1e17 k_B T deep at the cut-off.
        deep = LennardJones(epsilon=0.95, sigma=3.372e-10)
        with self.assertRaisesRegex(ValueError, "at the cut-off"):
            place(2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): deep})

    def test_refuses_a_hard_core_wider_than_the_cut_off(self):
        wide = SquareWell(epsilon=1.5e-21, sigma=8e-10, lambda_=1.5)
        with self.assertRaisesRegex(ValueError, "hard core is wider than the cut-off"):
            place(2, 300, 10, species=[ARGON], pair_potentials={(ARGON, ARGON): wide})

    def test_refuses_a_cross_potential_still_repulsive_at_the_cut_off(self):
        mistyped = dict(MIXTURE_MODEL["pair_potentials"])
        mistyped[(ARGON, LARGER)] = LennardJones(epsilon=1.577e-21, sigma=4.186)
        with self.assertRaisesRegex(ValueError, "between argon and larger"):
            place(4, 100, 60, species=MIXTURE_MODEL["species"], pair_potentials=mistyped)

    def test_names_half_the_box_when_the_cut_off_came_from_it(self):
        in_angstrom = LennardJones(epsilon=1.577e-21, sigma=3.372)
        with self.assertRaisesRegex(ValueError, r"\(half the box\).*Use a larger box"):
            place(2, 300, 8, species=[ARGON], pair_potentials={(ARGON, ARGON): in_angstrom})
