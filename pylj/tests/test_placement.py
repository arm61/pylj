import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from pylj import pairwise, placement
from pylj.configuration import Configuration
from pylj.constants import ATOMIC_MASS_UNIT
from pylj.model import Model
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
    max_strain=0.05,
    cut_off=None,
    seed=None,
    model=ARGON_MODEL,
):
    """placement.place with the argon model and a seeded generator."""
    return placement.place(
        n,
        temperature,
        box,
        model=model,
        init_conf=init_conf,
        placement_temperature=placement_temperature,
        max_strain=max_strain,
        cut_off=cut_off,
        rng=np.random.default_rng(seed),
    )


def neighbour_count(configuration):
    """Mean number of atoms at the nearest distance, over the whole cell.

    The window has to be wider than the spread the strain allows, which at
    the default ``max_strain`` of 0.05 is under 4 per cent, and narrower
    than the next shell at ``sqrt(3)`` times the nearest distance.
    """
    distance, _ = pairwise.dist(configuration.position, configuration.box)
    nearest = distance.min()
    return (distance < nearest * 1.07).sum() * 2 / configuration.number_of_atoms


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
        self.assertEqual(c.number_of_atoms, 10)

    def test_metropolis_places_a_hard_core_outside_its_diameter(self):
        # A trial inside the square well's core costs infinite energy and is
        # always rejected, so no pair is closer than sigma.
        c, cut_off = place(50, 300, 30, init_conf="metropolis", seed=1, model=WELL_MODEL)
        distance = c.pairs(WELL_MODEL, cut_off).distance
        self.assertGreaterEqual(distance.min(), WELL.sigma)

    def test_metropolis_places_a_mixture_with_each_pairs_own_potential(self):
        # Hard cores of three different diameters: no pair may sit inside the
        # core of its own potential. A placement using the wrong potential
        # for a pair lets it inside the true core.
        c, cut_off = place(30, 300, 40, init_conf="metropolis", seed=0, model=WELL_MIXTURE_MODEL)
        distance = c.pairs(WELL_MIXTURE_MODEL, cut_off).distance
        for mask, type_1, type_2 in pairwise.species_pairs(c.species_index):
            potential = WELL_MIXTURE_MODEL.potential(c.species[type_1], c.species[type_2])
            core = potential.sigma
            self.assertGreaterEqual(distance[mask].min(), core)

    def test_metropolis_places_buckingham_outside_its_min_separation(self):
        # The Buckingham formula falls to minus infinity inside its barrier,
        # so a trial there would be accepted as downhill; the potential's
        # min_separation makes such a trial cost infinite energy instead.
        c, cut_off = place(30, 300, 40, init_conf="metropolis", seed=0, model=BUCKINGHAM_MODEL)
        pairs = c.pairs(BUCKINGHAM_MODEL, cut_off)
        self.assertGreater(pairs.distance.min(), BUCKINGHAM_ARGON.min_separation)
        self.assertTrue(np.isfinite(pairs.energy).all())

    def test_metropolis_places_a_soft_potential_outside_its_core(self):
        # Lennard-Jones has no hard core, but at 100 K a pair inside 0.8
        # sigma costs over 40 well depths and is never accepted.
        c, cut_off = place(30, 100, 40, init_conf="metropolis", seed=0)
        distance = c.pairs(ARGON_MODEL, cut_off).distance
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
        # 50 argon atoms in a 27 Angstrom box: as near-hard discs of
        # diameter sigma (placement at 1 K) they exceed the packing that
        # sequential insertion reaches (the placement fails for every seed
        # tried), but at 1000 K closer contacts are tolerated.
        with self.assertRaisesRegex(ValueError, "Could not place"):
            place(50, 100, 27, init_conf="metropolis", seed=0, placement_temperature=1.0)
        hot, _ = place(50, 100, 27, init_conf="metropolis", seed=0, placement_temperature=1000)
        self.assertEqual(hot.number_of_atoms, 50)

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
        with self.assertRaisesRegex(ValueError, "'square', 'triangular' or 'metropolis'"):
            place(2, 300, 100, init_conf="horseradish")

    def test_refuses_fewer_than_one_atom(self):
        with self.assertRaisesRegex(ValueError, "at least one atom"):
            place(0, 300, 20)

    def test_rejects_a_non_positive_or_infinite_temperature(self):
        for temperature in (0, -10, np.inf):
            with self.assertRaisesRegex(ValueError, "temperature must be positive"):
                place(2, temperature, 8)

    def test_refuses_a_potential_still_repulsive_at_the_cut_off(self):
        # Sigma given in Angstrom: the pair energy is astronomically positive
        # at the cut-off, where a sensible potential has died away.
        in_angstrom = LennardJones(epsilon=1.577e-21, sigma=3.372)
        with self.assertRaisesRegex(ValueError, "at the cut-off"):
            place(2, 300, 8, model=Model.single(ARGON, in_angstrom))

    def test_refuses_a_potential_still_attractive_at_the_cut_off(self):
        # Epsilon typed in kJ/mol: a well about 1e17 k_B T deep at the cut-off.
        deep = LennardJones(epsilon=0.95, sigma=3.372e-10)
        with self.assertRaisesRegex(ValueError, "at the cut-off"):
            place(2, 300, 8, model=Model.single(ARGON, deep))

    def test_refuses_a_hard_core_wider_than_the_cut_off(self):
        wide = SquareWell(epsilon=1.5e-21, sigma=8e-10, lambda_=1.5)
        with self.assertRaisesRegex(ValueError, "hard core is wider than the cut-off"):
            place(2, 300, 10, model=Model.single(ARGON, wide))

    def test_refuses_a_cross_potential_still_repulsive_at_the_cut_off(self):
        mistyped = dict(MIXTURE_MODEL.pair_potentials)
        mistyped[(ARGON, LARGER)] = LennardJones(epsilon=1.577e-21, sigma=4.186)
        with self.assertRaisesRegex(ValueError, "between argon and larger"):
            place(4, 100, 60, model=Model(MIXTURE_MODEL.species, mistyped))

    def test_names_half_the_box_when_the_cut_off_came_from_it(self):
        in_angstrom = LennardJones(epsilon=1.577e-21, sigma=3.372)
        with self.assertRaisesRegex(ValueError, r"\(half the box\).*Use a larger box"):
            place(2, 300, 8, model=Model.single(ARGON, in_angstrom))

    def test_place_builds_a_triangular_lattice(self):
        c, _ = place(56, 300, 60, init_conf="triangular")
        self.assertAlmostEqual(neighbour_count(c), 6.0, places=6)


class TestPlaceTriangular(unittest.TestCase):
    def test_every_atom_has_six_nearest_neighbours(self):
        for atoms in (56, 168):
            with self.subTest(atoms=atoms):
                c = placement.place_triangular(atoms, (ARGON,), 60e-10)
                self.assertAlmostEqual(neighbour_count(c), 6.0, places=6)

    def test_fills_every_site(self):
        c = placement.place_triangular(56, (ARGON,), 60e-10)
        self.assertEqual(c.number_of_atoms, 56)
        distance, _ = pairwise.dist(c.position, c.box)
        self.assertGreater(distance.min(), 0)

    def test_rows_alternate_by_half_a_column(self):
        c = placement.place_triangular(56, (ARGON,), 56e-10)
        # Every atom of a row is built from the same expression, so the row
        # values are exactly equal and can be matched exactly. Any tolerance
        # here would have to be well under the row spacing, itself of order
        # 1e-10 metres.
        y = np.unique(c.position[:, 1])
        self.assertEqual(y.size, 8)
        first = np.sort(c.position[c.position[:, 1] == y[0], 0])
        second = np.sort(c.position[c.position[:, 1] == y[1], 0])
        spacing = 56e-10 / 7
        assert_allclose(second - first, spacing / 2)

    def test_strain_is_within_the_tolerance(self):
        box = 60e-10
        c = placement.place_triangular(56, (ARGON,), box)
        rows = np.unique(c.position[:, 1]).size
        columns = c.number_of_atoms // rows
        strain = abs(columns / rows / (np.sqrt(3) / 2) - 1)
        self.assertEqual(rows % 2, 0)
        self.assertLess(strain, 0.05)

    def test_refuses_a_count_that_does_not_fit_and_names_ones_that_do(self):
        with self.assertRaisesRegex(ValueError, "90") as caught:
            placement.place_triangular(100, (ARGON,), 60e-10)
        self.assertIn("120", str(caught.exception))
        placement.place_triangular(90, (ARGON,), 60e-10)
        placement.place_triangular(120, (ARGON,), 60e-10)

    def test_refuses_a_count_with_no_even_row_factorisation(self):
        # 97 is prime, so no even number of rows divides it.
        with self.assertRaisesRegex(ValueError, "Use 90 or 120 atoms"):
            placement.place_triangular(97, (ARGON,), 60e-10)

    def test_names_only_one_count_at_the_bottom_of_the_range(self):
        # Nothing below 30 fits, so only the count above is offered.
        with self.assertRaisesRegex(ValueError, r"Use 30 atoms\."):
            placement.place_triangular(25, (ARGON,), 60e-10)

    def test_a_looser_tolerance_accepts_more_counts(self):
        with self.assertRaises(ValueError):
            placement.place_triangular(100, (ARGON,), 60e-10)
        c = placement.place_triangular(100, (ARGON,), 60e-10, max_strain=0.2)
        self.assertEqual(c.number_of_atoms, 100)

    def test_rejects_a_max_strain_that_is_not_positive(self):
        with self.assertRaisesRegex(ValueError, "max_strain"):
            placement.place_triangular(56, (ARGON,), 60e-10, max_strain=0)

    def test_staggering_the_rows_lowers_the_energy(self):
        sigma = 3.372e-10
        atoms = 56
        box = np.sqrt(atoms * sigma**2 / 0.9)
        cut_off = min(15e-10, box / 2)
        triangular = placement.place_triangular(atoms, (ARGON,), box)
        rows = np.unique(triangular.position[:, 1]).size
        columns = atoms // rows
        # The same grid of sites with the rows lined up rather than
        # staggered, which is the one thing the triangular lattice changes.
        lined_up = Configuration(
            np.array(
                [
                    (i * box / columns, (j + 0.5) * box / rows)
                    for j in range(rows)
                    for i in range(columns)
                ]
            ),
            (ARGON,),
            np.zeros(atoms, dtype=np.int64),
            box,
        )
        self.assertLess(
            triangular.potential_energy(ARGON_MODEL, cut_off),
            lined_up.potential_energy(ARGON_MODEL, cut_off),
        )

    def test_fills_the_sites_row_by_row(self):
        # The species alternate along a row, so the first row of a 7 by 8
        # lattice reads 0, 1, 0, 1, 0, 1, 0 from left to right.
        c = placement.place_triangular(56, (ARGON, LARGER), 60e-10)
        y = np.unique(c.position[:, 1])
        first_row = c.position[:, 1] == y[0]
        order = np.argsort(c.position[first_row, 0])
        assert_equal(c.species_index[first_row][order], [0, 1, 0, 1, 0, 1, 0])

    def test_refuses_a_max_strain_above_the_limit(self):
        with self.assertRaisesRegex(ValueError, "max_strain is a fraction"):
            placement.place_triangular(56, (ARGON,), 60e-10, max_strain=5)

    def test_max_strain_is_a_fraction_of_the_ratio(self):
        # 7 by 8 sits 0.0090 from sqrt(3) / 2 in absolute terms and 0.0104
        # of it as a fraction, so a max_strain between the two refuses it.
        with self.assertRaisesRegex(ValueError, "max_strain"):
            placement.place_triangular(56, (ARGON,), 60e-10, max_strain=0.0095)
        placement.place_triangular(56, (ARGON,), 60e-10, max_strain=0.0105)

    def test_the_counts_that_fit_are_the_ones_documented(self):
        fits = []
        for atoms in range(2, 301):
            try:
                placement.place_triangular(atoms, (ARGON,), 60e-10)
            except ValueError:
                continue
            fits.append(atoms)
        self.assertEqual(fits, [30, 56, 90, 120, 168, 224, 270, 288])

    def test_every_atom_is_inside_the_box(self):
        box = 60e-10
        c = placement.place_triangular(56, (ARGON,), box)
        self.assertTrue(((c.position >= 0) & (c.position < box)).all())
