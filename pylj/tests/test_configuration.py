import copy
import pickle
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from pylj import placement
from pylj.configuration import Configuration, MDConfiguration
from pylj.constants import BOLTZMANN, KJ_PER_MOL
from pylj.model import Model
from pylj.potentials import PairPotential
from pylj.scattering import default_q_max
from pylj.tests.argon import (
    ARGON,
    ARGON_MODEL,
    BUCKINGHAM_ARGON,
    BUCKINGHAM_MODEL,
    LARGER,
    LJ_ARGON,
    LJ_ARGON_LARGER,
    MIXTURE_MODEL,
    WELL,
    WELL_MODEL,
)


def configuration(position, species=(ARGON,), species_index=None, box=30.0):
    """A Configuration from positions in Angstrom, argon unless told otherwise."""
    if species_index is None:
        species_index = np.zeros(len(position), dtype=np.int64)
    return Configuration(position, species, species_index, box)


def three_atoms(species_index=(0, 0, 0)):
    """Three atoms at (1, 0), (5, 0) and (0, 5) Angstrom in a 30 Angstrom box."""
    return configuration(
        [[1.0, 0.0], [5.0, 0.0], [0.0, 5.0]],
        species=(ARGON, LARGER),
        species_index=species_index,
    )


class TestConfiguration(unittest.TestCase):
    def test_compares_by_identity_not_by_array_contents(self):
        # numpy equality gives an array, not a truth value, so configurations
        # and pair data are equal only when they are the same object.
        c = three_atoms()
        self.assertEqual(c, c)
        self.assertNotEqual(c, c.copy())
        self.assertNotEqual(
            c.pairs(ARGON_MODEL, 15.0),
            c.pairs(ARGON_MODEL, 15.0),
        )

    def test_holds_positions_species_and_box(self):
        c = three_atoms([0, 1, 0])
        self.assertEqual(c.number_of_atoms, 3)
        assert_allclose(c.masses, [39.948, 80.0, 39.948])
        self.assertEqual(c.box, 30.0)

    def test_rejects_positions_that_are_not_n_by_2(self):
        with self.assertRaisesRegex(ValueError, r"\(N, 2\)"):
            configuration(np.zeros((3, 3)))

    def test_rejects_a_species_index_of_the_wrong_length_or_type(self):
        with self.assertRaisesRegex(ValueError, "species_index"):
            configuration(np.zeros((3, 2)), species_index=np.zeros(2, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "species_index"):
            configuration(np.zeros((3, 2)), species_index=np.zeros(3))

    def test_rejects_an_index_that_names_no_species(self):
        with self.assertRaisesRegex(ValueError, "index the 1 species"):
            configuration(np.zeros((2, 2)), species_index=np.array([0, 1]))

    def test_rejects_no_species(self):
        with self.assertRaisesRegex(ValueError, "at least one Species"):
            configuration(np.zeros((0, 2)), species=())

    def test_rejects_a_bad_box(self):
        for bad in (0.0, -1.0, np.inf):
            with self.assertRaisesRegex(ValueError, "box must be positive"):
                configuration(np.zeros((2, 2)), box=bad)

    def test_construction_copies_the_arrays(self):
        positions = np.array([[1.0, 0.0], [5.0, 0.0]])
        species_index = np.array([0, 1])
        c = configuration(positions, species=(ARGON, LARGER), species_index=species_index)
        positions[0] = 9.0
        species_index[0] = 1
        assert_equal(c.positions[0], [1.0, 0.0])
        assert_equal(c.species_index, [0, 1])

    def test_arrays_and_attributes_can_be_changed(self):
        c = three_atoms()
        c.positions[0] += 1.0
        c.box = 40.0
        assert_equal(c.positions[0], [2.0, 1.0])
        self.assertEqual(c.box, 40.0)

    def test_a_misspelt_attribute_raises(self):
        c = three_atoms()
        with self.assertRaises(AttributeError):
            c.postions = np.zeros((3, 2))  # type: ignore[attr-defined]

    def test_copy_is_independent(self):
        c = three_atoms()
        copied = c.copy()
        copied.positions[0] = 9.0
        copied.box = 40.0
        assert_equal(c.positions[0], [1.0, 0.0])
        self.assertEqual(c.box, 30.0)
        assert_equal(copied.species_index, c.species_index)
        self.assertEqual(copied.species, c.species)

    def test_accepts_lists(self):
        c = Configuration([[1, 0], [5, 0]], [ARGON], [0, 0], 30)
        self.assertEqual(c.positions.dtype, np.float64)
        self.assertEqual(c.species, (ARGON,))
        assert_equal(c.species_index, [0, 0])
        self.assertEqual(c.potential_energy(ARGON_MODEL, 15.0), LJ_ARGON.energies(4.0))

    def test_without_drops_one_atom_from_every_array(self):
        c = three_atoms([0, 1, 0])
        rest = c.without(1)
        self.assertEqual(rest.number_of_atoms, 2)
        assert_almost_equal(rest.positions, [[1.0, 0.0], [0.0, 5.0]])
        assert_equal(rest.species_index, [0, 0])

    def test_pairs_evaluates_each_pair_on_its_own_potential_and_applies_the_cut_off(self):
        # Pairs (0, 1) at 4, (0, 2) at 5.1 and (1, 2) at 7.07 Angstrom are
        # argon-larger, argon-argon and larger-argon; a 6 Angstrom cut-off
        # drops the last.
        c = three_atoms([0, 1, 0])
        pairs = c.pairs(MIXTURE_MODEL, 6.0, forces=True)
        assert_almost_equal(pairs.distances, [4.0, np.sqrt(26), np.sqrt(50)])
        expected = [
            LJ_ARGON_LARGER.energies(pairs.distances[0]),
            LJ_ARGON.energies(pairs.distances[1]),
            0.0,
        ]
        assert_allclose(pairs.energies, expected, rtol=1e-12)
        assert pairs.radial_forces is not None
        assert_allclose(
            pairs.radial_forces[0], LJ_ARGON_LARGER.forces(pairs.distances[0]), rtol=1e-12
        )
        self.assertEqual(pairs.radial_forces[2], 0.0)

    def test_pairs_needs_no_force_from_the_potential(self):
        c = three_atoms()
        pairs = c.pairs(WELL_MODEL, 15.0)
        assert_almost_equal(pairs.energies, [-WELL.epsilon, 0.0, 0.0])
        self.assertIsNone(pairs.radial_forces)
        with self.assertRaisesRegex(ValueError, "Monte Carlo"):
            c.pairs(WELL_MODEL, 15.0, forces=True)

    def test_potential_energy_is_the_sum_of_the_pair_energies(self):
        # Pairs at 4, sqrt(26) and sqrt(50) Angstrom, all argon.
        c = three_atoms()
        expected = LJ_ARGON.energies(np.array([4.0, np.sqrt(26), np.sqrt(50)]))
        assert_allclose(c.potential_energy(ARGON_MODEL, 15.0), expected.sum(), rtol=1e-12)

    def test_insertion_energy_is_the_energy_the_atom_adds(self):
        # The same minimum image, species lookup and cut-off on both paths:
        # inserting an atom into a mixture raises the total pair energy
        # by exactly its insertion energy, at a cut-off that drops pairs.
        rng = np.random.default_rng(1)
        n, box, cut_off = 8, 30.0, 9.0
        species_index = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        full = configuration(
            rng.uniform(0, box, (n, 2)),
            species=MIXTURE_MODEL.species,
            species_index=species_index,
            box=box,
        )
        without_last = full.without(n - 1)
        added = without_last.insertion_energy(
            full.positions[-1], int(species_index[-1]), MIXTURE_MODEL, cut_off
        )
        difference = full.potential_energy(MIXTURE_MODEL, cut_off) - without_last.potential_energy(
            MIXTURE_MODEL, cut_off
        )
        assert_allclose(added, difference, rtol=1e-9)

    def test_forces_and_virial_match_a_reference_loop(self):
        # An independent double loop over pairs is the oracle for the net
        # force on each atom and the virial. Full-box positions exercise
        # the minimum image; the cut-off is large so no pair is zeroed.
        rng = np.random.default_rng(0)
        n, box = 8, 30.0
        species_index = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        c = configuration(
            rng.uniform(0, box, (n, 2)),
            species=MIXTURE_MODEL.species,
            species_index=species_index,
            box=box,
        )
        reference = np.zeros((n, 2))
        virial = 0.0
        for a in range(n - 1):
            for b in range(a + 1, n):
                separation = c.positions[a] - c.positions[b]
                separation -= box * np.round(separation / box)
                dr = np.linalg.norm(separation)
                potential = MIXTURE_MODEL.potential(
                    c.species[species_index[a]], c.species[species_index[b]]
                )
                force = potential.forces(dr)
                reference[a] += force * separation / dr
                reference[b] -= force * separation / dr
                virial += force * dr
        assert_allclose(c.forces(MIXTURE_MODEL, 100.0), reference, rtol=1e-12)
        assert_allclose(c.virial(MIXTURE_MODEL, 100.0), virial, rtol=1e-12)

    def test_forces_are_equal_and_opposite_for_a_pair(self):
        c = configuration([[0.0, 0.0], [4.0, 0.0]])
        force = c.forces(ARGON_MODEL, 15.0)
        assert_allclose(force[0], -force[1])
        self.assertNotEqual(force[0, 0], 0.0)
        self.assertEqual(force[0, 1], 0.0)

    def test_insertion_energy_sums_the_pairs_with_the_neighbours_species(self):
        # An argon atom at the origin with a larger neighbour at 4
        # Angstrom and an argon one at 5: the cross potential for the first,
        # the self potential for the second. A third, at y = 20 Angstrom in
        # the 30 Angstrom box, is 10 Angstrom away and beyond the cut-off.
        c = configuration(
            [[4.0, 0.0], [0.0, 5.0], [0.0, 20.0]],
            species=(ARGON, LARGER),
            species_index=[1, 0, 0],
        )
        energy = c.insertion_energy((0.0, 0.0), 0, MIXTURE_MODEL, 6.0)
        expected = (
            LJ_ARGON_LARGER.energies(np.array([4.0]))[0] + LJ_ARGON.energies(np.array([5.0]))[0]
        )
        assert_allclose(energy, expected, rtol=1e-12)

    def test_insertion_energy_drops_pairs_beyond_the_cut_off_and_uses_the_minimum_image(self):
        # A neighbour at x = 29 Angstrom in a 30 Angstrom box is 1 Angstrom
        # away across the boundary, and one at 8 Angstrom is beyond a 6
        # Angstrom cut-off.
        c = configuration([[29.0, 0.0], [8.0, 0.0]])
        energy = c.insertion_energy((0.0, 0.0), 0, ARGON_MODEL, 6.0)
        assert_allclose(energy, LJ_ARGON.energies(np.array([1.0]))[0], rtol=1e-12)

    def test_pairs_forbid_a_separation_where_the_potential_is_unphysical(self):
        # Two argon 0.5 Angstrom apart, inside the Buckingham barrier: the
        # formula there is a deep negative number, but the pair is forbidden,
        # so the energy is infinite and asking for the force raises.
        c = configuration([[0.0, 0.0], [0.5, 0.0]])
        self.assertLess(BUCKINGHAM_ARGON.energies(np.array([0.5]))[0], 0.0)
        self.assertEqual(c.pairs(BUCKINGHAM_MODEL, 15.0).energies[0], np.inf)
        self.assertEqual(c.potential_energy(BUCKINGHAM_MODEL, 15.0), np.inf)
        with self.assertRaisesRegex(ValueError, "unphysical"):
            c.forces(BUCKINGHAM_MODEL, 15.0)
        with self.assertRaisesRegex(ValueError, "collapsed"):
            c.virial(BUCKINGHAM_MODEL, 15.0)

    def test_insertion_energy_forbids_a_separation_where_the_potential_is_unphysical(self):
        c = configuration([[0.0, 0.0]])
        energy = c.insertion_energy((0.5, 0.0), 0, BUCKINGHAM_MODEL, 15.0)
        self.assertEqual(energy, np.inf)

    def test_insertion_energy_with_no_atoms_is_zero(self):
        empty = configuration(np.zeros((0, 2)))
        self.assertEqual(
            empty.insertion_energy((1.0, 1.0), 0, ARGON_MODEL, 15.0),
            0.0,
        )

    def test_a_changed_species_index_is_evaluated(self):
        # Evaluating first builds the pairs grouped by species, which must
        # be rebuilt when species_index changes.
        c = three_atoms([0, 1, 0])
        c.pairs(MIXTURE_MODEL, 15.0)
        c.species_index = np.array([1, 0, 0])
        assert_allclose(
            c.pairs(MIXTURE_MODEL, 15.0).energies,
            three_atoms([1, 0, 0]).pairs(MIXTURE_MODEL, 15.0).energies,
        )

    def test_pairs_separations_point_from_the_second_atom_to_the_first(self):
        # r_i - r_j for the pair (0, 1).
        c = configuration([[1.0, 0.0], [5.0, 0.0]])
        assert_allclose(c.pairs(ARGON_MODEL, 15.0).separations, [[-4.0, 0.0]])

    def test_pairs_evaluates_each_potential_only_on_its_own_pairs(self):
        # GaussianCore is finite and non-zero at zero separation, so this
        # would fail if a potential were handed distances belonging to other
        # species pairs, zeroed out.
        class GaussianCore(PairPotential):
            def __init__(self, *, a, b):
                self.a = a
                self.b = b

            def energies(self, dr):
                dr = np.asarray(dr, dtype=float)
                return self.a * np.exp(-((dr / self.b) ** 2))

            def forces(self, dr):
                dr = np.asarray(dr, dtype=float)
                return 2 * self.a * dr / self.b**2 * np.exp(-((dr / self.b) ** 2))

        potentials = {
            (ARGON, ARGON): GaussianCore(a=1.0, b=2.0),
            (LARGER, LARGER): GaussianCore(a=3.0, b=4.0),
            (ARGON, LARGER): GaussianCore(a=2.0, b=3.0),
        }
        c = three_atoms([0, 1, 0])
        pairs = c.pairs(Model((ARGON, LARGER), potentials), 15.0, forces=True)
        # pairs (0, 1), (0, 2), (1, 2) are argon-larger, argon-argon, larger-argon
        kinds = [(ARGON, LARGER), (ARGON, ARGON), (ARGON, LARGER)]
        by_pair = list(zip(pairs.distances, kinds, strict=True))
        expected_energy = [potentials[k].energies(d) for d, k in by_pair]
        expected_force = [potentials[k].forces(d) for d, k in by_pair]
        assert_almost_equal(pairs.energies, expected_energy)
        assert_almost_equal(pairs.radial_forces, expected_force)


def two_atoms(distance, box=20.0):
    """Two argon atoms the given distance apart along x, in Angstrom."""
    return configuration([[1.0, 1.0], [1.0 + distance, 1.0]], box=box)


class TestRDF(unittest.TestCase):
    def test_two_atoms_fill_one_bin_near_their_distance(self):
        c = two_atoms(4.0)
        r, gr = c.rdf(bins=50)
        dr = c.box / 2 / 50
        self.assertEqual(r.size, 50)
        assert_allclose(r, np.arange(50) * dr + dr / 2)
        self.assertEqual(np.count_nonzero(gr), 1)
        (i,) = np.nonzero(gr)
        self.assertLess(abs(r[i] - 4.0), dr)
        # One pair: the bin's height is the box area over the area of the
        # bin's ring, 2 pi r dr.
        assert_allclose(gr[i], c.box**2 / (2 * np.pi * r[i] * dr))

    def test_r_max_sets_the_range(self):
        r, gr = two_atoms(4.0).rdf(bins=10, r_max=5.0)
        assert_allclose(r[-1], 5.0 - 0.25)

    def test_single_atom_gives_zeros(self):
        c = configuration([[1.0, 1.0]], box=20.0)
        r, gr = c.rdf(bins=10)
        self.assertEqual(r.size, 10)
        assert_allclose(gr, 0.0)

    def test_uniform_gas_gives_one(self):
        rng = np.random.default_rng(0)
        n = 3000
        c = Configuration(
            positions=rng.uniform(0, 20.0, size=(n, 2)),
            species=(ARGON,),
            species_index=np.zeros(n, dtype=int),
            box=20.0,
        )
        _, gr = c.rdf(bins=10)
        assert_allclose(gr, 1.0, atol=0.03)

    def test_pairs_are_measured_across_the_periodic_boundary(self):
        c = two_atoms(18.0)
        r, gr = c.rdf(bins=50)
        (i,) = np.nonzero(gr)
        self.assertLess(abs(r[i] - 2.0), c.box / 2 / 50)


class TestStructureFactor(unittest.TestCase):
    def test_two_atoms_match_the_shell_average_by_hand(self):
        box = 20.0
        c = configuration(np.array([[0.0, 0.0], [4.0, 0.0]]), box=box)
        q, s = c.structure_factor(q_max=3 * 2 * np.pi / box)
        wavevector = 2 * np.pi / box * np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        expected = np.mean(
            [abs(1 + np.exp(1j * np.dot(w, [4.0, 0.0]))) ** 2 / 2 for w in wavevector]
        )
        assert_allclose(s[0], expected)

    def test_is_never_negative(self):
        rng = np.random.default_rng(0)
        c = configuration(rng.uniform(0, 30.0, size=(50, 2)))
        _, s = c.structure_factor()
        self.assertTrue((s >= 0).all())

    def test_a_square_lattice_peaks_at_its_reciprocal_lattice(self):
        # A hundred atoms on a square lattice in a 40 Angstrom box sit 4
        # Angstrom apart, so every peak falls at 2 pi / 4 Angstrom times the
        # square root of a whole number.
        spacing = 4.0
        c = placement.place_square(100, (ARGON,), 40.0)
        q, s = c.structure_factor()
        peaks = q[s > 0.5 * s.max()]
        self.assertGreater(peaks.size, 5)
        whole = (peaks / (2 * np.pi / spacing)) ** 2
        assert_allclose(whole, np.rint(whole), atol=1e-9)

    def test_uncorrelated_atoms_give_one(self):
        rng = np.random.default_rng(0)
        c = configuration(rng.uniform(0, 40.0, size=(400, 2)), box=40.0)
        _, s = c.structure_factor()
        assert_allclose(s.mean(), 1.0, atol=0.05)

    def test_q_max_sets_the_largest_magnitude(self):
        box = 20.0
        q_max = 4 * 2 * np.pi / box
        q, _ = configuration(np.array([[0.0, 0.0], [4.0, 0.0]]), box=box).structure_factor(q_max)
        self.assertLessEqual(q.max(), q_max)
        self.assertGreater(q.max(), 0.9 * q_max)

    def test_the_default_range_follows_the_density(self):
        sparse = placement.place_square(25, (ARGON,), 40.0)
        dense = placement.place_square(100, (ARGON,), 40.0)
        q_sparse, _ = sparse.structure_factor()
        q_dense, _ = dense.structure_factor()
        self.assertLessEqual(q_sparse.max(), default_q_max(25, 40.0))
        self.assertLessEqual(q_dense.max(), default_q_max(100, 40.0))
        # Four times the atoms in the same box doubles the range.
        assert_allclose(q_dense.max() / q_sparse.max(), 2.0, rtol=0.02)

    def test_refuses_a_q_max_below_the_box(self):
        c = placement.place_square(4, (ARGON,), 40.0)
        with self.assertRaisesRegex(ValueError, "smallest wavevector"):
            c.structure_factor(q_max=0.1)


def md_configuration(position, velocity, box=8.0):
    """An argon MDConfiguration whose unwrapped positions start at the positions."""
    return MDConfiguration(
        positions=position,
        species=(ARGON,),
        species_index=np.zeros(len(position), dtype=np.int64),
        box=box,
        velocities=velocity,
        unwrapped=position,
    )


class TestMDConfiguration(unittest.TestCase):
    def test_rejects_velocities_or_unwrapped_positions_of_the_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "velocities"):
            md_configuration(np.zeros((2, 2)), np.zeros((3, 2)))
        c = md_configuration(np.zeros((2, 2)), np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "unwrapped"):
            c.replace(unwrapped=np.zeros((3, 2)))

    def test_is_immutable(self):
        c = md_configuration([[2.0, 2.0], [2.0, 6.0]], [[1.0, 0.0], [-1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "read-only"):
            c.velocities *= 2
        with self.assertRaisesRegex(ValueError, "read-only"):
            c.unwrapped[0] += 1.0
        with self.assertRaises(AttributeError):
            c.velocities = c.velocities * 2  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            c.temperature = 150  # type: ignore[method-assign, assignment]

    def test_keeps_its_own_copy_of_the_arrays(self):
        position = np.array([[2.0, 2.0], [2.0, 6.0]])
        velocity = np.array([[1.0, 0.0], [-1.0, 0.0]])
        c = md_configuration(position, velocity)
        position[0] = 9.0
        velocity[0] = 9.0
        assert_equal(c.unwrapped[0], [2.0, 2.0])
        assert_equal(c.velocities[0], [1.0, 0.0])

    def test_stays_immutable_when_pickled_or_deep_copied(self):
        c = md_configuration([[2.0, 2.0], [2.0, 6.0]], [[1.0, 0.0], [-1.0, 0.0]])
        for copied in (pickle.loads(pickle.dumps(c)), copy.deepcopy(c)):
            with self.assertRaisesRegex(ValueError, "read-only"):
                copied.positions[0] += 1.0
            with self.assertRaisesRegex(ValueError, "read-only"):
                copied.velocities *= 2

    def test_without_drops_one_atom_from_every_array(self):
        c = md_configuration(
            [[2.0, 2.0], [2.0, 6.0], [4.0, 4.0]], [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]
        )
        c = c.replace(
            species=(ARGON, LARGER),
            species_index=[0, 1, 0],
            unwrapped=c.positions + [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
        )
        rest = c.without(1)
        assert_equal(rest.positions, [[2.0, 2.0], [4.0, 4.0]])
        assert_equal(rest.species_index, [0, 0])
        assert_equal(rest.velocities, [[1.0, 0.0], [-1.0, -1.0]])
        assert_equal(rest.unwrapped, [[12.0, 2.0], [34.0, 4.0]])

    def test_kinetic_energy_and_temperature(self):
        # Two atoms with equal and opposite velocities of 1 Angstrom/ps in x
        # and y: kinetic energy m (vx^2 + vy^2) in amu Angstrom^2/ps^2,
        # divided by KJ_PER_MOL for kJ/mol, then by (N - 1) k_B with N - 1 = 1.
        c = md_configuration([[2.0, 2.0], [2.0, 6.0]], [[1.0, 1.0], [-1.0, -1.0]])
        expected = 39.948 * 2 / KJ_PER_MOL
        assert_allclose(c.kinetic_energy(), expected)
        assert_allclose(c.temperature(), expected / BOLTZMANN)

    def test_temperature_is_undefined_for_one_atom(self):
        # Two atoms at rest are 0 K; one atom is 0/0.
        self.assertEqual(
            md_configuration([[2.0, 2.0], [2.0, 6.0]], np.zeros((2, 2))).temperature(),
            0.0,
        )
        c = md_configuration([[2.0, 2.0]], [[1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "undefined for a single atom"):
            c.temperature()
