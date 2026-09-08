import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from pylj import pairwise
from pylj.configuration import Configuration, MDConfiguration
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN
from pylj.potentials import PairPotential
from pylj.tests.argon import (
    ARGON,
    ARGON_MODEL,
    BUCKINGHAM_ARGON,
    BUCKINGHAM_MODEL,
    LARGER,
    LJ_ARGON,
    LJ_ARGON_LARGER,
    MIXTURE_MODEL,
    WELL_MODEL,
)


def configuration(position, species=(ARGON,), species_index=None, box=30e-10):
    """A Configuration from positions in metres, argon unless told otherwise."""
    position = np.asarray(position, dtype=float)
    if species_index is None:
        species_index = np.zeros(position.shape[0], dtype=np.int64)
    return Configuration(position, tuple(species), np.asarray(species_index), box)


def three_atoms(species_index=(0, 0, 0)):
    """Three atoms at (1, 0), (5, 0) and (0, 5) Angstrom in a 30 Angstrom box."""
    return configuration(
        [[1e-10, 0.0], [5e-10, 0.0], [0.0, 5e-10]],
        species=(ARGON, LARGER),
        species_index=species_index,
    )


class TestConfiguration(unittest.TestCase):
    def test_compares_by_identity_not_by_array_contents(self):
        # A dataclass with array fields cannot compare field by field: numpy
        # equality gives an array, not a truth value. Two configurations are
        # equal only when they are the same object.
        c = three_atoms()
        self.assertEqual(c, c)
        self.assertNotEqual(c, c.replace())
        self.assertNotEqual(
            c.pairs(ARGON_MODEL["pair_potentials"], 15e-10),
            c.pairs(ARGON_MODEL["pair_potentials"], 15e-10),
        )

    def test_holds_positions_species_and_box(self):
        c = three_atoms([0, 1, 0])
        self.assertEqual(c.number_of_atoms, 3)
        assert_allclose(c.masses, np.array([39.948, 80.0, 39.948]) * ATOMIC_MASS_UNIT)
        self.assertEqual(c.box, 30e-10)

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
        for bad in (0.0, -1e-9, np.inf):
            with self.assertRaisesRegex(ValueError, "box must be positive"):
                configuration(np.zeros((2, 2)), box=bad)

    def test_is_immutable(self):
        c = three_atoms()
        with self.assertRaises(AttributeError):
            c.box = 1.0  # type: ignore[misc]

    def test_replace_gives_a_validated_copy(self):
        c = three_atoms()
        moved = c.replace(position=c.position + 1e-10)
        assert_almost_equal(moved.position, c.position + 1e-10)
        assert_almost_equal(c.position[0], [1e-10, 0.0])
        with self.assertRaisesRegex(ValueError, r"\(N, 2\)"):
            c.replace(position=np.zeros(3))

    def test_without_drops_one_atom_from_every_array(self):
        c = three_atoms([0, 1, 0])
        rest = c.without(1)
        self.assertEqual(rest.number_of_atoms, 2)
        assert_almost_equal(rest.position, [[1e-10, 0.0], [0.0, 5e-10]])
        assert_equal(rest.species_index, [0, 0])

    def test_pairs_evaluates_each_pair_on_its_own_potential_and_applies_the_cut_off(self):
        # Pairs (0, 1) at 4, (0, 2) at 5.1 and (1, 2) at 7.07 Angstrom are
        # argon-larger, argon-argon and larger-argon; a 6 Angstrom cut-off
        # drops the last.
        c = three_atoms([0, 1, 0])
        pairs = c.pairs(MIXTURE_MODEL["pair_potentials"], 6e-10, forces=True)
        assert_almost_equal(pairs.distance * 1e10, [4.0, np.sqrt(26), np.sqrt(50)])
        expected = [
            LJ_ARGON_LARGER.energies(pairs.distance[0]),
            LJ_ARGON.energies(pairs.distance[1]),
            0.0,
        ]
        assert_allclose(pairs.energy, expected, rtol=1e-12)
        assert pairs.radial_force is not None
        assert_allclose(
            pairs.radial_force[0], LJ_ARGON_LARGER.forces(pairs.distance[0]), rtol=1e-12
        )
        self.assertEqual(pairs.radial_force[2], 0.0)

    def test_pairs_needs_no_force_from_the_potential(self):
        c = three_atoms()
        pairs = c.pairs(WELL_MODEL["pair_potentials"], 15e-10)
        assert_almost_equal(pairs.energy * 1e21, [-1.5, 0.0, 0.0])
        self.assertIsNone(pairs.radial_force)
        with self.assertRaisesRegex(ValueError, "Monte Carlo"):
            c.pairs(WELL_MODEL["pair_potentials"], 15e-10, forces=True)

    def test_potential_energy_is_the_sum_of_the_pair_energies(self):
        # Pairs at 4, sqrt(26) and sqrt(50) Angstrom, all argon.
        c = three_atoms()
        expected = LJ_ARGON.energies(np.array([4e-10, np.sqrt(26) * 1e-10, np.sqrt(50) * 1e-10]))
        assert_allclose(
            c.potential_energy(ARGON_MODEL["pair_potentials"], 15e-10), expected.sum(), rtol=1e-12
        )

    def test_insertion_energy_is_the_energy_the_atom_adds(self):
        # The same minimum image, species lookup and cut-off on both paths:
        # inserting an atom into a mixture raises the total pair energy
        # by exactly its insertion energy, at a cut-off that drops pairs.
        rng = np.random.default_rng(1)
        n, box, cut_off = 8, 30e-10, 9e-10
        species_index = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        potentials = MIXTURE_MODEL["pair_potentials"]
        full = configuration(
            rng.uniform(0, box, (n, 2)),
            species=MIXTURE_MODEL["species"],
            species_index=species_index,
            box=box,
        )
        without_last = full.without(n - 1)
        added = without_last.insertion_energy(
            full.position[-1], int(species_index[-1]), potentials, cut_off
        )
        difference = full.potential_energy(potentials, cut_off) - without_last.potential_energy(
            potentials, cut_off
        )
        assert_allclose(added, difference, rtol=1e-9)

    def test_forces_and_virial_match_a_reference_loop(self):
        # An independent double loop over pairs is the oracle for the net
        # force on each atom and the virial. Full-box positions exercise
        # the minimum image; the cut-off is large so no pair is zeroed.
        rng = np.random.default_rng(0)
        n, box = 8, 30e-10
        species_index = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        c = configuration(
            rng.uniform(0, box, (n, 2)),
            species=MIXTURE_MODEL["species"],
            species_index=species_index,
            box=box,
        )
        potentials = MIXTURE_MODEL["pair_potentials"]
        reference = np.zeros((n, 2))
        virial = 0.0
        for a in range(n - 1):
            for b in range(a + 1, n):
                separation = c.position[a] - c.position[b]
                separation -= box * np.round(separation / box)
                dr = np.linalg.norm(separation)
                potential = pairwise.pair_potential(
                    potentials, c.species[species_index[a]], c.species[species_index[b]]
                )
                force = potential.forces(dr)
                reference[a] += force * separation / dr
                reference[b] -= force * separation / dr
                virial += force * dr
        assert_allclose(c.forces(potentials, 1e-8), reference, rtol=1e-12)
        assert_allclose(c.virial(potentials, 1e-8), virial, rtol=1e-12)

    def test_forces_are_equal_and_opposite_for_a_pair(self):
        c = configuration([[0.0, 0.0], [4e-10, 0.0]])
        force = c.forces(ARGON_MODEL["pair_potentials"], 15e-10)
        assert_allclose(force[0], -force[1])
        self.assertNotEqual(force[0, 0], 0.0)
        self.assertEqual(force[0, 1], 0.0)

    def test_insertion_energy_sums_the_pairs_with_the_neighbours_species(self):
        # An argon atom at the origin with a larger neighbour at 4
        # Angstrom and an argon one at 5: the cross potential for the first,
        # the self potential for the second. A third, at y = 20 Angstrom in
        # the 30 Angstrom box, is 10 Angstrom away and beyond the cut-off.
        c = three_atoms([1, 0, 0]).replace(
            position=np.array([[4e-10, 0.0], [0.0, 5e-10], [0.0, 20e-10]])
        )
        energy = c.insertion_energy((0.0, 0.0), 0, MIXTURE_MODEL["pair_potentials"], 6e-10)
        expected = (
            LJ_ARGON_LARGER.energies(np.array([4e-10]))[0] + LJ_ARGON.energies(np.array([5e-10]))[0]
        )
        assert_allclose(energy, expected, rtol=1e-12)

    def test_insertion_energy_drops_pairs_beyond_the_cut_off_and_uses_the_minimum_image(self):
        # A neighbour at x = 29 Angstrom in a 30 Angstrom box is 1 Angstrom
        # away across the boundary, and one at 8 Angstrom is beyond a 6
        # Angstrom cut-off.
        c = configuration([[29e-10, 0.0], [8e-10, 0.0]])
        energy = c.insertion_energy((0.0, 0.0), 0, ARGON_MODEL["pair_potentials"], 6e-10)
        assert_allclose(energy, LJ_ARGON.energies(np.array([1e-10]))[0], rtol=1e-12)

    def test_pairs_forbid_a_separation_where_the_potential_is_not_valid(self):
        # Two argon 0.5 Angstrom apart, inside the Buckingham barrier: the
        # formula there is a deep negative number, but the pair is forbidden,
        # so the energy is infinite and asking for the force raises.
        potentials = BUCKINGHAM_MODEL["pair_potentials"]
        c = configuration([[0.0, 0.0], [0.5e-10, 0.0]])
        self.assertLess(BUCKINGHAM_ARGON.energies(np.array([0.5e-10]))[0], 0.0)
        self.assertEqual(c.pairs(potentials, 15e-10).energy[0], np.inf)
        self.assertEqual(c.potential_energy(potentials, 15e-10), np.inf)
        with self.assertRaisesRegex(ValueError, "not valid"):
            c.forces(potentials, 15e-10)
        with self.assertRaisesRegex(ValueError, "collapsed"):
            c.virial(potentials, 15e-10)

    def test_insertion_energy_forbids_a_separation_where_the_potential_is_not_valid(self):
        c = configuration([[0.0, 0.0]])
        energy = c.insertion_energy((0.5e-10, 0.0), 0, BUCKINGHAM_MODEL["pair_potentials"], 15e-10)
        self.assertEqual(energy, np.inf)

    def test_insertion_energy_with_no_atoms_is_zero(self):
        empty = configuration(np.zeros((0, 2)))
        self.assertEqual(
            empty.insertion_energy((1e-10, 1e-10), 0, ARGON_MODEL["pair_potentials"], 15e-10),
            0.0,
        )

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
        pairs = c.pairs(potentials, 15e-10, forces=True)
        # pairs (0, 1), (0, 2), (1, 2) are argon-larger, argon-argon, larger-argon
        kinds = [(ARGON, LARGER), (ARGON, ARGON), (ARGON, LARGER)]
        by_pair = list(zip(pairs.distance, kinds, strict=True))
        expected_energy = [potentials[k].energies(d) for d, k in by_pair]
        expected_force = [potentials[k].forces(d) for d, k in by_pair]
        assert_almost_equal(pairs.energy, expected_energy)
        assert_almost_equal(pairs.radial_force, expected_force)


def md_configuration(position, velocity, box=8e-10):
    """An argon MDConfiguration whose unwrapped positions start at the positions."""
    position = np.asarray(position, dtype=float)
    return MDConfiguration(
        position=position,
        species=(ARGON,),
        species_index=np.zeros(position.shape[0], dtype=np.int64),
        box=box,
        velocity=np.asarray(velocity, dtype=float),
        unwrapped=position.copy(),
    )


class TestMDConfiguration(unittest.TestCase):
    def test_rejects_velocities_of_the_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "velocity"):
            md_configuration(np.zeros((2, 2)), np.zeros((3, 2)))

    def test_is_a_configuration(self):
        c = md_configuration([[2e-10, 2e-10], [2e-10, 6e-10]], np.zeros((2, 2)))
        self.assertIsInstance(c, Configuration)
        self.assertEqual(c.without(0).number_of_atoms, 1)
        assert_equal(c.without(0).velocity.shape, (1, 2))

    def test_kinetic_energy_and_temperature(self):
        # Two atoms with equal and opposite velocities of 1e-10 m/s in x
        # and y: kinetic energy m (vx^2 + vy^2), divided by (N - 1) k_B with
        # N - 1 = 1 for the two atoms.
        c = md_configuration([[2e-10, 2e-10], [2e-10, 6e-10]], [[1e-10, 1e-10], [-1e-10, -1e-10]])
        expected = 39.948 * ATOMIC_MASS_UNIT * 2e-20
        assert_allclose(c.kinetic_energy(), expected)
        assert_allclose(c.temperature(), expected / BOLTZMANN)

    def test_temperature_needs_two_atoms(self):
        c = md_configuration([[2e-10, 2e-10]], [[1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "at least two atoms"):
            c.temperature()

    def test_msd_uses_the_unwrapped_positions(self):
        # Displacements of (1, 1) and (5, 1) Angstrom give (2 + 26) / 2 = 14
        # Angstrom^2, from unwrapped positions that have left the box.
        start = md_configuration([[2e-10, 2e-10], [2e-10, 6e-10]], np.zeros((2, 2)))
        displacement = np.array([[1e-10, 1e-10], [5e-10, 1e-10]])
        moved = start.replace(unwrapped=start.unwrapped + displacement)
        assert_almost_equal(moved.msd(start) * 1e20, 14)
        self.assertEqual(start.msd(start), 0.0)
