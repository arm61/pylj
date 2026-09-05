import unittest
import warnings

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from pylj import pairwise, util
from pylj.constants import ATOMIC_MASS_UNIT
from pylj.potentials import PairPotential, Species
from pylj.tests.argon import (
    ARGON,
    ARGON_MODEL,
    LARGER,
    LJ_ARGON,
    LJ_ARGON_LARGER,
    MIXTURE_MODEL,
    WELL,
)


class gaussian_core(PairPotential):
    r"""A pair potential that is finite and non-zero at zero separation,
    used to check that compute_force never evaluates a potential on a
    pair of a different species.

    .. math::
        E = a e^{-(r / b)^{2}}

        f = \frac{2 a r}{b^{2}} e^{-(r / b)^{2}}
    """

    def __init__(self, *, a, b):
        self.a = a
        self.b = b

    def energies(self, dr):
        dr = np.asarray(dr, dtype=float)
        return self.a * np.exp(-((dr / self.b) ** 2))

    def forces(self, dr):
        dr = np.asarray(dr, dtype=float)
        return 2 * self.a * dr / self.b**2 * np.exp(-((dr / self.b) ** 2))


def three_particles(types):
    """Three particles at (1, 0), (5, 0) and (0, 5) Angstrom."""
    particles = np.zeros(3, dtype=util.particle_dt())
    particles["xposition"] = [1e-10, 5e-10, 0.0]
    particles["yposition"] = [0.0, 0.0, 5e-10]
    particles["types"] = types
    return particles


class TestPairwise(unittest.TestCase):
    def test_update_accelerations(self):
        # Three particles, so particle 0 receives two contributions and the
        # per-pair contributions must be accumulated, not assigned; each
        # particle is accelerated by its own mass.
        particles = np.zeros(3, dtype=util.particle_dt())
        # pairs (0, 1), (0, 2), (1, 2)
        forces = np.array([1.0, 2.0, 3.0])
        dx = np.array([1.0, 0.0, 1.0])
        dy = np.array([0.0, 1.0, 0.0])
        dr = np.array([1.0, 1.0, 1.0])
        masses = np.array([1.0, 2.0, 4.0])
        particles = pairwise.update_accelerations(particles, forces, masses, dx, dy, dr)
        assert_almost_equal(particles["xacceleration"], [1.0, -0.5 + 1.5, -0.75])
        assert_almost_equal(particles["yacceleration"], [2.0, 0.0, -0.5])

    def test_calculate_pressure(self):
        # The virial sum(f r) / (2 L^2) plus the ideal term N k_B T / L^2.
        distances = np.array([4e-10])
        forces = np.array([-9.5864009e-12])
        p = pairwise.calculate_pressure(
            distances,
            forces,
            30e-10,
            2,
            300,
        )
        virial = np.sum(forces * distances) / (2 * (30e-10) ** 2)
        ideal = 2 * 1.380649e-23 * 300 / (30e-10) ** 2
        assert_almost_equal(p, virial + ideal)

    def test_calculate_pressure_ideal_gas_limit(self):
        # With no pair forces the virial vanishes and the two-dimensional
        # pressure is the ideal-gas value N k_B T / L^2. Hand-computed:
        # 50 * 1.380649e-23 * 200 / (25e-10)^2.
        box_length = 25e-10
        p = pairwise.calculate_pressure(np.zeros(3), np.zeros(3), box_length, 50, 200)
        expected = 50 * 1.380649e-23 * 200 / box_length**2
        assert_almost_equal(p * 1e3, expected * 1e3)

    def test_calculate_pressure_adds_one_virial_term(self):
        # A single repulsive pair (positive force) 4 Angstrom apart adds a
        # positive virial f * r / (2 L^2) on top of the ideal term.
        box_length = 20e-10
        force, separation = np.array([2e-12]), np.array([4e-10])
        p = pairwise.calculate_pressure(separation, force, box_length, 2, 100)
        ideal = 2 * 1.380649e-23 * 100 / box_length**2
        virial = 2e-12 * 4e-10 / (2 * box_length**2)
        assert_almost_equal(p, ideal + virial)

    def test_dist_applies_the_minimum_image(self):
        # Pairs 1 Angstrom apart across the periodic boundary of a 10 Angstrom
        # box: the minimum image is 1 Angstrom, not the 9 Angstrom raw
        # separation, on either axis and in either direction. Pairs are
        # (0, 1), (0, 2), (1, 2) and components are x_i - x_j.
        xposition = np.array([0.5e-10, 9.5e-10, 0.5e-10])
        yposition = np.array([9.5e-10, 9.5e-10, 0.5e-10])
        dr, dx, dy = pairwise.dist(xposition, yposition, 10e-10)
        assert_almost_equal(dr * 1e10, [1.0, 1.0, np.sqrt(2)])
        assert_almost_equal(dx * 1e10, [1.0, 0.0, -1.0])
        assert_almost_equal(dy * 1e10, [0.0, -1.0, -1.0])

    def test_particle_masses_come_from_each_particles_species(self):
        particles = three_particles([0, 1, 0])
        assert_almost_equal(
            pairwise.particle_masses(particles, [ARGON, LARGER]), [39.948, 80.0, 39.948]
        )

    def test_pair_potential_is_found_in_either_key_order(self):
        pair_potentials = {(LARGER, ARGON): LJ_ARGON_LARGER}
        self.assertIs(pairwise.pair_potential(pair_potentials, ARGON, LARGER), LJ_ARGON_LARGER)
        self.assertIs(pairwise.pair_potential(pair_potentials, LARGER, ARGON), LJ_ARGON_LARGER)

    def test_compute_force_zeroes_pairs_beyond_the_cut_off(self):
        # cut_off is compared against the pair distances directly, so it is in
        # the same units as the positions (metres here). At 6e-10 m it sits
        # between the (1, 2) pair at 5e-10 and the (0, 2) pair at 8e-10, so one
        # pair is inside the cut-off and one is outside.
        particles = np.zeros(3, dtype=util.particle_dt())
        particles["xposition"] = [0.0, 3e-10, 8e-10]
        particles, distances, forces, energies = pairwise.compute_force(
            particles, 30, 6e-10, ARGON_MODEL["pair_potentials"], ARGON_MODEL["species"]
        )
        assert_almost_equal(distances, [3e-10, 8e-10, 5e-10])
        # The pair beyond the cut-off is zeroed; the two inside keep the values
        # the potential gives, so the mask is neither dropped nor inverted.
        self.assertEqual(energies[1], 0.0)
        self.assertEqual(forces[1], 0.0)
        assert_almost_equal(energies[[0, 2]] * 1e20, LJ_ARGON.energies(distances[[0, 2]]) * 1e20)
        assert_almost_equal(forces[[0, 2]] * 1e12, LJ_ARGON.forces(distances[[0, 2]]) * 1e12)

    def test_compute_force_does_not_warn_on_a_two_species_system(self):
        particles = three_particles([0, 1, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pairwise.compute_force(
                particles, 30, 15, MIXTURE_MODEL["pair_potentials"], MIXTURE_MODEL["species"]
            )

    def test_compute_force_evaluates_each_pair_on_its_own_distance(self):
        # gaussian_core is finite and non-zero at zero separation, so this
        # would fail if compute_force passed a potential distances belonging
        # to other species pairs, zeroed out.
        particles = three_particles([0, 1, 0])
        pair_potentials = {
            (ARGON, ARGON): gaussian_core(a=1.0, b=2.0),
            (LARGER, LARGER): gaussian_core(a=3.0, b=4.0),
            (ARGON, LARGER): gaussian_core(a=2.0, b=3.0),
        }
        particles, distances, forces, energies = pairwise.compute_force(
            particles, 30, 15, pair_potentials, [ARGON, LARGER]
        )
        # pairs (0, 1), (0, 2), (1, 2) are argon-larger, argon-argon, larger-argon
        pairs = [(ARGON, LARGER), (ARGON, ARGON), (ARGON, LARGER)]
        expected_energies = [
            pair_potentials[pair].energies(d) for d, pair in zip(distances, pairs, strict=True)
        ]
        expected_forces = [
            pair_potentials[pair].forces(d) for d, pair in zip(distances, pairs, strict=True)
        ]
        assert_almost_equal(energies, expected_energies)
        assert_almost_equal(forces, expected_forces)

    def test_compute_force_divides_each_pair_force_by_the_particles_own_mass(self):
        light = Species(mass=1.0, name="light")
        heavy = Species(mass=3.0, name="heavy")
        particles = np.zeros(2, dtype=util.particle_dt())
        particles["xposition"] = [0.0, 4e-10]
        particles["types"] = [0, 1]
        particles, _, _, _ = pairwise.compute_force(
            particles, 30, 15, {(light, heavy): LJ_ARGON}, [light, heavy]
        )
        # Equal and opposite forces, so m_light a_light = -m_heavy a_heavy.
        np.testing.assert_allclose(
            1.0 * particles["xacceleration"][0], -3.0 * particles["xacceleration"][1], rtol=1e-12
        )
        self.assertNotEqual(particles["xacceleration"][0], 0.0)

    def test_compute_energy_matches_the_energies_of_compute_force(self):
        particles = three_particles([0, 1, 0])
        pair_potentials = MIXTURE_MODEL["pair_potentials"]
        species = MIXTURE_MODEL["species"]
        distances, energies = pairwise.compute_energy(
            particles, 30, 6e-10, pair_potentials, species
        )
        _, force_distances, _, force_energies = pairwise.compute_force(
            particles.copy(), 30, 6e-10, pair_potentials, species
        )
        assert_almost_equal(distances, force_distances)
        assert_almost_equal(energies * 1e20, force_energies * 1e20)
        # (1, 2) at 7.07 Angstrom is beyond the 6 Angstrom cut-off.
        self.assertEqual(energies[2], 0.0)
        self.assertNotEqual(energies[1], 0.0)

    def test_compute_energy_does_not_touch_the_accelerations(self):
        particles = three_particles([0, 0, 0])
        particles["xacceleration"] = 7.0
        pairwise.compute_energy(
            particles, 30, 15, ARGON_MODEL["pair_potentials"], ARGON_MODEL["species"]
        )
        assert_almost_equal(particles["xacceleration"], [7.0, 7.0, 7.0])

    def test_compute_energy_needs_no_force_from_the_potential(self):
        # The square well has no finite force, so it drives the energy path
        # only.
        particles = three_particles([0, 0, 0])
        _, energies = pairwise.compute_energy(particles, 30, 15, {(ARGON, ARGON): WELL}, [ARGON])
        # (0, 1) at 4 Angstrom is in the well; (0, 2) at 5.1 and (1, 2) at
        # 7.07 are beyond it.
        assert_almost_equal(energies * 1e21, [-1.5, 0.0, 0.0])
        with pytest.raises(ValueError, match="Monte Carlo"):
            pairwise.compute_force(particles, 30, 15, {(ARGON, ARGON): WELL}, [ARGON])

    def test_particle_energy_sums_the_pairs(self):
        # Neighbours 4 and 5 Angstrom from the origin, both argon.
        others = np.zeros(2, dtype=util.particle_dt())
        others["xposition"] = [4e-10, 0.0]
        others["yposition"] = [0.0, 5e-10]
        energy = pairwise.particle_energy(
            (0.0, 0.0), 0, others, 30, 15, ARGON_MODEL["pair_potentials"], ARGON_MODEL["species"]
        )
        expected = LJ_ARGON.energies(np.array([4e-10, 5e-10])).sum()
        np.testing.assert_allclose(energy, expected, rtol=1e-12)

    def test_particle_energy_uses_the_neighbours_species(self):
        # An argon particle with a larger neighbour at 4 Angstrom and an argon
        # one at 5: the cross potential for the first, the self potential for
        # the second.
        others = np.zeros(2, dtype=util.particle_dt())
        others["xposition"] = [4e-10, 0.0]
        others["yposition"] = [0.0, 5e-10]
        others["types"] = [1, 0]
        energy = pairwise.particle_energy(
            (0.0, 0.0), 0, others, 30, 15,
            MIXTURE_MODEL["pair_potentials"], MIXTURE_MODEL["species"],
        )
        expected = (
            LJ_ARGON_LARGER.energies(np.array([4e-10]))[0]
            + LJ_ARGON.energies(np.array([5e-10]))[0]
        )
        np.testing.assert_allclose(energy, expected, rtol=1e-12)

    def test_particle_energy_drops_pairs_beyond_the_cut_off(self):
        others = np.zeros(2, dtype=util.particle_dt())
        others["xposition"] = [4e-10, 8e-10]
        energy = pairwise.particle_energy(
            (0.0, 0.0), 0, others, 30, 6e-10, ARGON_MODEL["pair_potentials"], ARGON_MODEL["species"]
        )
        np.testing.assert_allclose(energy, LJ_ARGON.energies(np.array([4e-10]))[0], rtol=1e-12)

    def test_particle_energy_with_no_neighbours_is_zero(self):
        none = np.zeros(0, dtype=util.particle_dt())
        self.assertEqual(
            pairwise.particle_energy(
                (1e-10, 1e-10), 0, none, 30, 15,
                ARGON_MODEL["pair_potentials"], ARGON_MODEL["species"],
            ),
            0.0,
        )

    def test_compute_force_matches_a_reference_loop(self):
        # An independent double loop over pairs is the oracle for the pairwise
        # distances, energies, forces and accelerations. Full-box positions
        # exercise the minimum image. cut_off is large so no pair is zeroed.
        rng = np.random.default_rng(0)
        n = 8
        box_length = 30e-10
        particles = np.zeros(n, dtype=util.particle_dt())
        particles["xposition"] = rng.uniform(0, box_length, n)
        particles["yposition"] = rng.uniform(0, box_length, n)
        types = [0, 0, 1, 1, 0, 1, 0, 1]
        particles["types"] = types
        species = MIXTURE_MODEL["species"]
        pair_potentials = MIXTURE_MODEL["pair_potentials"]
        masses_kg = np.array([species[t].mass for t in types]) * ATOMIC_MASS_UNIT
        cut_off = 1e-8

        ref_dr, ref_energy, ref_force = [], [], []
        ref_ax = np.zeros(n)
        ref_ay = np.zeros(n)
        for a in range(n - 1):
            for b in range(a + 1, n):
                dx = particles["xposition"][a] - particles["xposition"][b]
                dy = particles["yposition"][a] - particles["yposition"][b]
                dx -= box_length * np.round(dx / box_length)
                dy -= box_length * np.round(dy / box_length)
                dr = np.hypot(dx, dy)
                pair = pairwise.pair_potential(
                    pair_potentials, species[types[a]], species[types[b]]
                )
                force = pair.forces(dr)
                ref_dr.append(dr)
                ref_energy.append(pair.energies(dr))
                ref_force.append(force)
                ref_ax[a] += force * dx / dr / masses_kg[a]
                ref_ax[b] -= force * dx / dr / masses_kg[b]
                ref_ay[a] += force * dy / dr / masses_kg[a]
                ref_ay[b] -= force * dy / dr / masses_kg[b]

        particles, distances, forces, energies = pairwise.compute_force(
            particles, box_length, cut_off, pair_potentials, species
        )
        np.testing.assert_allclose(distances, ref_dr, rtol=1e-12)
        np.testing.assert_allclose(energies, ref_energy, rtol=1e-12)
        np.testing.assert_allclose(forces, ref_force, rtol=1e-12)
        np.testing.assert_allclose(particles["xacceleration"], ref_ax, rtol=1e-12)
        np.testing.assert_allclose(particles["yacceleration"], ref_ay, rtol=1e-12)
