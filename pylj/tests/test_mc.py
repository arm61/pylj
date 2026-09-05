import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import mc, pairwise
from pylj.constants import BOLTZMANN
from pylj.tests.argon import ARGON, ARGON_MODEL, MIXTURE_MODEL, WELL_MODEL


def run_moves(system, steps):
    """Propose, test and apply ``steps`` Monte Carlo moves."""
    for _ in range(steps):
        proposal = system.propose()
        if mc.accept(proposal.energy_change, system.temperature, rng=system.rng):
            system.apply(proposal)
    return system


def total_energy(system):
    """The exact total pair energy of the current configuration."""
    _, energies = pairwise.compute_energy(
        system.particles, system.box_length, system.cut_off, system.pair_potentials, system.species
    )
    return energies.sum()


class TestMc(unittest.TestCase):
    def test_initialise_square(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        assert_equal(a.number_of_particles, 2)
        assert_almost_equal(a.box_length, 8e-10)
        assert_almost_equal(a.temperature, 300)
        assert_almost_equal(a.particles["xposition"] * 1e10, [2, 2])
        assert_almost_equal(a.particles["yposition"] * 1e10, [2, 6])
        assert_equal(a.simulation, "mc")

    def test_initialise_computes_initial_energy(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        assert_almost_equal(a.distances * 1e10, [4.0])
        self.assertTrue(a.energies[0] != 0)

    def test_initialise_sets_the_starting_energy(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        assert_almost_equal(a.energy, a.energies.sum())
        self.assertTrue(a.energy != 0)

    def test_initialise_rejects_a_non_positive_or_infinite_temperature(self):
        for temperature in (0, -300, np.inf):
            with self.assertRaisesRegex(ValueError, "temperature must be positive"):
                mc.initialise(4, temperature, 8, "square", **ARGON_MODEL)

    def test_initialise_passes_the_placement_temperature_to_the_system(self):
        a = mc.initialise(4, 100, 20, "metropolis", placement_temperature=50, **ARGON_MODEL)
        self.assertEqual(a.placement_temperature, 50)

    def test_square_well_drives_monte_carlo(self):
        # Nine particles on a 3 by 3 lattice in a 12 Angstrom box: each has
        # four lattice neighbours 4 Angstrom away, inside the well, and four
        # diagonal ones 5.66 Angstrom away, beyond it, so 18 pairs sit at
        # -epsilon. The cut-off, half the box, is 6 Angstrom.
        system = mc.initialise(9, 300, 12, "square", seed=2, **WELL_MODEL)
        assert_almost_equal(system.energy * 1e21, -27.0)
        overlaps = 0
        for _ in range(50):
            proposal = system.propose()
            accepted = mc.accept(proposal.energy_change, system.temperature, rng=system.rng)
            if np.isinf(proposal.energy_change):
                # A trial inside a hard core is always rejected.
                overlaps += 1
                self.assertFalse(accepted)
            if accepted:
                system.apply(proposal)
        self.assertGreater(overlaps, 0)
        np.testing.assert_allclose(system.energy, total_energy(system), rtol=1e-9, atol=1e-33)
        self.assertTrue(np.all(system.particles["xacceleration"] == 0.0))

    def test_initialize_passes_keyword_arguments_through(self):
        a = mc.initialize(2, 300, 8, "square", seed=5, **ARGON_MODEL)
        assert_equal(a.species, [ARGON])
        assert_equal(a.pair_potentials, ARGON_MODEL["pair_potentials"])
        assert_equal(a.rng.random(), np.random.default_rng(5).random())

    def test_sample(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        a.step = 5
        a = mc.sample(300, a)
        assert_almost_equal(a.energy_sample, [300])
        assert_equal(a.step_sample, [5])

    def test_accept_takes_a_downhill_change_without_drawing(self):
        rng = np.random.default_rng(3)
        untouched = np.random.default_rng(3)
        self.assertTrue(mc.accept(-1e-20, 300, rng=rng))
        self.assertTrue(mc.accept(0.0, 300, rng=rng))
        self.assertEqual(rng.random(), untouched.random())

    def test_accept_tests_an_uphill_change_against_n(self):
        # A rise of 1e-20 J at 300 K has a Boltzmann factor of about 0.09.
        self.assertTrue(mc.accept(1e-20, 300, random_number=0.01))
        self.assertFalse(mc.accept(1e-20, 300, random_number=0.1))

    def test_accept_draws_from_the_supplied_generator(self):
        # With n the generator's first draw, an uphill change whose
        # acceptance probability is (1 + n) / 2 is accepted and one whose
        # probability is n / 2 is rejected.
        n = np.random.default_rng(5).random()
        accepted = -BOLTZMANN * 300 * np.log((1 + n) / 2)
        rejected = -BOLTZMANN * 300 * np.log(n / 2)
        self.assertTrue(mc.accept(accepted, 300, rng=np.random.default_rng(5)))
        self.assertFalse(mc.accept(rejected, 300, rng=np.random.default_rng(5)))

    def test_accept_draws_afresh_on_every_call(self):
        # An uphill change with acceptance probability one half: ten calls
        # give both outcomes.
        change = BOLTZMANN * 300 * np.log(2)
        rng = np.random.default_rng(0)
        outcomes = [mc.accept(change, 300, rng=rng) for _ in range(10)]
        self.assertIn(True, outcomes)
        self.assertIn(False, outcomes)

    def test_accept_without_a_generator_draws_its_own(self):
        # The bare call a student is most likely to type: an uphill change
        # with acceptance probability one half gives both outcomes.
        change = BOLTZMANN * 300 * np.log(2)
        outcomes = [mc.accept(change, 300) for _ in range(40)]
        self.assertIn(True, outcomes)
        self.assertIn(False, outcomes)

    def test_propose_leaves_the_configuration_untouched(self):
        system = mc.initialise(16, 300, 30, "square", seed=1, **ARGON_MODEL)
        x = system.particles["xposition"].copy()
        y = system.particles["yposition"].copy()
        energy = system.energy
        system.propose()
        assert_equal(system.particles["xposition"], x)
        assert_equal(system.particles["yposition"], y)
        self.assertEqual(system.energy, energy)

    def test_propose_moves_exactly_one_particle_inside_the_box(self):
        system = mc.initialise(16, 300, 30, "square", seed=1, **ARGON_MODEL)
        proposal = system.propose()
        moved = (proposal.xposition != system.particles["xposition"]) | (
            proposal.yposition != system.particles["yposition"]
        )
        self.assertEqual(moved.sum(), 1)
        self.assertTrue(0 <= proposal.xposition[moved][0] < system.box_length)
        self.assertTrue(0 <= proposal.yposition[moved][0] < system.box_length)

    def test_propose_energy_change_matches_a_full_recompute(self):
        # The oracle: apply the proposal to a copy and recompute every pair.
        # The mixture checks the moving particle's own species is used.
        for model in (ARGON_MODEL, MIXTURE_MODEL):
            system = mc.initialise(16, 300, 40, "square", seed=1, **model)
            for _ in range(5):
                proposal = system.propose()
                trial = system.particles.copy()
                trial["xposition"] = proposal.xposition
                trial["yposition"] = proposal.yposition
                _, energies = pairwise.compute_energy(
                    trial, system.box_length, system.cut_off, system.pair_potentials, system.species
                )
                np.testing.assert_allclose(
                    proposal.energy_change, energies.sum() - system.energy, rtol=1e-9, atol=1e-33
                )
                system.apply(proposal)

    def test_apply_updates_the_positions_and_the_energy(self):
        system = mc.initialise(16, 300, 30, "square", seed=1, **ARGON_MODEL)
        proposal = system.propose()
        system.apply(proposal)
        assert_equal(system.particles["xposition"], proposal.xposition)
        assert_equal(system.particles["yposition"], proposal.yposition)
        np.testing.assert_allclose(system.energy, total_energy(system), rtol=1e-9, atol=1e-33)

    def test_mc_sample_refreshes_the_pair_arrays_and_the_energy(self):
        system = mc.initialise(16, 300, 30, "square", seed=1, **ARGON_MODEL)
        for _ in range(20):
            system.apply(system.propose())
        # A corrupted running total is replaced by the exact one.
        system.energy = 1.0
        system.mc_sample()
        distances, _, _ = pairwise.dist(
            system.particles["xposition"], system.particles["yposition"], system.box_length
        )
        assert_equal(system.distances, distances)
        self.assertEqual(system.energy, float(system.energies.sum()))
        assert_equal(system.energy_sample, [system.energy])

    def test_seeded_runs_are_identical(self):
        def run(seed):
            system = mc.initialise(16, 300, 30, "metropolis", seed=seed, **ARGON_MODEL)
            return run_moves(system, 200)

        first = run(7)
        second = run(7)
        other = run(8)
        assert_equal(first.particles["xposition"], second.particles["xposition"])
        assert_equal(first.particles["yposition"], second.particles["yposition"])
        assert_equal(first.energy, second.energy)
        self.assertFalse(
            np.array_equal(first.particles["xposition"], other.particles["xposition"])
        )
