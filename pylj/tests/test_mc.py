import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import mc
from pylj.constants import BOLTZMANN
from pylj.mc import MCSimulation
from pylj.md import MDSimulation
from pylj.pairwise import minimum_image
from pylj.tests.argon import ARGON_MODEL, LJ_ARGON, MIXTURE_MODEL, WELL_MODEL


def total_energy(sim):
    """The exact total pair energy of the current configuration."""
    return sim.configuration.potential_energy(sim.pair_potentials, sim.cut_off)


class TestAccept(unittest.TestCase):
    def test_takes_a_downhill_change_without_drawing(self):
        rng = np.random.default_rng(3)
        untouched = np.random.default_rng(3)
        self.assertTrue(mc.accept(-1e-20, 300, rng=rng))
        self.assertTrue(mc.accept(0.0, 300, rng=rng))
        self.assertEqual(rng.random(), untouched.random())

    def test_tests_an_uphill_change_against_the_random_number(self):
        # A rise of 1e-20 J at 300 K has a Boltzmann factor of about 0.09.
        self.assertTrue(mc.accept(1e-20, 300, random_number=0.01))
        self.assertFalse(mc.accept(1e-20, 300, random_number=0.1))

    def test_draws_from_the_supplied_generator(self):
        # With n the generator's first draw, an uphill change whose
        # acceptance probability is (1 + n) / 2 is accepted and one whose
        # probability is n / 2 is rejected.
        n = np.random.default_rng(5).random()
        accepted = -BOLTZMANN * 300 * np.log((1 + n) / 2)
        rejected = -BOLTZMANN * 300 * np.log(n / 2)
        self.assertTrue(mc.accept(accepted, 300, rng=np.random.default_rng(5)))
        self.assertFalse(mc.accept(rejected, 300, rng=np.random.default_rng(5)))

    def test_draws_afresh_on_every_call(self):
        # An uphill change with acceptance probability one half: ten calls
        # give both outcomes.
        change = BOLTZMANN * 300 * np.log(2)
        rng = np.random.default_rng(0)
        outcomes = [mc.accept(change, 300, rng=rng) for _ in range(10)]
        self.assertIn(True, outcomes)
        self.assertIn(False, outcomes)

    def test_without_a_generator_draws_its_own(self):
        # The bare call a student is most likely to type: an uphill change
        # with acceptance probability one half gives both outcomes.
        change = BOLTZMANN * 300 * np.log(2)
        outcomes = [mc.accept(change, 300) for _ in range(40)]
        self.assertIn(True, outcomes)
        self.assertIn(False, outcomes)


class TestInitialise(unittest.TestCase):
    def test_square_lattice_at_a_temperature(self):
        a = MCSimulation.initialise(2, 300, 8, **ARGON_MODEL)
        c = a.configuration
        self.assertEqual(c.number_of_particles, 2)
        assert_almost_equal(c.box, 8e-10)
        assert_almost_equal(c.position * 1e10, [[2, 2], [2, 6]])
        assert_almost_equal(a.temperature, 300)
        assert_almost_equal(a.cut_off * 1e10, 4.0)
        self.assertEqual(a.steps, 0)
        self.assertEqual(a.accepted, 0)

    def test_sets_the_starting_energy(self):
        a = MCSimulation.initialise(2, 300, 8, **ARGON_MODEL)
        assert_almost_equal(a.energy, total_energy(a))
        self.assertNotEqual(a.energy, 0.0)

    def test_rejects_a_non_positive_or_infinite_temperature(self):
        for temperature in (0, -300, np.inf):
            with self.assertRaisesRegex(ValueError, "temperature must be positive"):
                MCSimulation.initialise(4, temperature, 8, **ARGON_MODEL)

    def test_passes_the_placement_temperature_and_seed_through(self):
        with self.assertRaisesRegex(ValueError, "Could not place"):
            MCSimulation.initialise(
                50,
                100,
                27,
                init_conf="metropolis",
                placement_temperature=1.0,
                seed=0,
                **ARGON_MODEL,
            )
        a = MCSimulation.initialise(2, 300, 8, seed=5, **ARGON_MODEL)
        self.assertEqual(a.rng.random(), np.random.default_rng(5).random())

    def test_refuses_a_lattice_inside_a_hard_core(self):
        # 16 particles on a 4 by 4 lattice in a 10 Angstrom box are 2.5
        # Angstrom apart, inside a 3 Angstrom hard core that the 5 Angstrom
        # cut-off clears: the lattice energy is infinite.
        with self.assertRaisesRegex(ValueError, "not finite"):
            MCSimulation.initialise(16, 300, 10, **WELL_MODEL)

    def test_one_particle_is_allowed(self):
        a = MCSimulation.initialise(1, 300, 20, **ARGON_MODEL)
        self.assertEqual(a.energy, 0.0)


class TestConstructor(unittest.TestCase):
    def test_accepts_an_md_configuration(self):
        md_simulation = MDSimulation.initialise(4, 100, 20, **ARGON_MODEL)
        a = MCSimulation(md_simulation.configuration, ARGON_MODEL["pair_potentials"], 100)
        self.assertIs(a.configuration, md_simulation.configuration)
        for _ in range(20):
            a.step()
        # The velocities are carried, untouched, through the moves.
        self.assertGreater(a.accepted, 0)
        assert_equal(a.configuration.velocity, md_simulation.configuration.velocity)

    def test_validates_the_temperature_before_the_model(self):
        c = MDSimulation.initialise(4, 100, 20, **ARGON_MODEL).configuration
        with self.assertRaisesRegex(ValueError, "temperature must be positive"):
            MCSimulation(c, {}, -1)


class TestMoves(unittest.TestCase):
    def test_square_well_drives_monte_carlo(self):
        # Nine particles on a 3 by 3 lattice in a 12 Angstrom box: each has
        # four lattice neighbours 4 Angstrom away, inside the well, and four
        # diagonal ones 5.66 Angstrom away, beyond it, so 18 pairs sit at
        # -epsilon. The cut-off, half the box, is 6 Angstrom.
        a = MCSimulation.initialise(9, 300, 12, seed=2, **WELL_MODEL)
        assert_almost_equal(a.energy * 1e21, -27.0)
        overlaps = 0
        for _ in range(50):
            proposal = a.propose()
            accepted = mc.accept(proposal.energy_change, a.temperature, rng=a.rng)
            if np.isinf(proposal.energy_change):
                # A trial inside a hard core is always rejected.
                overlaps += 1
                self.assertFalse(accepted)
            if accepted:
                a.apply(proposal)
        self.assertGreater(overlaps, 0)
        np.testing.assert_allclose(a.energy, total_energy(a), rtol=1e-9, atol=1e-33)

    def test_proposals_compare_by_identity(self):
        a = MCSimulation.initialise(16, 300, 30, seed=1, **ARGON_MODEL)
        proposal = a.propose()
        self.assertEqual(proposal, proposal)
        self.assertNotEqual(proposal, mc.Proposal(proposal.position, proposal.energy_change))

    def test_propose_leaves_the_configuration_untouched(self):
        a = MCSimulation.initialise(16, 300, 30, seed=1, **ARGON_MODEL)
        before = a.configuration
        position = before.position.copy()
        energy = a.energy
        a.propose()
        self.assertIs(a.configuration, before)
        assert_equal(a.configuration.position, position)
        self.assertEqual(a.energy, energy)

    def test_propose_moves_exactly_one_particle_inside_the_box(self):
        a = MCSimulation.initialise(16, 300, 30, seed=1, **ARGON_MODEL)
        proposal = a.propose()
        moved = np.any(proposal.position != a.configuration.position, axis=1)
        self.assertEqual(moved.sum(), 1)
        trial = proposal.position[moved][0]
        self.assertTrue(np.all((0 <= trial) & (trial < a.configuration.box)))

    def test_propose_energy_change_matches_a_full_recompute(self):
        # The oracle: apply the proposal to a copy and recompute every pair.
        # The mixture checks the moving particle's own species is used, so
        # the proposals must move particles of both species.
        for model in (ARGON_MODEL, MIXTURE_MODEL):
            a = MCSimulation.initialise(16, 300, 40, seed=1, **model)
            moved_species = set()
            for _ in range(5):
                proposal = a.propose()
                trial = a.configuration.replace(position=proposal.position)
                expected = trial.potential_energy(a.pair_potentials, a.cut_off) - total_energy(a)
                np.testing.assert_allclose(proposal.energy_change, expected, rtol=1e-9, atol=1e-33)
                moved = np.any(proposal.position != a.configuration.position, axis=1)
                moved_species.add(int(a.configuration.species_index[moved][0]))
                a.apply(proposal)
            self.assertEqual(moved_species, set(range(len(model["species"]))))

    def test_apply_updates_the_positions_and_the_energy(self):
        a = MCSimulation.initialise(16, 300, 30, seed=1, **ARGON_MODEL)
        proposal = a.propose()
        a.apply(proposal)
        assert_equal(a.configuration.position, proposal.position)
        np.testing.assert_allclose(a.energy, total_energy(a), rtol=1e-9, atol=1e-33)

    def test_step_proposes_decides_and_counts(self):
        a = MCSimulation.initialise(16, 300, 30, seed=1, **ARGON_MODEL)
        for _ in range(100):
            a.step()
        self.assertEqual(a.steps, 100)
        self.assertGreater(a.accepted, 0)
        self.assertLess(a.accepted, 100)
        np.testing.assert_allclose(a.energy, total_energy(a), rtol=1e-9, atol=1e-33)

    def test_sample_sets_the_exact_energy_and_records_it(self):
        a = MCSimulation.initialise(16, 300, 30, seed=1, **ARGON_MODEL)
        for _ in range(20):
            a.apply(a.propose())
        # A corrupted running total is replaced by the exact one.
        a.energy = 1.0
        a.sample()
        self.assertEqual(a.energy, total_energy(a))
        assert_equal(a.samples.potential_energy, [a.energy])
        assert_equal(a.samples.step, [0])

    def test_seeded_runs_are_identical(self):
        def run(seed):
            a = MCSimulation.initialise(
                16, 300, 30, init_conf="metropolis", seed=seed, **ARGON_MODEL
            )
            for _ in range(200):
                a.step()
            return a

        first, second, other = run(7), run(7), run(8)
        assert_equal(first.configuration.position, second.configuration.position)
        self.assertEqual(first.energy, second.energy)
        self.assertEqual(first.accepted, second.accepted)
        self.assertFalse(np.array_equal(first.configuration.position, other.configuration.position))

    def test_samples_the_boltzmann_distribution(self):
        # Two argon particles in a 12 Angstrom box at 300 K. The relative
        # position of a pair of uniformly placed particles is uniform over
        # the box, so the mean pair energy is the Boltzmann average of the
        # minimum-image pair energy over the box, which quadrature gives.
        box, cut_off, temperature = 12e-10, 6e-10, 300
        r = np.linspace(-box / 2, box / 2, 601)
        x, y = np.meshgrid(r, r)
        separation = minimum_image(np.stack([x, y], axis=-1), box)
        distance = np.linalg.norm(separation, axis=-1)
        energy = LJ_ARGON.energies(distance)
        energy[distance > cut_off] = 0.0
        weight = np.exp(-energy / (BOLTZMANN * temperature))
        # Inside the core the weight is zero and the energy infinite: no contribution.
        contribution = np.zeros_like(weight)
        inside = weight > 0
        contribution[inside] = energy[inside] * weight[inside]
        expected = contribution.sum() / weight.sum()
        a = MCSimulation.initialise(2, temperature, 12, seed=0, **ARGON_MODEL)
        energies = []
        for _ in range(40000):
            a.step()
            energies.append(a.energy)
        np.testing.assert_allclose(np.mean(energies[5000:]), expected, rtol=0.05)

    def test_restart_carries_the_energy_and_resets_the_counts(self):
        a = MCSimulation.initialise(4, 300, 12, **ARGON_MODEL)
        for _ in range(10):
            a.step()
        a.sample()
        production = a.restart()
        self.assertEqual(production.energy, a.energy)
        self.assertEqual(production.temperature, a.temperature)
        self.assertEqual(production.steps, 0)
        self.assertEqual(production.accepted, 0)
        self.assertIsInstance(production.samples, mc.MCSamples)
        self.assertEqual(production.samples.step.size, 0)
        self.assertEqual(a.samples.potential_energy.size, 1)
