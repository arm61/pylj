import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import mc
from pylj.constants import BOLTZMANN
from pylj.potentials import square_well
from pylj.tests.argon import ARGON, ARGON_MODEL


class TestMc(unittest.TestCase):
    def test_initialise_square(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        assert_equal(a.number_of_particles, 2)
        assert_almost_equal(a.box_length, 8e-10)
        assert_almost_equal(a.init_temp, 300)
        assert_almost_equal(a.particles["xposition"] * 1e10, [2, 2])
        assert_almost_equal(a.particles["yposition"] * 1e10, [2, 6])
        assert_equal(a.simulation, "mc")

    def test_initialise_computes_initial_energy(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        assert_almost_equal(a.distances * 1e10, [4.0])
        self.assertTrue(a.energies[0] != 0)

    def test_initialise_sets_the_starting_energy(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        assert_almost_equal(a.old_energy, a.energies.sum())
        self.assertTrue(a.old_energy != 0)

    def test_square_well_drives_monte_carlo(self):
        # Nine particles on a 3 by 3 lattice in a 12 Angstrom box: each has
        # four lattice neighbours 4 Angstrom away, inside the well, and four
        # diagonal ones 5.66 Angstrom away, beyond it, so 18 pairs sit at
        # -epsilon. The cut-off, half the box, is 6 Angstrom.
        well = square_well(epsilon=1.5e-21, sigma=3e-10, lambda_=1.5)
        model = {"species": [ARGON], "pair_potentials": {(ARGON, ARGON): well}}
        system = mc.initialise(9, 300, 12, "square", seed=2, **model)
        assert_almost_equal(system.old_energy * 1e21, -27.0)
        for _ in range(50):
            system.select_random_particle()
            system.new_random_position()
            system.compute_energy()
            system.new_energy = system.energies.sum()
            if system.metropolis():
                system.accept()
            else:
                system.reject()
        self.assertTrue(np.isfinite(system.old_energy))
        self.assertTrue(np.all(system.particles["xacceleration"] == 0.0))

    def test_initialize_passes_keyword_arguments_through(self):
        a = mc.initialize(2, 300, 8, "square", seed=5, **ARGON_MODEL)
        assert_equal(a.species, [ARGON])
        assert_equal(a.pair_potentials, ARGON_MODEL["pair_potentials"])
        assert_equal(a.rng.random(), np.random.default_rng(5).random())

    def test_initialize_square(self):
        a = mc.initialize(2, 300, 8, "square", **ARGON_MODEL)
        assert_equal(a.number_of_particles, 2)
        assert_almost_equal(a.box_length, 8e-10)
        assert_almost_equal(a.init_temp, 300)
        assert_almost_equal(a.particles["xposition"] * 1e10, [2, 2])
        assert_almost_equal(a.particles["yposition"] * 1e10, [2, 6])

    def test_sample(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        a.step = 5
        a = mc.sample(300, a)
        assert_almost_equal(a.energy_sample, [300])
        assert_equal(a.step_sample, [5])

    def test_select_random_particle(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        b, c = mc.select_random_particle(a.particles, np.random.default_rng(0))
        self.assertTrue(0 <= b < 2)
        self.assertTrue(0 <= c[0] <= 8e-10)
        self.assertTrue(0 <= c[1] <= 8e-10)

    def test_get_new_particle(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        rng = np.random.default_rng(0)
        b, c = mc.select_random_particle(a.particles, rng)
        d = mc.get_new_particle(a.particles, b, a.box_length, rng)
        self.assertTrue(0 <= d["xposition"][b] <= 8e-10)
        self.assertTrue(0 <= d["yposition"][b] <= 8e-10)

    def test_accept(self):
        a = mc.accept(300)
        assert_almost_equal(a, 300)

    def test_reject(self):
        a = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        b = [1e-10, 1e-10]
        c = mc.reject(b, a.particles, 1)
        assert_almost_equal(c["xposition"][1] * 1e10, 1)
        assert_almost_equal(c["yposition"][1] * 1e10, 1)

    def test_metropolis_energy_reduce(self):
        a = mc.metropolis(300, 100, 1)
        self.assertTrue(a)

    def test_metropolis_energy_increase_accept(self):
        a = mc.metropolis(300, 100e-20, 101e-20, n=0.01)
        self.assertTrue(a)

    def test_metropolis_energy_increase_reject(self):
        a = mc.metropolis(300, 100e-20, 101e-20, n=0.1)
        self.assertFalse(a)

    def test_metropolis_draws_a_new_random_number_on_every_call(self):
        # An uphill move whose acceptance probability is exactly one half, so
        # ten calls that each draw afresh give both outcomes.
        energy_difference = BOLTZMANN * 300 * np.log(2)
        rng = np.random.default_rng(0)
        outcomes = [mc.metropolis(300, 0.0, energy_difference, rng=rng) for _ in range(10)]
        self.assertIn(True, outcomes)
        self.assertIn(False, outcomes)

    def test_metropolis_draws_from_the_supplied_generator(self):
        # With n the generator's first draw, an uphill move whose acceptance
        # probability is (1 + n) / 2 is accepted and one whose probability is
        # n / 2 is rejected; a draw from any other generator would give the
        # wrong answer for one of the two on average half the time.
        n = np.random.default_rng(5).random()
        accepted = -BOLTZMANN * 300 * np.log((1 + n) / 2)
        rejected = -BOLTZMANN * 300 * np.log(n / 2)
        self.assertTrue(mc.metropolis(300, 0.0, accepted, rng=np.random.default_rng(5)))
        self.assertFalse(mc.metropolis(300, 0.0, rejected, rng=np.random.default_rng(5)))

    def test_system_metropolis_accepts_downhill_and_rejects_steep_uphill_moves(self):
        system = mc.initialise(2, 300, 8, "square", **ARGON_MODEL)
        system.old_energy = 1e-20
        system.new_energy = 0.0
        self.assertTrue(system.metropolis())
        system.old_energy = 0.0
        system.new_energy = 100 * BOLTZMANN * 300
        self.assertFalse(system.metropolis())

    def test_metropolis_does_not_draw_for_a_downhill_move(self):
        rng = np.random.default_rng(3)
        untouched = np.random.default_rng(3)
        mc.metropolis(300, 1e-20, 0.0, rng=rng)
        self.assertEqual(rng.random(), untouched.random())

    def test_seeded_runs_are_identical(self):
        def run(seed):
            system = mc.initialise(16, 300, 30, "random", seed=seed, **ARGON_MODEL)
            for _ in range(200):
                system.select_random_particle()
                system.new_random_position()
                system.compute_energy()
                system.new_energy = system.energies.sum()
                if system.metropolis():
                    system.accept()
                else:
                    system.reject()
            return system

        first = run(7)
        second = run(7)
        other = run(8)
        assert_equal(first.particles["xposition"], second.particles["xposition"])
        assert_equal(first.particles["yposition"], second.particles["yposition"])
        assert_equal(first.old_energy, second.old_energy)
        self.assertFalse(
            np.array_equal(first.particles["xposition"], other.particles["xposition"])
        )
