import unittest

from pylj import simulation
from pylj.tests.argon import ARGON_MODEL, WELL_MODEL
from pylj.tests.test_placement import place


class TestInitialEnergyCheck(unittest.TestCase):
    def test_refuses_an_overlapping_lattice(self):
        # 16 argon on a 4 by 4 lattice in a 10 Angstrom box are 2.5 Angstrom
        # apart, inside sigma: about 90 k_B T of potential energy per particle.
        c, cut_off = place(16, 300, 10)
        energy = c.potential_energy(ARGON_MODEL["pair_potentials"], cut_off)
        with self.assertRaisesRegex(ValueError, "k_B T of potential energy"):
            simulation._check_initial_energy(energy, 16, 300)

    def test_refuses_a_lattice_inside_a_hard_core(self):
        c, cut_off = place(16, 300, 10, model=WELL_MODEL)
        energy = c.potential_energy(WELL_MODEL["pair_potentials"], cut_off)
        with self.assertRaisesRegex(ValueError, "not finite"):
            simulation._check_initial_energy(energy, 16, 300)

    def test_accepts_a_lattice_below_the_limit(self):
        # 16 argon in a 12 Angstrom box store about 5.6 k_B T per particle
        # at 300 K, under the limit of 10.
        c, cut_off = place(16, 300, 12)
        energy = c.potential_energy(ARGON_MODEL["pair_potentials"], cut_off)
        simulation._check_initial_energy(energy, 16, 300)
