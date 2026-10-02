"""Physical constants in pylj's units: Angstrom, picoseconds, kJ/mol, atomic
mass units and kelvin."""

from scipy.constants import N_A, R, atomic_mass

BOLTZMANN = R / 1000
"""Boltzmann constant, kJ/mol/K, from scipy.constants."""

# One amu Angstrom^2 / ps^2 is atomic_mass * (1e-10)^2 / (1e-12)^2 joules.
KJ_PER_MOL = (1000 / N_A) / (atomic_mass * 1e4)
"""One kJ/mol in amu Angstrom^2 / ps^2, from scipy.constants."""
