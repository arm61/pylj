from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN
from pylj.potentials import PairPotential, Species

#: The potential acting between each pair of species, keyed by the two
#: species in either order.
PairPotentials = Mapping[tuple[Species, Species], PairPotential]


def pair_potential(
    pair_potentials: PairPotentials, species_1: Species, species_2: Species
) -> PairPotential:
    """Return the potential acting between two species.

    The mapping is keyed by unordered pairs, so the two species are looked
    up in either order.

    Args:
        pair_potentials: The potential between each pair of species.
        species_1: One species of the pair.
        species_2: The other species of the pair.

    Returns:
        The potential for the pair.

    Raises:
        KeyError: If the mapping has no entry for the pair in either order.
    """
    if (species_1, species_2) in pair_potentials:
        return pair_potentials[(species_1, species_2)]
    return pair_potentials[(species_2, species_1)]


def particle_masses(particles: np.ndarray, species: Sequence[Species]) -> NDArray[np.float64]:
    """Return the mass of each particle, in atomic mass units.

    Args:
        particles: The particles, as a ``util.particle_dt`` array whose
            ``types`` field indexes ``species``.
        species: The species, in the order ``types`` indexes.

    Returns:
        The mass of each particle, from its species.
    """
    return np.array([one.mass for one in species], dtype=float)[particles["types"]]


def _species_pairs(
    types: NDArray[np.int64],
) -> Iterator[tuple[NDArray[np.bool_], int, int]]:
    """Yield the pairs of each unordered pair of species indices present.

    A pair of species 0 and 1 is the same pair as 1 and 0, so each unordered
    pair is yielded once, with a mask over the i < j pair arrays selecting
    the pairs it covers.

    Args:
        types: The species index of each particle.

    Yields:
        The mask, the lower species index and the upper species index.
    """
    i, j = np.triu_indices(types.size, 1)
    lower = np.minimum(types[i], types[j])
    upper = np.maximum(types[i], types[j])
    for type_1, type_2 in sorted(set(zip(lower.tolist(), upper.tolist(), strict=True))):
        yield (lower == type_1) & (upper == type_2), type_1, type_2


def compute_energy(particles, box_length, cut_off, pair_potentials, species):
    """Calculate the pair distances and pair energies of the configuration.

    Only ``energies`` is called on the potentials, so a potential with no
    finite force, such as the square well, drives Monte Carlo through this
    path. The particles are not changed.

    Args:
        particles: The particles, as a ``util.particle_dt`` array whose
            ``types`` field indexes ``species``.
        box_length: Length of a single dimension of the simulation square,
            in metres.
        cut_off: The separation beyond which the pair energy is taken to be
            zero, in metres.
        pair_potentials: The potential between each pair of species.
        species: The species, in the order ``types`` indexes.

    Returns:
        The distance between each pair of particles, in metres, and the
        energy of each pair, in joules, both in i < j pair order.
    """
    distances, _, _ = dist(particles["xposition"], particles["yposition"], box_length)
    energies = np.zeros(distances.size)
    for mask, type_1, type_2 in _species_pairs(particles["types"]):
        potential = pair_potential(pair_potentials, species[type_1], species[type_2])
        energies[mask] = potential.energies(distances[mask])
    energies[distances > cut_off] = 0.0
    return distances, energies


def compute_force(particles, box_length, cut_off, pair_potentials, species):
    """Calculate the pair forces and the acceleration of each particle.

    Each pair's radial force is projected onto the pair separation and
    divided by the mass of the particle it acts on, so the accelerations
    replace those already on the particles.

    Args:
        particles: The particles, as a ``util.particle_dt`` array whose
            ``types`` field indexes ``species``.
        box_length: Length of a single dimension of the simulation square,
            in metres.
        cut_off: The separation beyond which the pair energy and force are
            taken to be zero, in metres.
        pair_potentials: The potential between each pair of species.
        species: The species, in the order ``types`` indexes.

    Returns:
        The particles with their accelerations replaced; the distance
        between each pair of particles, in metres; the force on each pair,
        in newtons; and the energy of each pair, in joules. The pair arrays
        are in i < j pair order.
    """
    particles["xacceleration"] = 0.0
    particles["yacceleration"] = 0.0
    distances, dx, dy = dist(particles["xposition"], particles["yposition"], box_length)
    forces = np.zeros(distances.size)
    energies = np.zeros(distances.size)
    for mask, type_1, type_2 in _species_pairs(particles["types"]):
        potential = pair_potential(pair_potentials, species[type_1], species[type_2])
        energies[mask] = potential.energies(distances[mask])
        forces[mask] = potential.forces(distances[mask])
    forces[distances > cut_off] = 0.0
    energies[distances > cut_off] = 0.0
    masses_kg = particle_masses(particles, species) * ATOMIC_MASS_UNIT
    particles = update_accelerations(particles, forces, masses_kg, dx, dy, distances)
    return particles, distances, forces, energies


def update_accelerations(particles, f, m, dx, dy, dr):
    """Add the accelerations from the pair forces to each particle.

    The accelerations already on the particles are added to, so the caller
    zeroes them first. The pair arrays are in i < j order, as returned by
    dist.

    Args:
        particles: The particles, as a ``util.particle_dt`` array.
        f: The force on each pair of particles, in newtons.
        m: The mass of each particle, in kilograms.
        dx: The x-dimension component of each pair separation, x_i - x_j,
            in metres.
        dy: The y-dimension component of each pair separation, y_i - y_j,
            in metres.
        dr: The distance between each pair of particles, in metres.

    Returns:
        The particles with their accelerations accumulated from the pairs.
    """
    i, j = np.triu_indices(particles.size, 1)
    fx = f * dx / dr
    fy = f * dy / dr
    # each pair pushes particle i one way and particle j the other, and
    # each is accelerated by its own mass.
    np.add.at(particles["xacceleration"], i, fx / m[i])
    np.add.at(particles["xacceleration"], j, -fx / m[j])
    np.add.at(particles["yacceleration"], i, fy / m[i])
    np.add.at(particles["yacceleration"], j, -fy / m[j])
    return particles


def calculate_pressure(
    distances, forces, box_length, number_of_particles, temperature
):
    r"""Calculate the instantaneous pressure of the simulation cell in two
    dimensions, from the pair distances and forces of the configuration:

    .. math::
        p = \frac{N k_B T}{L^2} + \frac{1}{2 L^2} \sum_{i} \sum_{j > i}
        r_{ij} f_{ij}

    Parameters
    ----------
    distances: float, array_like
        The distance between each pair of particles, in metres.
    forces: float, array_like
        The force between each pair of particles, in newtons.
    box_length: float
        Length of a single dimension of the simulation square, in metres.
    number_of_particles: int
        The number of particles in the simulation.
    temperature: float
        Instantaneous temperature of the simulation, in kelvin.

    Returns
    -------
    float:
        Instantaneous pressure of the simulation, in N / m (a two-dimensional
        pressure).
    """
    virial = np.sum(forces * distances) / (2 * box_length * box_length)
    ideal = number_of_particles * BOLTZMANN * temperature / (box_length * box_length)
    return virial + ideal


def dist(xposition, yposition, box_length):
    """Return the minimum-image distances between every pair of particles.

    Parameters
    ----------
    xposition: float, array_like (N)
        The x-dimension positions of the N particles, in metres.
    yposition: float, array_like (N)
        The y-dimension positions of the N particles, in metres.
    box_length: float
        The box length of the simulation cell, in metres.

    Returns
    -------
    dr: float, array_like (N (N - 1) / 2)
        The distance between each pair of particles, in metres, in i < j pair
        order.
    dx: float, array_like (N (N - 1) / 2)
        The x-dimension component of each pair separation, x_i - x_j, in
        metres.
    dy: float, array_like (N (N - 1) / 2)
        The y-dimension component of each pair separation, y_i - y_j, in
        metres.
    """
    i, j = np.triu_indices(xposition.size, 1)
    dx = xposition[i] - xposition[j]
    dy = yposition[i] - yposition[j]
    dx -= box_length * np.round(dx / box_length)
    dy -= box_length * np.round(dy / box_length)
    dr = np.hypot(dx, dy)
    return dr, dx, dy
