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


def particle_energy(
    position: tuple[float, float],
    species_index: int,
    others: np.ndarray,
    box_length: float,
    cut_off: float,
    pair_potentials: PairPotentials,
    species: Sequence[Species],
) -> float:
    """Return the interaction energy of one particle with a set of others.

    The particle is described by its position and species; ``others`` are
    the particles it interacts with. Each pair is evaluated on the potential
    for the two species, at the minimum-image separation, and pairs beyond
    the cut-off contribute nothing. Only ``energies`` is called on the
    potentials.

    Args:
        position: The particle's ``(x, y)`` position, in metres.
        species_index: The index of the particle's species in ``species``.
        others: The particles it interacts with, as a ``util.particle_dt``
            array whose ``types`` field indexes ``species``. May be empty.
        box_length: Length of a single dimension of the simulation square,
            in metres.
        cut_off: The separation beyond which the pair energy is taken to be
            zero, in metres.
        pair_potentials: The potential between each pair of species.
        species: The species, in the order ``types`` indexes.

    Returns:
        The sum of the pair energies, in joules; zero for no others.
    """
    dx = position[0] - others["xposition"]
    dy = position[1] - others["yposition"]
    dx -= box_length * np.round(dx / box_length)
    dy -= box_length * np.round(dy / box_length)
    dr = np.hypot(dx, dy)
    energies = np.zeros(dr.size)
    for other_species in np.unique(others["types"]):
        mask = others["types"] == other_species
        potential = pair_potential(
            pair_potentials, species[species_index], species[int(other_species)]
        )
        energies[mask] = potential.energies(dr[mask])
    energies[dr > cut_off] = 0.0
    return float(energies.sum())


def species_pairs(
    species_index: NDArray[np.int64],
) -> Iterator[tuple[NDArray[np.bool_], int, int]]:
    """Group the particle pairs by the two species they join.

    Each unordered pair of species present is yielded once (species 0 with
    1 is the same pair as 1 with 0), with a mask selecting the entries of the
    i < j pair arrays returned by :func:`dist` that join those two species.

    Args:
        species_index: The species index of each particle.

    Yields:
        The mask, the lower species index and the upper species index.
    """
    i, j = np.triu_indices(species_index.size, 1)
    lower = np.minimum(species_index[i], species_index[j])
    upper = np.maximum(species_index[i], species_index[j])
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
    position = np.column_stack([particles["xposition"], particles["yposition"]])
    distances, _ = dist(position, box_length)
    energies = np.zeros(distances.size)
    for mask, type_1, type_2 in species_pairs(particles["types"]):
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
    position = np.column_stack([particles["xposition"], particles["yposition"]])
    distances, separation = dist(position, box_length)
    dx, dy = separation[:, 0], separation[:, 1]
    forces = np.zeros(distances.size)
    energies = np.zeros(distances.size)
    for mask, type_1, type_2 in species_pairs(particles["types"]):
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
    virial: float, box: float, number_of_particles: int, temperature: float
) -> float:
    r"""Return the instantaneous pressure of the cell in two dimensions.

    .. math::
        p = \frac{N k_B T}{L^2} + \frac{1}{2 L^2} \sum_{i} \sum_{j > i}
        f_{ij} r_{ij}

    Args:
        virial: The sum over pairs of the radial force times the distance,
            in joules.
        box: The side length of the square periodic box, in metres.
        number_of_particles: The number of particles.
        temperature: The instantaneous temperature, in kelvin.

    Returns:
        The pressure, in newtons per metre (a two-dimensional pressure).
    """
    area = box * box
    return virial / (2 * area) + number_of_particles * BOLTZMANN * temperature / area


def minimum_image(separation: NDArray[np.float64], box: float) -> NDArray[np.float64]:
    """Return separations wrapped to the nearest periodic image.

    Args:
        separation: Separation vectors, shape ``(..., 2)``, in metres.
        box: The side length of the square periodic box, in metres.

    Returns:
        The separations with each component brought into ``[-box / 2,
        box / 2]``, so each is the shortest of the periodic copies.
    """
    return separation - box * np.round(separation / box)


def dist(
    position: NDArray[np.float64], box: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the minimum-image distance and separation of every pair.

    Args:
        position: The position of each particle, shape ``(N, 2)``, in
            metres.
        box: The side length of the square periodic box, in metres.

    Returns:
        The distance between each pair, shape ``(M,)``, and the separation
        ``r_i - r_j`` of each pair, shape ``(M, 2)``, both in metres and in
        i < j pair order, where ``M = N (N - 1) / 2``.
    """
    i, j = np.triu_indices(position.shape[0], 1)
    separation = minimum_image(position[i] - position[j], box)
    return np.linalg.norm(separation, axis=1), separation
