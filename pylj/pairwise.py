import numpy as np

from pylj import pairwise as heavy
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN


def compute_force(particles, box_length, cut_off, constants, forcefield, mass):
    r"""Calculates the forces and therefore the accelerations on each of the
    particles in the simulation.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    cut_off: float
        The distance greater than which the forces between particles is taken
        as zero.
    constants: float, array_like (optional)
        The constants associated with the particular forcefield used, e.g. for
        the function forcefields.lennard_jones, theses are [A, B]
    forcefield: function (optional)
        The particular forcefield to be used to find the energy and forces.
    mass: float (optional)
        The mass of the particle being simulated (units of atomic mass units).

    Returns
    -------
    util.particle_dt, array_like
        Information about particles, with updated accelerations and forces.
    float, array_like
        Current distances between pairs of particles in the simulation.
    float, array_like
        Current forces between pairs of particles in the simulation.
    float, array_like
        Current energies between pairs of particles in the simulation.
    """
    particles["xacceleration"] = np.zeros(particles["xacceleration"].size)
    particles["yacceleration"] = np.zeros(particles["yacceleration"].size)
    mass_kg = mass * ATOMIC_MASS_UNIT
    distances, dx, dy = heavy.dist(
        particles["xposition"], particles["yposition"], box_length
    )
    i, j = np.triu_indices(particles.size, 1)
    types = np.array(particles["types"], dtype=int)
    lower = np.minimum(types[i], types[j])
    upper = np.maximum(types[i], types[j])
    forces = np.zeros(distances.size)
    energies = np.zeros(distances.size)
    # '0,1' and '1,0' are the same pair of types, so evaluate each unordered
    # pair once, on only the distances belonging to it.
    for type_1, type_2 in sorted(set(zip(lower.tolist(), upper.tolist(), strict=True))):
        mask = (lower == type_1) & (upper == type_2)
        ff = forcefield(np.array(constants[type_1]))
        if type_1 != type_2:
            ff.mixing(np.array(constants[type_2]))
        forces[mask] = ff.force(distances[mask])
        energies[mask] = ff.energy(distances[mask])
    forces[distances > cut_off] = 0.0
    energies[distances > cut_off] = 0.0
    particles = update_accelerations(particles, forces, mass_kg, dx, dy, distances)
    return particles, distances, forces, energies


def update_accelerations(particles, f, m, dx, dy, dr):
    """Set the accelerations on each particle from the pair forces.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    f: float, array_like
        The force on each pair of particles.
    m: float
        Mass of the particles, in kilograms.
    dx: float, array_like
        The x-dimension component of each pair separation.
    dy: float, array_like
        The y-dimension component of each pair separation.
    dr: float, array_like
        The distance between each pair of particles.

    Returns
    -------
    util.particle_dt, array_like
        The particles with their accelerations accumulated from the pairs.
    """
    i, j = np.triu_indices(particles.size, 1)
    ax = f * dx / dr / m
    ay = f * dy / dr / m
    # each pair accelerates particle i one way and particle j the other.
    np.add.at(particles["xacceleration"], i, ax)
    np.add.at(particles["xacceleration"], j, -ax)
    np.add.at(particles["yacceleration"], i, ay)
    np.add.at(particles["yacceleration"], j, -ay)
    return particles


def lennard_jones_energy(A, B, dr):
    """pairwise.lennard_jones_energy has been deprecated, please use
    forcefields.lennard_jones instead

    Calculate the energy of a pair of particles at a given distance.

    Parameters
    ----------
    A: float
        The value of the A parameter for the Lennard-Jones potential.
    B: float
        The value of the B parameter for the Lennard-Jones potential.
    dr: float
        The distance between the two particles.

    Returns
    -------
    float:
        The potential energy between the two particles.
    """
    print(
        "pairwise.lennard_jones_energy has been deprecated, please use "
        "forcefields.lennard_jones instead"
    )
    return A * np.power(dr, -12) - B * np.power(dr, -6)


def lennard_jones_force(A, B, dr):
    """pairwise.lennard_jones_energy has been deprecated, please use
    forcefields.lennard_jones with force=True instead

    Calculate the force between a pair of particles at a given distance.

    Parameters
    ----------
    A: float
        The value of the A parameter for the Lennard-Jones potential.
    B: float
        The value of the B parameter for the Lennard-Jones potential.
    dr: float
        The distance between the two particles.

    Returns
    -------
    float:
        The force between the two particles.
    """
    print(
        "pairwise.lennard_jones_energy has been deprecated, please use "
        "forcefields.lennard_jones with force=True instead"
    )
    return 12 * A * np.power(dr, -13) - 6 * B * np.power(dr, -7)


def calculate_pressure(
    particles, box_length, temperature, cut_off, constants, forcefield, mass
):
    r"""Calculates the instantaneous pressure of the simulation cell, found
    with the following relationship:
    .. math::
        p = \langle \rho k_b T \rangle + \bigg\langle \frac{1}{3V}\sum_{i}
        \sum_{j<i} \mathbf{r}_{ij}\mathbf{f}_{ij} \bigg\rangle

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    temperature: float
        Instantaneous temperature of the simulation.
    cut_off: float
        The distance greater than which the forces between particles is taken
        as zero.
    constants: float, array_like (optional)
        The constants associated with the particular forcefield used, e.g. for
        the function forcefields.lennard_jones, theses are [A, B]
    forcefield: function (optional)
        The particular forcefield to be used to find the energy and forces.
    mass: float (optional)
        The mass of the particle being simulated (units of atomic mass units).

    Returns
    -------
    float:
        Instantaneous pressure of the simulation.
    """
    particles, distances, forces, energies = heavy.compute_force(
        particles, box_length, cut_off, constants, forcefield, mass
        )
    pres = np.sum(forces * distances)
    pres = 1.0 / (2 * box_length * box_length) * pres + (
        particles["xposition"].size
        / (box_length * box_length)
        * BOLTZMANN
        * temperature
    )
    return pres


def dist(xposition, yposition, box_length):
    """Return the minimum-image distances between every pair of particles.

    Parameters
    ----------
    xposition: float, array_like (N)
        The x-dimension positions of the N particles.
    yposition: float, array_like (N)
        The y-dimension positions of the N particles.
    box_length: float
        The box length of the simulation cell.

    Returns
    -------
    dr: float, array_like (N (N - 1) / 2)
        The distance between each pair of particles, in i < j pair order.
    dx: float, array_like (N (N - 1) / 2)
        The x-dimension component of each pair separation.
    dy: float, array_like (N (N - 1) / 2)
        The y-dimension component of each pair separation.
    """
    i, j = np.triu_indices(xposition.size, 1)
    dx = xposition[i] - xposition[j]
    dy = yposition[i] - yposition[j]
    dx -= box_length * np.round(dx / box_length)
    dy -= box_length * np.round(dy / box_length)
    dr = np.hypot(dx, dy)
    return dr, dx, dy
