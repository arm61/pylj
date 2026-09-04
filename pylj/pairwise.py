import numpy as np

from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN


def compute_force(particles, box_length, cut_off, constants, forcefield, mass):
    r"""Calculates the forces and therefore the accelerations on each of the
    particles in the simulation.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in metres.
    cut_off: float
        The separation beyond which the pair energy and force are taken to be
        zero, in metres.
    constants: float, array_like
        The constants associated with the particular forcefield used, e.g. for
        the class forcefields.lennard_jones, these are [A, B]
    forcefield: class
        The particular forcefield to be used to find the energy and forces.
    mass: float
        The mass of the particle being simulated (units of atomic mass units).

    Returns
    -------
    util.particle_dt, array_like
        Information about particles, with updated accelerations.
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
    distances, dx, dy = dist(
        particles["xposition"], particles["yposition"], box_length
    )
    i, j = np.triu_indices(particles.size, 1)
    types = np.array(particles["types"], dtype=int)
    lower = np.minimum(types[i], types[j])
    upper = np.maximum(types[i], types[j])
    forces = np.zeros(distances.size)
    energies = np.zeros(distances.size)
    # A pair of types 0 and 1 is the same pair as 1 and 0, so evaluate each
    # unordered pair once, on only the distances belonging to it.
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
    """Add the accelerations from the pair forces to each particle.

    The accelerations already on the particles are added to, so the caller
    zeroes them first. The pair arrays are in i < j order, as returned by
    dist.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    f: float, array_like
        The force on each pair of particles, in Newtons.
    m: float
        Mass of the particles, in kilograms.
    dx: float, array_like
        The x-dimension component of each pair separation, x_i - x_j, in
        metres.
    dy: float, array_like
        The y-dimension component of each pair separation, y_i - y_j, in
        metres.
    dr: float, array_like
        The distance between each pair of particles, in metres.

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
        The force between each pair of particles, in Newtons.
    box_length: float
        Length of a single dimension of the simulation square, in metres.
    number_of_particles: int
        The number of particles in the simulation.
    temperature: float
        Instantaneous temperature of the simulation, in Kelvin.

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
