from __future__ import division
import numpy as np
from pylj import util

def compute_forces(particles, box_length, cut_off):
    """Calculates the forces and therefore the accelerations on each of the particles in the simulation. This uses a
    12-6 Lennard-Jones potential model for Argon with values:

    - A = 1.363e-134 J m :math:`^{12}`
    - B = 9.273e-78 J m :math:`^6`

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    cut_off: float
        The distance greater than which the forces between particles is taken as zero.

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
    particles['xacceleration'] = np.zeros(particles['xacceleration'].size)
    particles['yacceleration'] = np.zeros(particles['yacceleration'].size)
    pairs = int((particles['xacceleration'].size - 1) * particles['xacceleration'].size / 2)
    forces = np.zeros(pairs)
    distances = np.zeros(pairs)
    energies = np.zeros(pairs)
    k = 0
    A = 1.363e-134  # joules / metre ^ {12}
    B = 9.273e-78  # joules / meter ^ {6}
    atomic_mass_unit = 1.660539e-27  # kilograms
    mass_of_argon_amu = 39.948  # amu
    mass_of_argon = mass_of_argon_amu * atomic_mass_unit # kilograms
    for i in range(0, particles['xposition'].size-1):
        for j in range(i+1, particles['xposition'].size):
            dx = particles['xposition'][i] - particles['xposition'][j]
            dy = particles['yposition'][i] - particles['yposition'][j]
            dx = util.pbc_correction(dx, box_length)
            dy = util.pbc_correction(dy, box_length)
            dr = separation(dx, dy)
            distances[k] = dr
            if dr <= cut_off:
                f = lennard_jones_force(A, B, dr)
                forces[k] = f
                e = lennard_jones_energy(A, B, dr)
                energies[k] = e
                particles = update_accelerations(particles, f, mass_of_argon, dx, dy, dr, i, j)
            else:
                forces[k] = 0.
                energies[k] = 0.
            k += 1
    return particles, distances, forces, energies


def separation(dx, dy):
    """Calculate the distance in 2D space.

    Parameters
    ----------
    dx: float
        Vector in the x dimension
    dy: float
        Vector in the y dimension

    Returns
    float:
        Magnitude of the 2D vector.
    """
    return np.sqrt(dx * dx + dy * dy)


def update_accelerations(particles, f, m, dx, dy, dr, i, j):
    """Update the acceleration arrays of particles.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    f: float
        The force on the pair of particles.
    m: float
        Mass of the particles.
    dx: float
        Distance between the particles in the x dimension.
    dy: float
        Distance between the particles in the y dimension.
    dr: float
        Distance between the particles.
    i: int
        Particle index 1.
    j: int
        Particle index 2.

    Returns
    -------
    util.particle_dt, array_like
        Information about the particles with updated accelerations.
    """
    particles['xacceleration'][i] += second_law(f, m, dx, dr)
    particles['yacceleration'][i] += second_law(f, m, dy, dr)
    particles['xacceleration'][j] -= second_law(f, m, dx, dr)
    particles['yacceleration'][j] -= second_law(f, m, dy, dr)
    return particles


def second_law(f, m, d1, d2):
    """Newton's second law of motion to get the acceleration of the particle in a given dimension.

    Parameters
    ----------
    f: float
        The force on the pair of particles.
    m: float
        Mass of the particle.
    d1: float
        Distance between the particles in a single dimension.
    d2: float
        Distance between the particles across all dimensions.

    Returns
    -------
    float:
        Acceleration of the particle in a given dimension.
    """
    return (f * d1 / d2) / m


def lennard_jones_energy(A, B, dr):
    """Calculate the energy of a pair of particles at a given distance.

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
    return A * np.power(dr, -12) - B * np.power(dr, -6)


def lennard_jones_force(A, B, dr):
    """Calculate the force between a pair of particles at a given distance.

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
    return 12 * A * np.power(dr, -13) - 6 * B * np.power(dr, -7)


def compute_energy(particles, box_length, cut_off):
    """Calculates the total energy of the simulation. This uses a
    12-6 Lennard-Jones potential model for Argon with values:

    - A = 1.363e-134 J m :math:`^{12}`
    - B = 9.273e-78 J m :math:`^6`

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    cut_off: float
        The distance greater than which the energies between particles is taken as zero.

    Returns
    -------
    util.particle_dt, array_like
        Information about particles, with updated accelerations and forces.
    float, array_like
        Current distances between pairs of particles in the simulation.
    float, array_like
        Current energies between pairs of particles in the simulation.
    """
    pairs = int((particles['xacceleration'].size - 1) * particles['xacceleration'].size / 2)
    distances = np.zeros(pairs)
    energies = np.zeros(pairs)
    k = 0
    A = 1.363e-134  # joules / metre ^ {12}
    B = 9.273e-78  # joules / meter ^ {6}
    for i in range(0, particles['xposition'].size-1):
        for j in range(i+1, particles['xposition'].size):
            dx = particles['xposition'][i] - particles['xposition'][j]
            dy = particles['yposition'][i] - particles['yposition'][j]
            dx = util.pbc_correction(dx, box_length)
            dy = util.pbc_correction(dy, box_length)
            dr = separation(dx, dy)
            distances[k] = dr
            if dr <= cut_off:
                e = lennard_jones_energy(A, B, dr)
                energies[k] = e
            else:
                energies[k] = 0.
            k += 1
    return distances, energies


def calculate_pressure(particles, box_length, temperature, cut_off):
    r"""Calculates the instantaneous pressure of the simulation cell, found with the following relationship:

    .. math::
        p = \langle \rho k_b T \rangle + \bigg\langle \frac{1}{3V}\sum_{i}\sum_{j<i} \mathbf{r}_{ij}\mathbf{f}_{ij} \bigg\rangle

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    temperature: float
        Instantaneous temperature of the simulation.
    cut_off: float
        The distance greater than which the forces between particles is taken as zero.

    Returns
    -------
    float:
        Instantaneous pressure of the simulation.
    """
    pres = 0.
    A = 1.363e-134  # joules / metre ^ {12}
    B = 9.273e-78  # joules / meter ^ {6}
    for i in range(0, particles['xposition'].size - 1):
        for j in range(i+1, particles['xposition'].size):
            dx = particles['xposition'][i] - particles['xposition'][j]
            dy = particles['yposition'][i] - particles['yposition'][j]
            dx = util.pbc_correction(dx, box_length)
            dy = util.pbc_correction(dy, box_length)
            dr = separation(dx, dy)
            if dr <= cut_off:
                f = lennard_jones_force(A, B, dr)
                pres += f * dr
    boltzmann_constant = 1.3806e-23 # joules / kelvin
    pres = 1. / (2 * box_length * box_length) * pres + (particles['xposition'].size / (box_length * box_length)
                                                        * boltzmann_constant * temperature)
    return pres


def heat_bath(particles, temperature_sample, bath_temp):
    r"""Rescales the velocities of the particles in the system to control the temperature of the simulation. Thereby
    allowing for an NVT ensemble. The velocities are rescaled according the following relationship,

    .. math::
        v_{\text{new}} = v_{\text{old}} \times \sqrt{\frac{T_{\text{desired}}}{\bar{T}}}

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    temperature_sample: float, array_like
        The temperature at each timestep in the simulation.
    bath_temp: float
        The desired temperature of the simulation.

    Returns
    -------
    util.particle_dt, array_like
        Information about the particles with new, rescaled velocities.
    """
    average_temp = np.average(temperature_sample)
    particles['xvelocity'] = particles['xvelocity'] * np.sqrt(bath_temp / average_temp)
    particles['yvelocity'] = particles['yvelocity'] * np.sqrt(bath_temp / average_temp)
    return particles
