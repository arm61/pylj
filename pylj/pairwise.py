from __future__ import division
import numpy as np
try:
    from pylj import comp as heavy
except ImportError:
    print("WARNING, using slow force and energy calculations")
    from pylj import pairwise as heavy


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
    particles['xacceleration'] = np.zeros(particles['xacceleration'].size)
    particles['yacceleration'] = np.zeros(particles['yacceleration'].size)
    pairs = int((particles['xacceleration'].size - 1) *
                particles['xacceleration'].size / 2)
    forces = np.zeros(pairs)
    distances = np.zeros(pairs)
    energies = np.zeros(pairs)
    atomic_mass_unit = 1.660539e-27  # kilograms
    mass_amu = mass  # amu
    mass_kg = mass_amu * atomic_mass_unit  # kilograms
    distances, dx, dy = heavy.dist(particles['xposition'],
                                   particles['yposition'], box_length)
    forces = forcefield(distances, constants, force=True)
    energies = forcefield(distances, constants, force=False)
    forces[np.where(distances > cut_off)] = 0.
    energies[np.where(distances > cut_off)] = 0.
    particles = update_accelerations(particles, forces, mass_kg, dx, dy,
                                     distances)
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


def update_accelerations(particles, f, m, dx, dy, dr):
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
    Returns
    -------
    util.particle_dt, array_like
        Information about the particles with updated accelerations.
    """
    k = 0
    for i in range(0, particles.size-1):
        for j in range(i + 1, particles.size):
            particles['xacceleration'][i] += second_law(f[k], m, dx[k], dr[k])
            particles['yacceleration'][i] += second_law(f[k], m, dy[k], dr[k])
            particles['xacceleration'][j] -= second_law(f[k], m, dx[k], dr[k])
            particles['yacceleration'][j] -= second_law(f[k], m, dy[k], dr[k])
            k += 1
    return particles


def second_law(f, m, d1, d2):
    """Newton's second law of motion to get the acceleration of the particle
    in a given dimension.
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
    print("pairwise.lennard_jones_energy has been deprecated, please use "
          "forcefields.lennard_jones instead")
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
    print("pairwise.lennard_jones_energy has been deprecated, please use "
          "forcefields.lennard_jones with force=True instead")
    return 12 * A * np.power(dr, -13) - 6 * B * np.power(dr, -7)


def compute_energy(particles, box_length, cut_off, constants, forcefield):
    r"""Calculates the total energy of the simulation.
    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    cut_off: float
        The distance greater than which the energies between particles is
        taken as zero.
    constants: float, array_like (optional)
        The constants associated with the particular forcefield used, e.g. for
        the function forcefields.lennard_jones, theses are [A, B]
    forcefield: function (optional)
        The particular forcefield to be used to find the energy and forces.
    Returns
    -------
    util.particle_dt, array_like
        Information about particles, with updated accelerations and forces.
    float, array_like
        Current distances between pairs of particles in the simulation.
    float, array_like
        Current energies between pairs of particles in the simulation.
    """
    pairs = int((particles['xacceleration'].size - 1) *
                particles['xacceleration'].size / 2)
    distances = np.zeros(pairs)
    energies = np.zeros(pairs)
    distances, dx, dy = heavy.dist(particles['xposition'],
                                   particles['yposition'], box_length)
    energies = forcefield(distances, constants, force=False)
    energies[np.where(distances > cut_off)] = 0.
    return distances, energies


def calculate_pressure(particles, box_length, temperature,
                       cut_off, constants, forcefield):
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
    Returns
    -------
    float:
        Instantaneous pressure of the simulation.
    """
    distances, dx, dy = heavy.dist(particles['xposition'],
                                   particles['yposition'], box_length)
    forces = forcefield(distances, constants, force=True)
    forces[np.where(distances > cut_off)] = 0.
    pres = np.sum(forces * distances)
    boltzmann_constant = 1.3806e-23  # joules / kelvin
    pres = 1. / (2 * box_length * box_length) * pres + (
        particles['xposition'].size / (box_length * box_length) *
        boltzmann_constant * temperature)
    return pres


def heat_bath(particles, temperature_sample, bath_temp):
    r"""Rescales the velocities of the particles in the system to control the
    temperature of the simulation. Thereby allowing for an NVT ensemble. The
    velocities are rescaled according the following relationship,
    .. math::
        v_{\text{new}} = v_{\text{old}} \times
        \sqrt{\frac{T_{\text{desired}}}{\bar{T}}}
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
    particles['xvelocity'] = particles['xvelocity'] * np.sqrt(bath_temp /
                                                              average_temp)
    particles['yvelocity'] = particles['yvelocity'] * np.sqrt(bath_temp /
                                                              average_temp)
    return particles


def dist(xposition, yposition, box_length):
    """Returns the distance array for the set of particles.
    Parameters
    ----------
    xpos: float, array_like (N)
        Array of length N, where N is the number of particles, providing the
        x-dimension positions of the particles.
    ypos: float, array_like (N)
        Array of length N, where N is the number of particles, providing the
        y-dimension positions of the particles.
    box_length: float
        The box length of the simulation cell.
    Returns
    -------
    drr float, array_like ((N - 1) * N / 2))
        The pairs of distances between the particles.
    dxr float, array_like ((N - 1) * N / 2))
        The pairs of distances between the particles, in only the x-dimension.
    dyr float, array_like ((N - 1) * N / 2))
        The pairs of distances between the particles, in only the y-dimension.
    """
    drr = np.zeros(int((xposition.size - 1) * xposition.size / 2))
    dxr = np.zeros(int((xposition.size - 1) * xposition.size / 2))
    dyr = np.zeros(int((xposition.size - 1) * xposition.size / 2))
    k = 0
    for i in range(0, xposition.size - 1):
        for j in range(i+1, xposition.size):
            dx = xposition[i] - xposition[j]
            dy = yposition[i] - yposition[j]
            dx = pbc_correction(dx, box_length)
            dy = pbc_correction(dy, box_length)
            dr = separation(dx, dy)
            drr[k] = dr
            dxr[k] = dx
            dyr[k] = dy
            k += 1
    return drr, dxr, dyr


def pbc_correction(position, cell):
    """Correct for the periodic boundary condition.
    Parameters
    ----------
    position: float
        Particle position.
    cell: float
        Cell vector.
    Returns
    -------
    float:
        Corrected particle position."""
    if np.abs(position) > 0.5 * cell:
        position *= 1 - cell / np.abs(position)
    return position
