import numpy as np

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
    particles['xaccleration'] = np.zeros(particles['xaccleration'].size)
    particles['yaccleration'] = np.zeros(particles['yaccleration'].size)
    pairs = int((particles['xaccleration'].size - 1) * particles['xaccleration'].size / 2)
    forces = np.zeros(pairs)
    distances = np.zeros(pairs)
    energies = np.zeros(pairs)
    k = 0
    A = 1.363e-134 # joules / metre ^ {12}
    B = 9.273e-78 # joules / meter ^ {6}
    atomic_mass_unit = 1.660539e-27 # kilograms
    mass_of_argon_amu = 39.948 # amu
    mass_of_argon = mass_of_argon_amu * atomic_mass_unit # kilograms
    for i in range(0, particles['xposition'].size-1):
        for j in range(i, particles['xposition'].size):
            dx = particles['xposition'][i] - particles['xposition'][j]
            dy = particles['yposition'][i] - particles['yposition'][j]
            if np.abs(dx) > 0.5 * box_length:
                dx *= 1 - box_length / np.abs(dx)
            if np.abs(dy) > 0.5 * box_length:
                dy *= 1 - box_length / np.abs(dy)
            dr = np.sqrt(dx * dx + dy * dy)
            distances[k] = dr
            if dr <= cut_off:
                inv_dr_1 = 1.0 / dr
                inv_dr_6 = np.power(inv_dr_1, 6)
                f = (12 * A * (inv_dr_1 * inv_dr_6 * inv_dr_6) - 6 * B * (inv_dr_1 * inv_dr_6))
                forces[k] = f
                e = (A * (inv_dr_6 * inv_dr_6) - B * inv_dr_6)
                energies[k] = e
                particles['xacceleration'][i] += (f * dx / dr) / mass_of_argon
                particles['yacceleration'][i] += (f * dy / dr) / mass_of_argon
                particles['xacceleration'][j] -= (f * dx / dr) / mass_of_argon
                particles['yacceleration'][j] -= (f * dy / dr) / mass_of_argon
            else:
                forces[k] = 0.
                energies[k] = 0.
            k += 1
    return particles, distances, forces, energies

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
    pairs = int((particles['xaccleration'].size - 1) * particles['xaccleration'].size / 2)
    distances = np.zeros(pairs)
    energies = np.zeros(pairs)
    k = 0
    A = 1.363e-134  # joules / metre ^ {12}
    B = 9.273e-78  # joules / meter ^ {6}
    for i in range(0, particles['xpositions']-1):
        for j in range(i, particles['xpositions']):
            dx = particles['xposition'][i] - particles['xposition'][j]
            dy = particles['yposition'][i] - particles['yposition'][j]
            if np.abs(dx) > 0.5 * box_length:
                dx *= 1 - box_length / np.abs(dx)
            if np.abs(dy) > 0.5 * box_length:
                dy *= 1 - box_length / np.abs(dy)
            dr = np.sqrt(dx * dx + dy * dy)
            distances[k] = dr
            if dr <= cut_off:
                inv_dr_1 = 1.0 / dr
                inv_dr_6 = np.power(inv_dr_1, 6)
                e = (A * (inv_dr_6 * inv_dr_6) - B * inv_dr_6)
                energies[k] = e
            else:
                energies[k] = 0.
            k += 1
    return particles, distances, energies

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
    for i in range(0, particles['xpositions'] - 1):
        for j in range(i, particles['xpositions']):
            dx = particles['xposition'][i] - particles['xposition'][j]
            dy = particles['yposition'][i] - particles['yposition'][j]
            if np.abs(dx) > 0.5 * box_length:
                dx *= 1 - box_length / np.abs(dx)
            if np.abs(dy) > 0.5 * box_length:
                dy *= 1 - box_length / np.abs(dy)
            dr = np.sqrt(dx * dx + dy * dy)
            distances[k] = dr
            if dr <= cut_off:
                inv_dr_1 = 1.0 / dr
                inv_dr_6 = np.power(inv_dr_1, 6)
                f = (12 * A * (inv_dr_1 * inv_dr_6 * inv_dr_6) - 6 * B * (inv_dr_1 * inv_dr_6))
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