import numpy as np
try:
    from pylj import comp as heavy
except ImportError:
    print("WARNING, using slow force and energy calculations")
    from pylj import pairwise as heavy


def initialise(number_of_particles, temperature, box_length, init_conf, timestep_length=1e-14):
    """Initialise the particle positions (this can be either as a square or random arrangement), velocities (based on
    the temperature defined, and calculate the initial forces/accelerations.

    Parameters
    ----------
    number_of_particles: int
        Number of particles to simulate.
    temperature: float
        Initial temperature of the particles, in Kelvin.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    init_conf: string, optional
        The way that the particles are initially positioned. Should be one of:
        - 'square'
        - 'random'
    timestep_length: float (optional)
        Length for each Velocity-Verlet integration step, in seconds.

    Returns
    -------
    System
        System information.
    """
    from pylj import util
    system = util.System(number_of_particles, temperature, box_length, init_conf=init_conf,
                         timestep_length=timestep_length)
    v = np.random.rand(system.particles.size, 2) - 0.5
    v2sum = np.average(np.square(v))
    v = (v - np.average(v)) * np.sqrt(2 * system.init_temp / v2sum)
    system.particles['xvelocity'] = v[:, 0]
    system.particles['yvelocity'] = v[:, 1]
    return system

def initialize(number_particles, temperature, box_length, init_conf, timestep_length=1e-14):
    """Maps to the md.initialise function to account for US english spelling.
    """
    a = initialise(number_particles, temperature, box_length, init_conf, timestep_length)
    return a


def velocity_verlet(particles, timestep_length, box_length, cut_off):
    """Uses the Velocity-Verlet integrator to move forward in time. The

    Updates the particles positions and velocities in terms of the Velocity Verlet algorithm. Also calculates the
    instanteous temperature, pressure, and force and appends these to the appropriate system array.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    timestep_length: float
        Length for each Velocity-Verlet integration step, in seconds.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.

    Returns
    -------
    util.particle_dt, array_like:
        Information about the particles, with new positions and velocities.
    """
    xposition_store = list(particles['xposition'])
    yposition_store = list(particles['yposition'])
    [particles['xposition'], particles['yposition']] = update_positions([particles['xposition'],
                                                                         particles['yposition']],
                                                                        [particles['xvelocity'],
                                                                         particles['yvelocity']],
                                                                        [particles['xacceleration'],
                                                                         particles['yacceleration']], timestep_length,
                                                                        box_length)
    xacceleration_store = list(particles['xacceleration'])
    yacceleration_store = list(particles['yacceleration'])
    particles, distances, forces, energies = heavy.compute_forces(particles, box_length, cut_off)
    [particles['xvelocity'], particles['yvelocity']] = update_velocities([particles['xvelocity'],
                                                                          particles['yvelocity']],
                                                                         [xacceleration_store, yacceleration_store],
                                                                         [particles['xacceleration'],
                                                                          particles['yacceleration']], timestep_length)
    particles['xprevious_position'] = xposition_store
    particles['yprevious_position'] = yposition_store
    return particles


def sample(particles, box_length, initial_particles, system):
    """Sample parameters of interest in the simulation.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    initial_particles: util.particle_dt, array-like
        Information about the initial particle conformation.
    system: System
        Details about the whole system

    Returns
    -------
    System:
        Details about the whole system, with the new temperature, pressure, msd, and force appended to the appropriate
        arrays.
    """
    temperature_new = calculate_temperature(particles)
    system.temperature_sample = np.append(system.temperature_sample, temperature_new)
    pressure_new = heavy.calculate_pressure(particles, box_length, temperature_new, system.cut_off)
    msd_new = calculate_msd(particles, initial_particles, box_length)
    system.pressure_sample = np.append(system.pressure_sample, pressure_new)
    system.force_sample = np.append(system.force_sample, np.sum(system.forces))
    system.energy_sample = np.append(system.energy_sample, np.sum(system.energies))
    system.msd_sample = np.append(system.msd_sample, msd_new)
    return system


def calculate_msd(particles, initial_particles, box_length):
    """Determines the mean squared displacement of the particles in the system.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    initial_particles: util.particle_dt, array_like
        Information about the initial state of the particles.
    box_length: float
        Size of the cell vector.

    Returns
    -------
    float:
        Mean squared deviation for the particles at the given timestep.
    """
    dx = particles['xposition'] - initial_particles['xposition']
    dy = particles['yposition'] - initial_particles['yposition']
    for i in range(0, particles['xposition'].size):
        if np.abs(dx[i]) > 0.5 * box_length:
            dx[i] *= 1 - box_length / np.abs(dx[i])
        if np.abs(dy[i]) > 0.5 * box_length:
            dy[i] *= 1 - box_length / np.abs(dy[i])
    dr = np.sqrt(dx * dx + dy * dy)
    return np.average(dr ** 2)


def update_positions(positions, velocities, accelerations, timestep_length, box_length):
    """Update the particle positions using the Velocity-Verlet integrator.

    Parameters
    ----------
    positions: (2, N) array_like
        Where N is the number of particles, and the first row are the x positions and the second row the y positions.
    velocities: (2, N) array_like
        Where N is the number of particles, and the first row are the x velocities and the second row the y velocities.
    accelerations: (2, N) array_like
        Where N is the number of particles, and the first row are the x accelerations and the second row the y
        accelerations.
    timestep_length: float
        Length for each Velocity-Verlet integration step, in seconds.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.

    Returns
    -------
    (2, N) array_like:
        Updated positions.
    """
    positions[0] += velocities[0] * timestep_length + 0.5 * accelerations[0] * timestep_length * timestep_length
    positions[1] += velocities[1] * timestep_length + 0.5 * accelerations[1] * timestep_length * timestep_length
    positions[0] = positions[0] % box_length
    positions[1] = positions[1] % box_length
    return [positions[0], positions[1]]


def update_velocities(velocities, accelerations_old, accelerations_new, timestep_length):
    """Update the particle velocities using the Velocity-Verlet algoritm.

    Parameters
    ----------
    velocities: (2, N) array_like
        Where N is the number of particles, and the first row are the x velocities and the second row the y velocities.
    accelerations: (2, N) array_like
        Where N is the number of particles, and the first row are the x accelerations and the second row the y
        accelerations.
    timestep_length: float
        Length for each Velocity-Verlet integration step, in seconds.

    Returns
    -------
    (2, N) array_like:
        Updated velocities.
    """
    velocities[0] += 0.5 * (accelerations_old[0] + accelerations_new[0]) * timestep_length
    velocities[1] += 0.5 * (accelerations_old[1] + accelerations_new[1]) * timestep_length
    return [velocities[0], velocities[1]]


def calculate_temperature(particles):
    """Determine the instantaneous temperature of the system.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.

    Returns
    -------
    float:
        Calculated instantaneous simulation temperature.
    """
    k = 0
    for i in range(0, particles['xposition'].size):
        v = np.sqrt(particles['xvelocity'][i] * particles['xvelocity'][i] + particles['yvelocity'][i] *
                    particles['yvelocity'][i])
        boltzmann_constant = 1.3806e-23  # joules/kelvin
        atomic_mass_unit = 1.660539e-27 # kilograms
        mass_of_argon_amu = 39.948 # amu
        mass_of_argon = mass_of_argon_amu * atomic_mass_unit # kilograms
        k += mass_of_argon * v * v / (boltzmann_constant * 2 * particles['xposition'].size)
    return k

