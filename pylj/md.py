import numpy as np
from pylj import comp, util


def initialise(number_of_particles, temperature, box_length, init_conf, timestep_length=5e-3):
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
    system = util.System(number_of_particles, temperature, box_length, init_conf=init_conf,
                         timestep_length=timestep_length)
    v = np.sqrt(2 * system.init_temp)
    theta = 2 * np.pi * np.random.randn(system.particles.size)
    system.particles['xvelocity'] = v * np.cos(theta)
    system.particles['yvelocity'] = v * np.sin(theta)
    system.particles, system.distances, system.forces = comp.compute_forces(system.particles, system.distances,
                                                                            system.forces, system.box_length)
    return system

def velocity_verlet(particles, timestep_length, box_length):
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
    xposition_store = particles['xposition']
    yposition_store = particles['yposition']
    [particles['xposition'], particles['yposition']] = update_positions([particles['xposition'],
                                                                         particles['yposition']],
                                                                        [particles['xvelocity'],
                                                                         particles['yvelocity']],
                                                                        [particles['xacceleration'],
                                                                         particles['yacceleration']], timestep_length,
                                                                        box_length)
    [particles['xvelocity'], particles['yvelocity']] = update_velocities([particles['xvelocity'], particles['yvelocity']],
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
    temperature_new = util.calculate_temperature(particles)
    pressure_new = comp.calculate_pressure(particles, box_length, temperature_new)
    msd_new = util.calculate_msd(particles, initial_particles, box_length)
    system.temperature_sample = np.append(system.temperature_sample, temperature_new)
    system.pressure_sample = np.append(system.pressure_sample, pressure_new)
    system.force_sample = np.append(system.force_sample, np.sum(system.forces))
    system.msd_sample = np.append(system.msd_sample, msd_new)
    return system

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

def update_velocities(velocities, accelerations, timestep_length):
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
    velocities[0] += 0.5 * accelerations[0] * timestep_length
    velocities[1] += 0.5 * accelerations[1] * timestep_length
    return [velocities[0], velocities[1]]

