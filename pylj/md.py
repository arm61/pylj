import numpy as np
from pylj import force, sample, util


def initialise(number_of_particles, temperature, timestep_length, init_conf):
    """Initial particle positions (simple square arrangment), velocities and get initial forces/accelerations.

    Parameters
    ----------
    number_of_particles: int
        Number of particles in the system.
    temperature: float
        Temperature of the system.
    init_conf: string, optional
        Selection for the way the particles are initially populated. Should be one of
        - 'square'
        - 'random'

    Returns
    -------
    System
        System information.
    """
    system = util.System(number_of_particles, temperature, 16, timestep_length, init_conf=init_conf)
    v = np.sqrt(2 * system.init_temp)
    theta = 2 * np.pi * np.random.randn(system.particles.size)
    system.particles['xvelocity'] = v * np.cos(theta)
    system.particles['yvelocity'] = v * np.sin(theta)
    system = force.compute_forces(system)
    system.temp_sum = 0.
    return system

def reset_histogram(system):
    """Reset the velocity histogram.

    Parameters
    ----------
    system: System
        System parameters.

    Returns
    -------
    System
        System parameters with the velocity bins and temperature summation set to zero.
    """
    system.velocity_bins[:]= 0
    system.temp_sum = 0

def velocity_verlet(system):
    """Update the positions, velocities, get temperature and pressure.

    Parameters
    ----------
    particles: Particle array
        All particles in the system.
    system: System
        Whole system information.

    Returns
    -------
    Particle array:
        Particles with updated positions and velocities.
    System:
        Whole system information with new temperature and pressure.
    """
    xposition_store = system.particles['xposition']
    yposition_store = system.particles['yposition']
    system.particles['xposition'], system.particles['yposition'] = update_pos_vv([system.particles['xposition'], system.particles['yposition']], [system.particles['xvelocity'], system.particles['yvelocity']], [system.particles['xacceleration'], system.particles['yacceleration']], system.timestep_length, system.box_length)
    system.particles['xvelocity'], system.particles['yvelocity'] = update_velocities_vv([system.particles['xvelocity'], system.particles['yvelocity']], [system.particles['xacceleration'], system.particles['yacceleration']], system.timestep_length)
    system.particles['xprevious_position'] = xposition_store
    system.particles['yprevious_position'] = yposition_store

    temp = util.calculate_temperature(system.number_of_particles, system.particles)
    pres = force.calculate_pressure(system.number_of_particles, system.particles, system.forces, system.box_length)
    system.temperature = np.append(system.temperature, temp)
    system.pressure = np.append(system.pressure, pres)
    system.force = np.append(system.force, np.sum(system.forces))
    return system

def clear_accelerations(particles):
    """Reset all particle accelerations to 0.

    Parameters
    ----------
    particles: Particle array
        The particles in the system.

    Returns
    -------
    Particle array
        The particles in the system now with zero acceleration.
    """
    particles['xacceleration'] = np.zeros(particles.size)
    particles['yacceleration'] = np.zeros(particles.size)
    return particles

def initialise_from_em(particles, system, temperature, timestep_length):
    system.temperature = temperature
    system.timestep_length = timestep_length
    v = np.sqrt(2 * system.temperature)
    for i in range(0, system.number_of_particles):
        theta = 2 * np.pi * np.random.randn()
        particles[i].xvel = v * np.cos(theta)
        particles[i].yvel = v * np.sin(theta)
    particles, system = force.compute_forces(particles, system)
    system = reset_histogram(system)
    return particles, system


def update_pos_verlet(particle, system):
    """Update the positions of a given particle based on the integrator.

    Parameters
    ----------
    particle: Particle
        A particle in the system.
    system: System
        Whole system information.

    Returns
    -------
    Particle
        Particle with updated positions.
    """
    particle.xpos = particle.xpos - particle.xpos_prev + particle.xacc * system.timestep_length * system.timestep_length
    particle.ypos = particle.ypos - particle.ypos_prev + particle.yacc * system.timestep_length * system.timestep_length
    particle.xpos = particle.xpos % system.box_length
    particle.ypos = particle.ypos % system.box_length
    return particle


def update_pos_vv(positions, velocities, accelerations, timestep_length, box_length):
    """Update the positions of a given particle based on the integrator.

    Parameters
    ----------
    particle: Particle
        A particle in the system.
    system: System
        Whole system information.

    Returns
    -------
    Particle
        Particle with updated positions.
    """
    positions[0] += velocities[0] * timestep_length + 0.5 * accelerations[0] * timestep_length * timestep_length
    positions[1] += velocities[1] * timestep_length + 0.5 * accelerations[1] * timestep_length * timestep_length
    positions[0] = positions[0] % box_length
    positions[1] = positions[1] % box_length
    return positions[0], positions[1] 


def update_velocities_verlet(particle, system):
    """Update the velocities of a given particles based on the acceleration.

    Parameters
    ----------
    particle: Particle
        A particle in the system.
    system: System
        Whole system information.

    Returns
    -------
    Particle
        Particle with updated velocities.
    """
    particle.xvel = (particle.xpos - particle.xpos_prev) / (2 * system.timestep_length)
    particle.yvel = (particle.ypos - particle.ypos_prev) / (2 * system.timestep_length)
    return particle


def update_velocities_vv(velocities, accelerations, timestep_length):
    """Update the velocities of a given particles based on the acceleration.

    Parameters
    ----------
    particle: Particle
        A particle in the system.
    system: System
        Whole system information.

    Returns
    -------
    Particle
        Particle with updated velocities.
    """
    velocities[0] += 0.5 * accelerations[0] * timestep_length
    velocities[1] += 0.5 * accelerations[1] * timestep_length
    return velocities[0], velocities[1] 

def verlet(particles, system):
    """Update the positions, velocities, get temperature and pressure.

    Parameters
    ----------
    particles: Particle array
        All particles in the system.
    system: System
        Whole system information.

    Returns
    -------
    Particle array:
        Particles with updated positions and velocities.
    System:
        Whole system information with new temperature and pressure.
    """
    system = reset_histogram(system)
    system.step += 1
    for i in range(0, system.number_of_particles):
        particles[i] = update_pos_verlet(particles[i], system)
        particles[i] = update_velocities_verlet(particles[i], system)
    particles, system = util.calculate_temperature(particles, system)
    system = util.calculate_pressure(system)
    return particles, system

def update_velocity_bins(particle, velocity_bins, max_vel):
    """Updates the velocity bins. It is not clear if this is completely required if the velocity bins are not
    being plotted.
    
    Parameters
    ----------
    particle: Particle
        A particle in the system.
    system: System
        Whole system information.
    
    Returns
    -------
    System
        Whole system information with the velocity bin for particle updated.
    float
        The total velocity for a given particle.
    """
    v = np.sqrt(particle['xvelocity'] * particle['xvelocity'] + particle['yvelocity'] * particle['yvelocity'])
    bin_s = int(len(velocity_bins) * v / max_vel)
    if bin_s < 0:
        bin_s = 0
    if bin_s >= len(velocity_bins):
        bin_s = len(velocity_bins) - 1
    velocity_bins[bin_s] += 1
    return velocity_bins, v

