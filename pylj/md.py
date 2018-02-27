import numpy as np
from pylj import force, sample, util


def clear_accelerations(particles):
    """Reset the accelerations to 0.

    Parameters
    ----------
    particles: Particle array
        The particles in the system.

    Returns
    -------
    Particle array
        The particles in the system now with zero acceleration.
    """
    for i in range(0, len(particles)):
        particles[i].xacc = 0
        particles[i].yacc = 0
    return particles


def compute_acceleration(particles, system):
    """A pure python method for the calculation of forces and accelerations on each atom.

    Parameters
    ----------
    particles: Particle array
        The particles in the system.
    system: System
        System parameters.

    Returns
    -------
    Particle array
        The particles with the accelerations calculated.
    System
        System parameters, where the distances array has been updated.
    """
    system.distances = []
    particles = clear_accelerations(particles)
    for i in range(0, len(particles) - 1):
        for j in range(i + 1, len(particles)):
            dx = particles[i].xpos - particles[j].xpos
            dy = particles[i].ypos - particles[j].ypos
            dx = util.pbc_correction(dx, system.box_length)
            dy = util.pbc_correction(dy, system.box_length)
            dr = np.sqrt(dx * dx + dy * dy)
            system.distances = np.append(system.distances, dr)
            f = 48 * np.power(dr, -13.) - 24 * np.power(dr, -7.)
            particles[i].xacc += f * dx / dr
            particles[i].yacc += f * dy / dr
            particles[j].xacc -= f * dx / dr
            particles[j].yacc -= f * dy / dr
    return particles, system


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
    for i in range(0, len(system.vel_bins)):
        system.vel_bins[i] = 0
    system.temp_sum = 0
    system.step0 = system.step
    return system


def initialise(number_of_particles, temperature, timestep_length, arrangement):
    """Initial particle positions (simple square arrangment), velocities and get initial forces/accelerations.

    Parameters
    ----------
    number_of_particles: int
        Number of particles in the system.
    temperature: float
        Temperature of the system.
    arrangement: func
        A function to set the arrangement of the particles.

    Returns
    -------
    Particle array
        An array containing the initial particle positions, velocities and accelerations set.
    System
        System information.
    """
    system = util.System(number_of_particles, temperature, 16., timestep_length)
    particles = arrangement(system)
    v = np.sqrt(2 * system.temperature)
    for i in range(0, system.number_of_particles):
        theta = 2 * np.pi * np.random.randn()
        particles[i].xvel = v * np.cos(theta)
        particles[i].yvel = v * np.sin(theta)
    particles, system = force.compute_forces(particles, system)
    system = reset_histogram(system)
    return particles, system


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


def update_pos_vv(particle, system):
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
    particle.xpos += particle.xvel * system.timestep_length + 0.5 * particle.xacc * \
                                                                      system.timestep_length * system.timestep_length
    particle.ypos += particle.yvel * system.timestep_length + 0.5 * particle.yacc * \
                                                                      system.timestep_length * system.timestep_length
    particle.xpos = particle.xpos % system.box_length
    particle.ypos = particle.ypos % system.box_length
    return particle


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


def update_velocities_vv(particle, system):
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
    particle.xvel += 0.5 * particle.xacc * system.timestep_length
    particle.yvel += 0.5 * particle.yacc * system.timestep_length
    return particle


def update_velocity_bins(particle, system):
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
    v = np.sqrt(particle.xvel * particle.xvel + particle.yvel * particle.yvel)
    bin_s = int(len(system.vel_bins) * v / system.max_vel)
    if bin_s < 0:
        bin_s = 0
    if bin_s >= len(system.vel_bins):
        bin_s = len(system.vel_bins) - 1
    system.vel_bins[bin_s] += 1
    return system, v


def velocity_verlet(particles, system):
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
        position_store = [particles[i].xpos, particles[i].ypos]
        particles[i] = update_pos_vv(particles[i], system)
        particles[i] = update_velocities_vv(particles[i], system)
        particles[i].xpos_prev = position_store[0]
        particles[i].ypos_prev = position_store[1]
    particles, system = util.calculate_temperature(particles, system)
    system = util.calculate_pressure(system)
    return particles, system


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


def scale_velocities(particles, system):
    """Pure python for the velocity rescaling.

    Parameters
    ----------
    particles: Particle array
        All particles in the system.
    system: System
        Whole system information.

    Returns
    -------
    Particle array:
        All particles with velocities rescaled.
    """
    for i in range(0, len(particles)):
        particles[i].xvel *= 1 / np.average(system.temp_array)
        particles[i].yvel *= 1 / np.average(system.temp_array)
    return particles