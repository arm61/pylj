import numpy as np
from pylj import comp, util


def initialise(number_of_particles, temperature, box_length, init_conf):
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

    Returns
    -------
    System
        System information.
    """
    system = util.System(number_of_particles, temperature, box_length, init_conf=init_conf)
    system.particles['xvelocity'] = 0
    system.particles['yvelocity'] = 0
    return system


def sample(particles, box_length, total_energy, system):
    """Sample parameters of interest in the simulation.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    total_energy: float
        The total system energy.
    system: System
        Details about the whole system

    Returns
    -------
    System:
        Details about the whole system, with the new temperature, pressure, msd, and force appended to the appropriate
        arrays.
    """
    temperature_new = util.calculate_temperature(particles)
    system.temperature_sample = np.append(system.temperature_sample, temperature_new)
    pressure_new = comp.calculate_pressure(particles, box_length, temperature_new)
    system.pressure_sample = np.append(system.pressure_sample, pressure_new)
    system.energy_sample = np.append(system.energy_sample, total_energy)
    return system


def select_random_particle(particles):
    """Selects a random particle from the system and return its index and current position.

    Parameters
    ----------
    particles: util.particle.dt, array_like
        Information about the particles.

    Returns
    -------
    int:
        Index of the random particle that is selected.
    float, array_like:
        The current position of the chosen particle.
    """
    random_particle = np.random.randint(0, particles.size-1)
    position_store = [particles['xposition'][random_particle], particles['yposition'][random_particle]]
    return random_particle, position_store


def get_new_particle(particles, random_particle, box_length):
    """Generates a new position for the particle.

    Parameters
    ----------
    particles: util.particle.dt, array_like
        Information about the particles.
    random_particle: int
        Index of the random particle that is selected.
    box_length: float
        Length of a single dimension of the simulation square.

    Returns
    -------
    util.particle.dt, array_like
        Information about the particles, updated to account for the change of selected particle position.
    """
    particles['xposition'][random_particle] = np.random.uniform(0, box_length)
    particles['yposition'][random_particle] = np.random.uniform(0, box_length)
    return particles


def accept(new_energy):
    """Accept the move.

    Parameters
    ----------
    new_energy: float
        A new total energy for the system.

    Returns
    -------
    float:
        A new total energy for the system.
    """
    return new_energy


def reject(position_store, particles, random_particle):
    """Reject the move and return the particle to the original place.

    Parameters
    ----------
    position_store: float, array_like
        The x and y positions previously held by the particle that has moved.
    particles: util.particle.dt, array_like
        Information about the particles.
    random_particle: int
        Index of the random particle that is selected.

    Returns
    -------
    util.particle.dt, array_like
        Information about the particles, with the particle returned to the original position
    """
    particles['xposition'][random_particle] = position_store[0]
    particles['yposition'][random_particle] = position_store[1]
    return particles


def metropolis(temperature, old_energy, new_energy):
    """Determines if the move is accepted or rejected based on the metropolis condition.

    Parameters
    ----------
    temperature: float
        Simulation temperature.
    old_energy: float
        The total energy of the simulation in the previous configuration.
    new_energy: float
        The total energy of the simulation in the current configuration.

    Returns
    -------
    bool
        True if the move should be accepted.
    """
    beta = 1 / (1.3806e-23 * temperature)
    energy_difference = new_energy - old_energy
    metropolis_factor = np.exp(-beta * energy_difference)
    n = np.random.rand()
    if n < metropolis_factor:
        return True
    else:
        return False