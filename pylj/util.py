import numpy as np
from pylj import md, force


class System:
    """Whole system.

    This class stores a large amount of information about the job that is being run.

    Parameters
    ----------
    number_of_particles: int
        Number of particles to be simulated.
    kinetic_energy: float
        The initial kinetic energy of the system, this same value is used to reweight the velocities.
    box_length: float
        Size of the simulation cell.
    timestep_length: float
        Duration of timestep in MD integration.
    max_vel: float, optional
        Maximum velocity allowed in the velocity rebinning stage.
    threshold: float, optional
        The energy change threshold for running an energy minimisation.
    max_steps: int, optional
        The maximum number of steps in the energy minimisation process.
    """
    def __init__(self, number_of_particles, temperature, box_length, timestep_length,
                 max_vel=4, threshold=1e-20, max_steps=1000):
        self.number_of_particles = number_of_particles
        self.temperature = temperature
        self.box_length = box_length
        self.timestep_length = timestep_length
        self.temp_sum = 0.
        self.vel_bins = np.zeros(500)
        self.max_vel = max_vel
        self.step = 0
        self.step0 = 0
        pairs = (self.number_of_particles-1)*self.number_of_particles/2
        self.distances = np.zeros(int(pairs))
        self.forces = np.zeros(int(pairs))
        self.temp_array = []
        self.press_array = [0]
        self.bin_width = 0.1
        self.time = 0
        self.total_force = np.zeros(3)
        self.old_total_force = np.zeros(3)
        self.threshold = threshold
        self.max_steps = max_steps


class Particle:
    """Particles

    This stores particle relevant information.

    Parameters
    ----------
    xpos: float
        Position in the x-axis.
    ypos: float
        Position in the y-axis.
    xvel: float
        Velocity in the x-axis.
    yvel: float
        Velocity in the y-axis.
    xacc: float
        Acceleration in the x-axis.
    yacc: float
        Acceleration in the y-axis.
    """
    def __init__(self, xpos, ypos, xvel, yvel, xacc, yacc):
        self.xpos = xpos
        self.ypos = ypos
        self.xvel = xvel
        self.yvel = yvel
        self.xacc = xacc
        self.yacc = yacc
        self.xpos_prev = 0.
        self.ypos_prev = 0.
        self.energy = 0.
        self.xforce = 0.
        self.yforce = 0.
        self.xforcedash = 0.
        self.yforcedash = 0.


def pbc_correction(d, l):
    """Test and correct for the periodic boundary condition.

    Parameters
    ----------
    d: float
        Particle position.
    l: float
        Box vector.

    Returns
    -------
    float
        Corrected particle position."""
    if np.abs(d) > 0.5 * l:
        d *= 1 - l / np.abs(d)
    return d

def calculate_pressure(system):
    """Calculates the instantaneous pressure of the system.

    Parameters
    ----------
    system: System
        Whole system information

    Returns
    -------
    System
        System with updated press_array to include newest instantaneous pressure.
    """
    w = (-1. / 3. * np.sum(system.distances * -1 * system.forces))
    system.press_array.append(system.number_of_particles * system.temperature + w)
    return system

def calculate_temperature(particles, system):
    """Determine the instantaneous temperature of the system.

    Parameters
    ----------
    particles: Particle array
        All particles in the system.
    system: System
        Whole system information.

    Returns
    -------
    Particle array:
        All particles updated with their velocities scaled.
    System:
        Whole system information with the temperature updated."""
    k = 0
    for i in range(0, system.number_of_particles):
        system, v = md.update_velocity_bins(particles[i], system)
        k += 0.5 * v * v
    system.temp_sum += k / system.number_of_particles
    temp = system.temp_sum / (system.step - system.step0)
    system.temp_array.append(temp)
    return particles, system


def set_particles_square(system):
    """Set the initial particle positions on a square lattice.

    Parameters
    ----------
    system: System
        Whole system information.

    Returns
    -------
    Particle array
        The particles with positions on a square lattice.
    """
    particles = np.array([], dtype=Particle)
    m = int(np.ceil(np.sqrt(system.number_of_particles)))
    d = system.box_length / m
    n = 0
    for i in range(0, m):
        for j in range(0, m):
            if n < system.number_of_particles:
                part = Particle((i + 0.5) * d, (j + 0.5) * d, 0, 0, 0, 0)
                particles = np.append(particles, part)
                n += 1
    return particles


def set_particles_random(system):
    """Set the initial particle positions in a random arrangement.

    Parameters
    ----------
    system: System
        Whole system information.

    Returns
    -------
    Particle array
        The particles with position set randomly.
    """
    particles = np.array([], dtype=Particle)
    for i in range(0, system.number_of_particles):
        part = Particle(np.random.uniform(0, system.box_length), np.random.uniform(0, system.box_length),
                        0, 0, 0, 0)
        particles = np.append(particles, part)
    return particles