import numpy as np
from pylj import md, force


class System:
    """Whole system.

    This class stores a large amount of information about the job that is being run.

    Parameters
    ----------
    number_of_particles: int
        Number of particles to be simulated.
    temperature: float
        The initial temperature of the system.
    box_length: float
        Size of the simulation cell.
    timestep_length: float
        Duration of timestep in MD integration.
    max_vel: float, optional
        Maximum velocity allowed in the velocity rebinning stage.
    init_conf: string, optional
        Selection for the way the particles are initially populated. Should be one of
        - 'square'
        - 'random'
    """
    def __init__(self, number_of_particles, temperature, box_length, timestep_length, 
                 max_vel=4, init_conf='square'):
        self.number_of_particles = number_of_particles
        self.init_temp = temperature
        self.box_length = box_length
        self.timestep_length = timestep_length
        self.max_vel = max_vel
        if init_conf == 'square':
            self.square()
        elif init_conf == 'random':
            self.random()
        else:
            raise NotImplementedError('The initial configuration type {} is not recognised. Available options are: square or random'.format(init_conf))
        self.step = 0
        self.time = 0.
        self.temp_sum = 0.
        self.distances = np.zeros(self.number_of_pairs())
        self.forces = np.zeros(self.number_of_pairs())
        self.velocity_bins = np.zeros(500)
        self.temperature = []
        self.pressure = []
        self.force = []

    def number_of_pairs(self):
        return int((self.number_of_particles - 1) * self.number_of_particles / 2)
        
    def square(self):
        """Set the initial particle positions on a square lattice.

        Returns
        -------
        Particle array
            The particles with positions on a square lattice.
        """
        part_dt = particle_dt()
        self.particles = np.zeros(self.number_of_particles, dtype=part_dt)
        m = int(np.ceil(np.sqrt(self.number_of_particles)))
        d = self.box_length / m
        n = 0
        for i in range(0, m):
            for j in range(0, m):
                if n < self.number_of_particles:
                    self.particles[n]['xposition'] = (i + 0.5) * d
                    self.particles[n]['yposition'] = (j + 0.5) * d
                    n += 1
    
    def random(self):
        """Set the initial particle positions in a random arrangement.
        
        Returns
        -------
        Particle array
            The particles with position set randomly.
        """
        part_dt = particle_dt()
        self.particles = np.zeros(self.number_of_particles, dtype=part_dt)
        self.particles['xposition'] = np.random.uniform(0, self.box_length, self.number_of_particles)
        self.particles['yposition'] = np.random.uniform(0, self.box_length, self.number_of_particles)   

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

def calculate_pressure(number_of_particles, particles, forces, box_length, velocity_bins, max_vel):
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
    k = 0
    pres = 0.
    for i in range(0, number_of_particles-1):
        for j in range(i+1, number_of_particles):
            velocity_bins, v = md.update_velocity_bins(particles[i], velocity_bins, max_vel)
            pres += forces[k] + v
    pres /= 3
    pres /= (box_length * box_length)
    return pres, velocity_bins

def calculate_temperature(system):
    """Determine the instantaneous temperature of the system.

    Parameters
    ----------
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
        system.velocity_bins, v = md.update_velocity_bins(system.particles[i], system.velocity_bins, system.max_vel)
        k += 0.5 * v * v
    system.temp_sum += k / system.number_of_particles
    temp = system.temp_sum
    system.temperature.append(temp)
    return system

def particle_dt():
    return np.dtype([('xposition', np.float64), ('yposition', np.float64), ('xvelocity', np.float64), ('yvelocity', np.float64), ('xacceleration', np.float64), ('yacceleration', np.float64), ('xprevious_position', np.float64), ('yprevious_position', np.float64), ('xforce', np.float64), ('yforce', np.float64), ('energy', np.float64)])
