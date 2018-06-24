from __future__ import division
import numpy as np
import webbrowser



class System:
    """Simulation system.

    This class is designed to store all of the information about the job that is being run. This includes the particles
    object, as will as sampling objects such as the temperature, pressure, etc. arrays.

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
    """
    def __init__(self, number_of_particles, temperature, box_length, init_conf='square', timestep_length=1e-14,
                 cut_off=15):
        self.number_of_particles = number_of_particles
        self.init_temp = temperature
        if box_length <= 600:
            self.box_length = box_length * 1e-10
        else:
            raise AttributeError('With a box length of {} the particles are probably too small to be seen in the '
                                 'viewer. Try something (much) less than 600.'.format(box_length))
        if box_length >= 4:
            self.box_length = box_length * 1e-10
        else:
            raise AttributeError('With a box length of {} the cell is too small to really hold more than one '
                                 'particle.'.format(box_length))
        self.timestep_length = timestep_length
        self.particles = None
        if init_conf == 'square':
            self.square()
        elif init_conf == 'random':
            self.random()
        else:
            raise NotImplementedError('The initial configuration type {} is not recognised. '
                                      'Available options are: square or random'.format(init_conf))
        self.cut_off = cut_off * 1e-10
        self.step = 0
        self.time = 0.
        self.distances = np.zeros(self.number_of_pairs())
        self.forces = np.zeros(self.number_of_pairs())
        self.energies = np.zeros(self.number_of_pairs())
        self.temperature_sample = np.array([])
        self.pressure_sample = np.array([])
        self.force_sample = np.array([])
        self.msd_sample = np.array([])
        self.energy_sample = np.array([])
        self.initial_particles = np.array(self.particles)
        self.position_store = [0, 0]

    def number_of_pairs(self):
        """Calculates the number of pairwise interactions in the simulation.

        Returns
        -------
        int:
            Number of pairwise interactions in the system.
        """
        return int((self.number_of_particles - 1) * self.number_of_particles / 2)
        
    def square(self):
        """Sets the initial positions of the particles on a square lattice.
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
        """Sets the initial positions of the particles in a random arrangement.
        """
        part_dt = particle_dt()
        self.particles = np.zeros(self.number_of_particles, dtype=part_dt)
        self.particles['xposition'] = np.random.uniform(0, self.box_length, self.number_of_particles)
        self.particles['yposition'] = np.random.uniform(0, self.box_length, self.number_of_particles)   


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
        k += 66.234e-27 * v * v / (1.3806e-23 * 2 * particles['xposition'].size)
    return k


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


def __cite__(): #pragma: no cover
    """This function will launch the Zenodo website for the latest release of pylj."""
    webbrowser.open('https://zenodo.org/badge/latestdoi/119863480')


def __version__(): #pragma: no cover
    """This will print the number of the pylj version currently in use."""
    major = 1
    minor = 0
    micro = 0 
    print('pylj-{:d}.{:d}.{:d}'.format(major, minor, micro))


def particle_dt():
    """Builds the data type for the particles, this consists of:

    - xposition and yposition
    - xvelocity and yvelocity
    - xacceleration and yacceleration
    - xprevious_position and ypresvious_position
    - xforce and yforce
    - energy
    """
    return np.dtype([('xposition', np.float64), ('yposition', np.float64), ('xvelocity', np.float64),
                     ('yvelocity', np.float64), ('xacceleration', np.float64), ('yacceleration', np.float64),
                     ('xprevious_position', np.float64), ('yprevious_position', np.float64), ('energy', np.float64)])
