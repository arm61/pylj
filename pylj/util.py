from __future__ import division
import numpy as np
import webbrowser
from pylj import md, mc
try:
    from pylj import comp as heavy
except ImportError:
    print("WARNING, using slow force and energy calculations")
    from pylj import pairwise as heavy



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
        if box_length > 30:
            self.cut_off = cut_off * 1e-10
        else:
            self.cut_off = box_length / 2 * 1e-10
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
        self.old_energy = 0
        self.new_energy = 0
        self.random_particle = 0

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

    def compute_force(self):
        """Maps to the compute_force function in either the comp (if Cython is installed) or the pairwise module and
        allows for a cleaner interface.
        """
        self.particles, self.distances, self.forces, self.energies = heavy.compute_forces(self.particles,
                                                                                         self.box_length,
                                                                                         self.cut_off)

    def compute_energy(self):
        """Maps to the compute_energy function in either the comp (if Cython is installed) or the pairwise module
        and allows for a cleaner interface.
        """
        self.distances, self.energies = heavy.compute_energy(self.particles,
                                                             self.box_length,
                                                             self.cut_off)

    def integrate(self, method):
        """Maps the chosen integration method.

        Parameters
        ----------
        method: method
            The integration method to be used, e.g. md.velocity_verlet.
        """
        self.particles = method(self.particles, self.timestep_length, self.box_length, self.cut_off)

    def md_sample(self):
        """Maps to the md.sample function.
        """
        md.sample(self.particles, self.box_length, self.initial_particles, self)

    def heat_bath(self, bath_temperature):
        """Maps to the heat_bath function in either the comp (if Cython is installed) or the pairwise modules.

        Parameters
        ----------
        target_temperature: float
            The target temperature for the simulation.
        """
        self.particles = heavy.heat_bath(self.particles, self.temperature_sample, bath_temperature)

    def mc_sample(self):
        """Maps to the mc.sample function.

        Parameters
        ----------
        energy: float
            Energy to add to the sample
        """
        mc.sample(self.old_energy, self)

    def select_random_particle(self):
        """Maps to the mc.select_random_particle function.
        """
        self.random_particle, self.position_store = mc.select_random_particle(self.particles)

    def new_random_position(self):
        """Maps to the mc.get_new_particle function.
        """
        self.particles = mc.get_new_particle(self.particles, self.random_particle, self.box_length)

    def accept(self):
        """Maps to the mc.accept function.
        """
        self.old_energy = mc.accept(self.new_energy)

    def reject(self):
        """Maps to the mc.reject function.
        """
        self.particles = mc.reject(self.position_store, self.particles, self.random_particle)


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


def __cite__(): #pragma: no cover
    """This function will launch the Zenodo website for the latest release of pylj."""
    webbrowser.open('https://zenodo.org/badge/latestdoi/119863480')


def __version__(): #pragma: no cover
    """This will print the number of the pylj version currently in use."""
    major = 1
    minor = 0
    micro = 1
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
