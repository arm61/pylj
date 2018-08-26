from __future__ import division, absolute_import
import numpy as np

cimport numpy as np

cdef extern from "comp.h":
    void compute_accelerations(int len_particles, const double *xpos,
                               const double *ypos, double *xacc, double *yacc,
                               double *distances_arr, double box_l,
                               double *force_arr, double *energy_arr,
                               double cut, double ac, double bc, double massc)
    void compute_energies(int len_particles, const double *xpos,
                          const double *ypos, double *distances_arr,
                          double box_l, double *energy_arr, double cut,
                          double ac, double bc)
    double compute_pressure(int number_of_particles, const double *xvel,
                            const double *yvel, double box_length,
                            double temperature, double cut, double ac,
                            double bc)
    void scale_velocities(int len_particles, double *xvel, double *yvel,
                          double average_temp, double tempature)

    void get_distances(int len_particles, double *xpositions, double *ypositions,
                     double box_l, double *distances, double *xdistances,
                     double *ydistances)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def compute_force(particles, box_length, cut_off,
                  constants=[1.363e-134, 9.273e-78], mass=39.948):
    """Calculates the forces and therefore the accelerations on each of the
    particles in the simulation. This uses a 12-6 Lennard-Jones potential
    model for Argon with values:

    - A = 1.363e-134 J m :math:`^{12}`
    - B = 9.273e-78 J m :math:`^6`

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    cut_off: float
        The distance greater than which the forces between particles is taken
        as zero.
    constants: float, array_like (optional)
        The constants associated with the particular forcefield used, e.g. for
        the function forcefields.lennard_jones, theses are [A, B].
    mass: float (optional)
        The mass of the particle being simulated (units of atomic mass units).

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
    print("This function is deprecated, please use pairwise.compute_force, if "
          "you have the compiled dist function, it is just as fast. ")
    cdef int len_particles = particles['xposition'].size
    pairs = int((len_particles - 1) * len_particles / 2)
    cdef double box_l = box_length
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] xacc = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] yacc = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] distances_arr = np.zeros(pairs)
    cdef np.ndarray[DTYPE_t, ndim=1] force_arr = np.zeros(pairs)
    cdef np.ndarray[DTYPE_t, ndim=1] energy_arr = np.zeros(pairs)
    cdef double cut = cut_off
    cdef double ac = constants[0]
    cdef double bc = constants[1]
    cdef double massc = mass

    for i in range(0, len_particles):
        xpos[i] = particles['xposition'][i]
        ypos[i] = particles['yposition'][i]
        xacc[i] = 0
        yacc[i] = 0

    compute_accelerations(len_particles, <const double*>xpos.data,
                          <const double*>ypos.data, <double*>xacc.data,
                          <double*>yacc.data, <double*>distances_arr.data,
                          box_l, <double*>force_arr.data,
                          <double*>energy_arr.data, cut, ac, bc, massc)

    for i in range(0, len_particles):
        particles['xacceleration'][i] = xacc[i]
        particles['yacceleration'][i] = yacc[i]


    return particles, distances_arr, force_arr, energy_arr

def compute_energy(particles, box_length, cut_off,
                   constants=[1.363e-134, 9.273e-78]):
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
        The distance greater than which the energies between particles is
        taken as zero.
    constants: float, array_like (optional)
        The constants associated with the particular forcefield used, e.g. for
        the function forcefields.lennard_jones, theses are [A, B].

    Returns
    -------
    util.particle_dt, array_like
        Information about particles, with updated accelerations and forces.
    float, array_like
        Current distances between pairs of particles in the simulation.
    float, array_like
        Current energies between pairs of particles in the simulation.
    """
    print("This function is deprecated, please use pairwise.compute_energy, "
          "if you have the compiled dist function, it is just as fast. ")
    cdef int len_particles = particles['xposition'].size
    pairs = int((len_particles - 1) * len_particles / 2)
    cdef double box_l = box_length
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] distances_arr = np.zeros(pairs)
    cdef np.ndarray[DTYPE_t, ndim=1] energy_arr = np.zeros(pairs)
    cdef double cut = cut_off
    cdef double ac = constants[0]
    cdef double bc = constants[1]


    for i in range(0, len_particles):
        xpos[i] = particles['xposition'][i]
        ypos[i] = particles['yposition'][i]


    compute_energies(len_particles, <const double*>xpos.data,
                     <const double*>ypos.data, <double*>distances_arr.data,
                     box_l, <double*>energy_arr.data, cut, ac, bc)

    return distances_arr, energy_arr

def calculate_pressure(particles, box_length, temperature, cut_off,
                       constants=[1.363e-134, 9.273e-78]):
    r"""Calculates the instantaneous pressure of the simulation cell, found
    with the following relationship:

    .. math::
        p = \langle \rho k_b T \rangle + \bigg\langle
        \frac{1}{3V}\sum_{i}\sum_{j<i} \mathbf{r}_{ij}\mathbf{f}_{ij}
        \bigg\rangle

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    temperature: float
        Instantaneous temperature of the simulation.
    cut_off: float
        The distance greater than which the forces between particles is taken
        as zero.
    constants: float, array_like (optional)
        The constants associated with the particular forcefield used, e.g. for
        the function forcefields.lennard_jones, theses are [A, B].


    Returns
    -------
    float:
        Instantaneous pressure of the simulation.
    """
    print("This function is deprecated, please use "
          "pairwise.calculate_pressure, if you have the compiled dist "
          "function, it is just as fast. ")
    cdef int number_of_particles = particles['xposition'].size
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(number_of_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(number_of_particles)
    cdef double box_l = box_length
    cdef double pressure = 0.
    cdef double temp = temperature
    cdef double cut = cut_off
    cdef double ac = constants[0]
    cdef double bc = constants[1]


    for i in range(0, number_of_particles):
        xpos[i] = particles['xposition'][i]
        ypos[i] = particles['yposition'][i]

    pressure = compute_pressure(number_of_particles, <const double*>xpos.data,
                                <const double*>ypos.data, box_l, temperature,
                                cut, ac, bc)

    return pressure

def heat_bath(particles, temperature_sample, bath_temp):
    r"""Rescales the velocities of the particles in the system to control the
    temperature of the simulation. Thereby allowing for an NVT ensemble. The
    velocities are rescaled according the following relationship,

    .. math::
        v_{\text{new}} = v_{\text{old}} \times
        \sqrt{\frac{T_{\text{desired}}}{\bar{T}}}

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
    cdef int len_particles = particles['xposition'].size
    cdef np.ndarray[DTYPE_t, ndim=1] xvel = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] yvel = np.zeros(len_particles)
    cdef double average_temp = np.average(temperature_sample)
    cdef double temperature = bath_temp

    for i in range(0, len_particles):
        xvel[i] = particles['xvelocity'][i]
        yvel[i] = particles['yvelocity'][i]

    scale_velocities(len_particles, <double*>xvel.data, <double*>yvel.data,
                     average_temp, temperature)

    for i in range(0, len_particles):
        particles['xvelocity'][i] = xvel[i]
        particles['yvelocity'][i] = yvel[i]

    return particles

def dist(xpos, ypos, box_length):
    """Returns the distance array for the set of particles.

    Parameters
    ----------
    xpos: float, array_like (N)
        Array of length N, where N is the number of particles, providing the
        x-dimension positions of the particles.
    ypos: float, array_like (N)
        Array of length N, where N is the number of particles, providing the
        y-dimension positions of the particles.
    box_length: float
        The box length of the simulation cell.

    Returns
    -------
    distances float, array_like ((N - 1) * N / 2))
        The pairs of distances between the particles.
    xdistances float, array_like ((N - 1) * N / 2))
        The pairs of distances between the particles, in only the x-dimension.
    ydistances float, array_like ((N - 1) * N / 2))
        The pairs of distances between the particles, in only the y-dimension.
    """
    cdef int len_particles = int(xpos.size)
    cdef np.ndarray[DTYPE_t, ndim=1] xpositions = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] ypositions = np.zeros(len_particles)
    cdef double box_l = box_length
    cdef np.ndarray[DTYPE_t, ndim=1] distances = np.zeros(int((len_particles - 1) * len_particles / 2))
    cdef np.ndarray[DTYPE_t, ndim=1] xdistances = np.zeros(int((len_particles - 1) * len_particles / 2))
    cdef np.ndarray[DTYPE_t, ndim=1] ydistances = np.zeros(int((len_particles - 1) * len_particles / 2))


    for i in range(0, len_particles):
        xpositions[i] = xpos[i]
        ypositions[i] = ypos[i]

    get_distances(len_particles, <double*>xpositions.data, <double*>ypositions.data,
                  box_l, <double*>distances.data, <double*>xdistances.data, <double*>ydistances.data)

    return distances, xdistances, ydistances
