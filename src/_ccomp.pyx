from __future__ import division, absolute_import
import numpy as np

cimport numpy as np

cdef extern from "comp.h":
    void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                               double *distances_arr, double box_l, double *force_arr, double *energy_arr, double cut)
    void compute_energies(int len_particles, const double *xpos, const double *ypos, double *distances_arr, double box_l,
                        double *energy_arr, double cut)
    double compute_pressure(int number_of_particles, const double *xvel, const double *yvel, double box_length,
                            double temperature, double cut)
    void scale_velocities(int len_particles, double *xvel, double *yvel, double average_temp, double tempature)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def compute_forces(particles, box_length, cut_off):
    """Calculates the forces and therefore the accelerations on each of the particles in the simulation. This uses a
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
        The distance greater than which the forces between particles is taken as zero.

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

    for i in range(0, len_particles):
        xpos[i] = particles['xposition'][i]
        ypos[i] = particles['yposition'][i]
        xacc[i] = 0
        yacc[i] = 0

    compute_accelerations(len_particles, <const double*>xpos.data, <const double*>ypos.data, <double*>xacc.data,
                          <double*>yacc.data, <double*>distances_arr.data, box_l, <double*>force_arr.data,
                          <double*>energy_arr.data, cut)

    for i in range(0, len_particles):
        particles['xacceleration'][i] = xacc[i]
        particles['yacceleration'][i] = yacc[i]


    return particles, distances_arr, force_arr, energy_arr

def compute_energy(particles, box_length, cut_off):
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
        The distance greater than which the energies between particles is taken as zero.

    Returns
    -------
    util.particle_dt, array_like
        Information about particles, with updated accelerations and forces.
    float, array_like
        Current distances between pairs of particles in the simulation.
    float, array_like
        Current energies between pairs of particles in the simulation.
    """
    cdef int len_particles = particles['xposition'].size
    pairs = int((len_particles - 1) * len_particles / 2)
    cdef double box_l = box_length
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] distances_arr = np.zeros(pairs)
    cdef np.ndarray[DTYPE_t, ndim=1] energy_arr = np.zeros(pairs)
    cdef double cut = cut_off


    for i in range(0, len_particles):
        xpos[i] = particles['xposition'][i]
        ypos[i] = particles['yposition'][i]


    compute_energies(len_particles, <const double*>xpos.data, <const double*>ypos.data,
                          <double*>distances_arr.data, box_l, <double*>energy_arr.data, cut)

    return particles, distances_arr, energy_arr

def calculate_pressure(particles, box_length, temperature, cut_off):
    r"""Calculates the instantaneous pressure of the simulation cell, found with the following relationship:

    .. math::
        p = \langle \rho k_b T \rangle + \bigg\langle \frac{1}{3V}\sum_{i}\sum_{j<i} \mathbf{r}_{ij}\mathbf{f}_{ij} \bigg\rangle

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    temperature: float
        Instantaneous temperature of the simulation.
    cut_off: float
        The distance greater than which the forces between particles is taken as zero.

    Returns
    -------
    float:
        Instantaneous pressure of the simulation.
    """
    cdef int number_of_particles = particles['xposition'].size
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(number_of_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(number_of_particles)
    cdef double box_l = box_length
    cdef double pressure = 0.
    cdef double temp = temperature
    cdef double cut = cut_off


    for i in range(0, number_of_particles):
        xpos[i] = particles['xposition'][i]
        ypos[i] = particles['yposition'][i]

    pressure = compute_pressure(number_of_particles, <const double*>xpos.data, <const double*>ypos.data, box_l,
                                temperature, cut)

    return pressure

def heat_bath(particles, temperature_sample, bath_temp):
    r"""Rescales the velocities of the particles in the system to control the temperature of the simulation. Thereby
    allowing for an NVT ensemble. The velocities are rescaled according the following relationship,

    .. math::
        v_{\text{new}} = v_{\text{old}} \times \sqrt{\frac{T_{\text{desired}}}{\bar{T}}}

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

    scale_velocities(len_particles, <double*>xvel.data, <double*>yvel.data, average_temp, temperature)

    for i in range(0, len_particles):
        particles['xvelocity'][i] = xvel[i]
        particles['yvelocity'][i] = yvel[i]

    return particles