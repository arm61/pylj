from __future__ import division, absolute_import
import numpy as np

cimport numpy as np

cdef extern from "comp.h":
    void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                               double *distances_arr, double *xforce, double *yforce, double box_l,
                               double *force_arr)
    double compute_pressure(int number_of_particles, const double *xvel, const double *yvel, double box_length,
                            double temperature)
    void scale_velocities(int len_particles, double *xvel, double *yvel, double average_temp, double tempature)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def compute_forces(particles, distances, forces, box_length):
    """Calculates the forces and therefore the accelerations on each of the particles in the simulation. This uses a
    12-6 Lennard-Jones potential model for Argon with values:

    - A = 1.89774e-13 J Å :math:`^{12}`
    - B = 5.1186e-19 J Å :math:`^6`

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    distances: float, array_like
        Old distances between each of the pairs of particles in the simulation.
    forces: float, array_like
        Old forces between each of the pairs of particles in the simulation.

    Returns
    -------
    util.particle_dt, array_like
        Information about particles, with updated accelerations and forces.
    float, array_like
        Current distances between pairs of particles in the simulation.
    float, array_like
        Current forces between pairs of particles in the simulation.
    """
    cdef int len_particles = particles['xposition'].size
    cdef double box_l = box_length
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] xacc = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] yacc = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] distances_arr = np.zeros(len(distances))
    cdef np.ndarray[DTYPE_t, ndim=1] force_arr = np.zeros(len(distances))
    cdef np.ndarray[DTYPE_t, ndim=1] xforce = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] yforce = np.zeros(len_particles)

    for i in range(0, len_particles):
        xpos[i] = particles['xposition'][i]
        ypos[i] = particles['yposition'][i]
        xacc[i] = particles['xacceleration'][i]
        yacc[i] = particles['yacceleration'][i]

    compute_accelerations(len_particles, <const double*>xpos.data, <const double*>ypos.data, <double*>xacc.data,
                          <double*>yacc.data, <double*>distances_arr.data, <double*>xforce.data, <double*>yforce.data,
                          box_l, <double*>force_arr.data)

    for i in range(0, len_particles):
        particles['xacceleration'][i] = xacc[i]
        particles['yacceleration'][i] = yacc[i]
        particles['xforce'][i] = xforce[i]
        particles['yforce'][i] = yforce[i]

    distances = distances_arr
    forces = force_arr

    return particles, distances, forces

def calculate_pressure(particles, box_length, temperature):
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

    for i in range(0, number_of_particles):
        xpos[i] = particles['xposition'][i]
        ypos[i] = particles['yposition'][i]

    pressure = compute_pressure(number_of_particles, <const double*>xpos.data, <const double*>ypos.data, box_l,
                                temperature)

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