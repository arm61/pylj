from __future__ import division, absolute_import
import numpy as np

cimport numpy as np

cdef extern from "force.h":
    void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                               double *distances, double box_length)
    void scale_velocities(int len_particles, double *xvel, double *yvel, double average_temp, double tempature)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def compute_forces(particles, system):
    cdef int len_particles = particles.size
    cdef double box_length = system.box_length
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] xacc = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] yacc = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] distances = system.distances

    for i in range(0, particles.size):
        xpos[i] = particles[i].xpos
        ypos[i] = particles[i].ypos
        xacc[i] = particles[i].xacc
        yacc[i] = particles[i].yacc

    compute_accelerations(len_particles, <const double*>xpos.data, <const double*>ypos.data, <double*>xacc.data,
                          <double*>yacc.data, <double*>distances.data, box_length)

    for i in range(0, particles.size):
        particles[i].xpos = xpos[i]
        particles[i].ypos = ypos[i]
        particles[i].xacc = xacc[i]
        particles[i].yacc = yacc[i]

    system.distances = distances


    return particles, system

def scale_velo(particles, system):
    cdef int len_particles = particles.size
    cdef np.ndarray[DTYPE_t, ndim=1] xvel = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] yvel = np.zeros(particles.size)
    cdef double average_temp = np.average(system.temp_array)
    cdef double temperature = system.kinetic_energy

    for i in range(0, particles.size):
        xvel[i] = particles[i].xvel
        yvel[i] = particles[i].yvel

    scale_velocities(len_particles, <double*>xvel.data, <double*>yvel.data, average_temp, temperature)

    for i in range(0, particles.size):
        particles[i].xvel = xvel[i]
        particles[i].yvel = yvel[i]

    return particles