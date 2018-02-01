from __future__ import division, absolute_import
import numpy as np

cimport numpy as np

cdef extern from "slowstuff.h":
    void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                               double *distances, double box_length)
    void debye(int len_distances, const double *cdistances, int len_qs, const double *cqs, double *inten)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def comp_accel(particles, system):
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

def get_scat(distances, qs):
    cdef np.ndarray[DTYPE_t, ndim=1] cdistances = distances
    cdef np.ndarray[DTYPE_t, ndim=1] cqs = qs
    cdef int len_distances = distances.size
    cdef int len_qs = qs.size
    cdef np.ndarray[DTYPE_t, ndim=1] inten = np.zeros(qs.size)


    debye(len_distances, <const double*>cdistances.data, len_qs, <const double*>cqs.data, <double*>inten.data)



    return inten
