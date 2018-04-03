from __future__ import division, absolute_import
import numpy as np

cimport numpy as np

cdef extern from "force.h":
    void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                               double *distances, double* forces, double box_length)
    void compute_sd(int len_particles, const double *xpos, const double *ypos, double *energy, double *xforce,
                    double *yforce, double box_length)
    void compute_energy_and_force(int len_particles, const double *xpos, const double *ypos, double *energy,
                                  double *xforce, double *yforce, double *xforcedash, double *yforcedash,
                                  double box_length)
    void compute_force(int len_particles, const double *xpos, const double *ypos, double *xforce, double *yforce,
                       double box_length)
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
    cdef np.ndarray[DTYPE_t, ndim=1] distances = np.zeros(len(system.distances))
    cdef np.ndarray[DTYPE_t, ndim=1] forces = np.zeros(len(system.forces))

    for i in range(0, particles.size):
        xpos[i] = particles[i].xpos
        ypos[i] = particles[i].ypos
        xacc[i] = particles[i].xacc
        yacc[i] = particles[i].yacc

    compute_accelerations(len_particles, <const double*>xpos.data, <const double*>ypos.data, <double*>xacc.data,
                          <double*>yacc.data, <double*>distances.data, <double*>forces.data, box_length)

    for i in range(0, particles.size):
        particles[i].xpos = xpos[i]
        particles[i].ypos = ypos[i]
        particles[i].xacc = xacc[i]
        particles[i].yacc = yacc[i]

    system.distances = distances
    system.forces = forces

    return particles, system

def calculate_sd(particles, system):
    cdef int len_particles = particles.size
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] energy = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] xforce = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] yforce = np.zeros(particles.size)
    cdef double box_length = system.box_length

    for i in range(0, particles.size):
        xpos[i] = particles[i].xpos
        ypos[i] = particles[i].ypos

    compute_sd(len_particles, <const double*> xpos.data, <const double*> ypos.data, <double*> energy.data,
                             <double*>xforce.data, <double*> yforce.data, box_length)

    for i in range(0, particles.size):
        particles[i].energy = energy[i]
        particles[i].xforce = xforce[i]
        particles[i].yforce = yforce[i]

    return particles

def calculate_energy_and_force(particles, system):
    cdef int len_particles = particles.size
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] energy = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] xforce = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] yforce = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] xforcedash = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] yforcedash = np.zeros(particles.size)
    cdef double box_length = system.box_length

    for i in range(0, particles.size):
        xpos[i] = particles[i].xpos
        ypos[i] = particles[i].ypos


    compute_energy_and_force(len_particles, <const double*> xpos.data, <const double*> ypos.data, <double*> energy.data,
                             <double*>xforce.data, <double*> yforce.data, <double*> xforcedash.data,
                             <double*> yforcedash.data, box_length)

    for i in range(0, particles.size):
        particles[i].energy = energy[i]
        particles[i].xforce = xforce[i]
        particles[i].yforce = yforce[i]
        particles[i].xforcedash = xforcedash[i]
        particles[i].yforcedash = yforcedash[i]

    return particles

def get_forces(particles, system):
    cdef int len_particles = particles.size
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] xforce = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] yforce = np.zeros(particles.size)
    cdef double box_length = system.box_length

    for i in range(0, particles.size):
        xpos[i] = particles[i].xpos
        ypos[i] = particles[i].ypos


    compute_force(len_particles, <const double*> xpos.data, <const double*> ypos.data, <double*>xforce.data,
                  <double*> yforce.data, box_length)

    return xforce, yforce


def heat_bath(particles, system):
    cdef int len_particles = particles.size
    cdef np.ndarray[DTYPE_t, ndim=1] xvel = np.zeros(particles.size)
    cdef np.ndarray[DTYPE_t, ndim=1] yvel = np.zeros(particles.size)
    cdef double average_temp = np.average(system.temp_array)
    cdef double temperature = system.temperature

    for i in range(0, particles.size):
        xvel[i] = particles[i].xvel
        yvel[i] = particles[i].yvel

    scale_velocities(len_particles, <double*>xvel.data, <double*>yvel.data, average_temp, temperature)

    for i in range(0, particles.size):
        particles[i].xvel = xvel[i]
        particles[i].yvel = yvel[i]

    return particles