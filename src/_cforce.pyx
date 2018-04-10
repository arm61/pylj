from __future__ import division, absolute_import
import numpy as np

cimport numpy as np

cdef extern from "force.h":
    double compute_pressure(int number_of_particles, const double *xvel, const double *yvel, const double *forces, 
                          double box_length)
    void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                               double *distances, double *xforce, double *yforce, double box_length, double *force_arr)
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

def calculate_pressure(parts, partic, forc, box_l):
    cdef int number_of_particles = parts
    cdef np.ndarray[DTYPE_t, ndim=1] xvel = np.zeros(number_of_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] yvel = np.zeros(number_of_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] forces = np.zeros(len(forc))
    cdef double box_length = box_l
    cdef double pres = 0.

    for i in range(0, number_of_particles):
        xvel[i] = partic['xvelocity'][i]
        yvel[i] = partic['yvelocity'][i]

    pres = compute_pressure(number_of_particles, <const double*>xvel.data, <const double*>yvel.data, <const double*>forces.data, box_length)

    return pres
    

def compute_forces(system):
    cdef int len_particles = system.number_of_particles 
    cdef double box_length = system.box_length
    cdef np.ndarray[DTYPE_t, ndim=1] xpos = np.zeros(len_particles) 
    cdef np.ndarray[DTYPE_t, ndim=1] ypos = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] xacc = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] yacc = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] distances = np.zeros(len(system.distances))
    cdef np.ndarray[DTYPE_t, ndim=1] force_arr = np.zeros(len(system.distances))
    cdef np.ndarray[DTYPE_t, ndim=1] xforce = np.zeros(len_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] yforce = np.zeros(len_particles)

    for i in range(0, len_particles):
        xpos[i] = system.particles['xposition'][i]
        ypos[i] = system.particles['yposition'][i]
        xacc[i] = system.particles['xacceleration'][i]
        yacc[i] = system.particles['yacceleration'][i]
    
    compute_accelerations(len_particles, <const double*>xpos.data, <const double*>ypos.data, <double*>xacc.data,
                          <double*>yacc.data, <double*>distances.data, <double*>xforce.data, <double*>yforce.data,
                          box_length, <double*>force_arr.data)

    for i in range(0, len_particles):
        system.particles['xposition'][i] = xpos[i]
        system.particles['yposition'][i] = ypos[i]
        system.particles['xacceleration'][i] = xacc[i]
        system.particles['yacceleration'][i] = yacc[i]
        system.particles['xforce'][i] = xforce[i]
        system.particles['yforce'][i] = yforce[i]
    
    system.distances = distances
    system.forces = force_arr

    return system

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


def heat_bath(system, bath_temp):
    cdef int len_particles = system.number_of_particles
    cdef np.ndarray[DTYPE_t, ndim=1] xvel = np.zeros(system.number_of_particles)
    cdef np.ndarray[DTYPE_t, ndim=1] yvel = np.zeros(system.number_of_particles)
    cdef double average_temp = np.average(system.temperature)
    cdef double temperature = bath_temp

    for i in range(0, system.number_of_particles):
        xvel[i] = system.particles['xvelocity'][i]
        yvel[i] = system.particles['yvelocity'][i]

    scale_velocities(len_particles, <double*>xvel.data, <double*>yvel.data, average_temp, temperature)

    for i in range(0, system.number_of_particles):
        system.particles['xvelocity'][i] = xvel[i]
        system.particles['yvelocity'][i] = yvel[i]

    return system
