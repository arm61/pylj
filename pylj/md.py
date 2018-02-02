import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pyqtgraph as pg
from IPython import display
from pylj import slow, plot


class System:
    def __init__(self, number_of_particles, kinetic_energy, box_length, timestep_length,
                 max_vel):
        if number_of_particles > 324:
            raise ValueError("Density too high!")
        self.number_of_particles = number_of_particles
        self.kinetic_energy = kinetic_energy
        self.box_length = box_length
        self.timestep_length = timestep_length
        self.temp_sum = 0.
        self.vel_bins = np.zeros(500)
        self.max_vel = max_vel
        self.step = 0
        self.step0 = 0
        pairs = (self.number_of_particles-1)*self.number_of_particles/2
        self.distances = np.zeros(int(pairs))
        self.temp_array = []


class Particle:
    def __init__(self, xpos, ypos, xvel, yvel, xacc, yacc):
        self.xpos = xpos
        self.ypos = ypos
        self.xvel = xvel
        self.yvel = yvel
        self.xacc = xacc
        self.yacc = yacc


def clear_accerlerations(particles):
    for i in range(0, len(particles)):
        particles[i].xacc = 0
        particles[i].yacc = 0
    return particles


def pbc_correction(d, l):
    if np.abs(d) > 0.5 * l:
        d *= 1 - l / np.abs(d)
    return d


def compute_acceleration(particles, system):
    system.distances = []
    particles = clear_accerlerations(particles)
    for i in range(0, len(particles) - 1):
        for j in range(i + 1, len(particles)):
            dx = particles[i].xpos - particles[j].xpos
            dy = particles[i].ypos - particles[j].ypos
            dx = pbc_correction(dx, system.box_length)
            dy = pbc_correction(dy, system.box_length)
            dr = np.sqrt(dx * dx + dy * dy)
            system.distances = np.append(system.distances, dr)
            f = 48 * np.power(dr, -13.) - 24 * np.power(dr, -7.)
            particles[i].xacc += f * dx / dr
            particles[i].yacc += f * dy / dr
            particles[j].xacc -= f * dx / dr
            particles[j].yacc -= f * dy / dr
    return particles, system


def reset_histogram(system):
    for i in range(0, len(system.vel_bins)):
        system.vel_bins[i] = 0
    system.temp_sum = 0
    return system.step


def initialise(system):
    particles = np.array([], dtype=Particle)
    m = int(np.ceil(np.sqrt(system.number_of_particles)))
    d = system.box_length / m
    n = 0
    for i in range(0, m):
        for j in range(0, m):
            if n < system.number_of_particles:
                part = Particle((i + 0.5)*d, (j + 0.5) * d, 0, 0, 0, 0)
                particles = np.append(particles, part)
                n += 1
    v = np.sqrt(2 * system.kinetic_energy)
    for i in range(0, system.number_of_particles):
        theta = 2 * np.pi * np.random.randn()
        particles[i].xvel = v * np.cos(theta)
        particles[i].yvel = v * np.sin(theta)
    particles, system = slow.comp_accel(particles, system)
    system.step0 = reset_histogram(system)
    temp = system.kinetic_energy
    return particles, temp


def update_positions(particles, system, i):
    particles[i].xpos += particles[i].xvel * system.timestep_length + 0.5 * particles[i].xacc * \
                                                                      system.timestep_length * system.timestep_length
    particles[i].ypos += particles[i].yvel * system.timestep_length + 0.5 * particles[i].yacc * \
                                                                      system.timestep_length * system.timestep_length
    particles[i].xpos = particles[i].xpos % system.box_length
    particles[i].ypos = particles[i].ypos % system.box_length
    return particles


def update_velocities(particles, system, i):
    particles[i].xvel += 0.5 * particles[i].xacc * system.timestep_length
    particles[i].yvel += 0.5 * particles[i].yacc * system.timestep_length
    return particles


def update_velocity_bins(particles, system, i):
    v = np.sqrt(particles[i].xvel * particles[i].xvel + particles[i].yvel * particles[i].yvel)
    bin_s = int(len(system.vel_bins) * v / system.max_vel)
    if bin_s < 0:
        bin_s = 0
    if bin_s >= len(system.vel_bins):
        bin_s = len(system.vel_bins) - 1
    system.vel_bins[bin_s] += 1
    return system, v


def time_step(particles, system, time):
    time += system.timestep_length
    system.step += 1
    k = 0
    for i in range(0, system.number_of_particles):
        particles = update_positions(particles, system, i)
        particles = update_velocities(particles, system, i)
    particles, system = slow.comp_accel(particles, system)
    for i in range(0, system.number_of_particles):
        system, v = update_velocity_bins(particles, system, i)
        k += 0.5 * v * v
    system.temp_sum += k / system.number_of_particles
    temp = system.temp_sum / (system.step - system.step0)
    system.temp_array.append(temp)
    return particles, time, system


def run(number_of_particles, kinetic_energy, number_steps):
    system = System(number_of_particles, kinetic_energy, 16., 0.01, 4)
    particles, temp = initialise(system)
    plot_ob = plot.liveplot(system)
    time = 0
    for i in range(0, number_steps):
        particles, time, system = time_step(particles, system, time)
        if system.step % 10 == 0:
            bin_width = 0.1
            hist, bin_edges = np.histogram(system.distances, bins=np.arange(0, 12.5, bin_width))
            gr = hist / (system.number_of_particles * (system.number_of_particles / system.box_length ** 2) * np.pi *
                         (bin_edges[:-1] + bin_width / 2.) * bin_width)
            x = bin_edges[:-1] + bin_width / 2
            plot_ob.update(x, gr, np.fft.rfftfreq(len(gr))[5:], np.log10(np.fft.rfft(gr)[5:]), particles, system)
        system.step0 = reset_histogram(system)
    plt.show()