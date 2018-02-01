import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from IPython import display
from pylj import slow

class System:
    def __init__(self, number_of_particles, kinetic_energy, box_length, timestep_length, number_vel_bins,
                 max_vel):
        self.number_of_particles = number_of_particles
        self.kinetic_energy = kinetic_energy
        self.box_length = box_length
        self.timestep_length = timestep_length
        self.temp_sum = 0.
        self.vel_bins = np.zeros(number_vel_bins)
        self.max_vel = max_vel
        self.step = 0
        self.step0 = 0
        pairs = (self.number_of_particles-1)*self.number_of_particles/2
        self.distances = np.zeros(int(pairs))


class Particle:
    def __init__(self, xpos, ypos, xvel, yvel, xacc, yacc):
        self.xpos = xpos
        self.ypos = ypos
        self.xvel = xvel
        self.yvel = yvel
        self.xacc = xacc
        self.yacc = yacc


def compute_acceleration(particles, system):
    system.distances = []
    for i in range(0, len(particles)):
        particles[i].xacc = 0
        particles[i].yacc = 0
    for i in range(0, len(particles) - 1):
        for j in range(i + 1, len(particles)):
            dx = particles[i].xpos - particles[j].xpos
            dy = particles[i].ypos - particles[j].ypos
            if np.abs(dx) > 0.5 * system.box_length:
                dx *= 1 - system.box_length / np.abs(dx)
            if np.abs(dy) > 0.5 * system.box_length:
                dy *= 1 - system.box_length / np.abs(dy)
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
    particles = np.array([])
    m = int(np.ceil(np.sqrt(system.number_of_particles)))
    d = system.box_length / m
    n = 0
    for i in range(0, m):
        for j in range(0, m):
            if n < system.number_of_particles:
                particle = Particle((i + 0.5)*d, (j + 0.5) * d, 0, 0, 0, 0)
                particles = np.append(particles, particle)
                n += 1
    v = np.sqrt(2 * system.kinetic_energy)
    for i in range(0, system.number_of_particles):
        theta = 2 * np.pi * np.random.randn()
        particles[i].xvel = v * np.cos(theta)
        particles[i].yvel = v * np.sin(theta)
    #particles, system = compute_accelerations(particles, system)
    particles, system = slow.comp_accel(particles, system)
    system.step0 = reset_histogram(system)
    T = system.kinetic_energy
    return particles, T


def time_step(particles, system, time):
    time += system.timestep_length
    system.step += 1
    K = 0
    W_sum = 0
    for i in range(0, system.number_of_particles):
        particles[i].xpos += particles[i].xvel * system.timestep_length + 0.5 * particles[i].xacc * system.timestep_length * system.timestep_length
        particles[i].ypos += particles[i].yvel * system.timestep_length + 0.5 * particles[i].yacc * system.timestep_length * system.timestep_length
        if particles[i].xpos < 0:
            particles[i].xpos += system.box_length
        if particles[i].xpos >= system.box_length:
            particles[i].xpos -= system.box_length
        if particles[i].ypos < 0:
            particles[i].ypos += system.box_length
        if particles[i].ypos >= system.box_length:
            particles[i].ypos -= system.box_length
        particles[i].xvel += 0.5 * particles[i].xacc * system.timestep_length
        particles[i].yvel += 0.5 * particles[i].yacc * system.timestep_length
    #particles, system = compute_accelerations(particles, system)
    particles, system = slow.comp_accel(particles, system)
    for i in range(0, system.number_of_particles):
        particles[i].xvel += 0.5 * particles[i].xacc * system.timestep_length
        particles[i].yvel += 0.5 * particles[i].yacc * system.timestep_length
        v = np.sqrt(particles[i].xvel * particles[i].xvel + particles[i].yvel * particles[i].yvel)
        K += 0.5 * v * v
        bin_s = int(len(system.vel_bins) * v / system.max_vel)
        if bin_s > 0 and bin_s < len(system.vel_bins):
            system.vel_bins[bin_s] += 1
    system.temp_sum += K / system.number_of_particles
    T = system.temp_sum / (system.step - system.step0)
    P = 1 / (3 * system.box_length ** 2) * (2 * K + W_sum)
    return particles, time, T, P, system


def plot_particles(particles, system, T, P):
    x = np.array([])
    y = np.array([])
    for i in range(0, system.number_of_particles):
        x = np.append(x, particles[i].xpos)
        y = np.append(y, particles[i].ypos)
    x_T = []
    for i in range(0, len(T)):
        x_T.append(i*10)
    plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 2)
    ax0 = plt.subplot(gs[:,0])
    ax0.plot(x,y, 'o', markersize=30, markeredgecolor='black')
    ax0.set_xlim([0, system.box_length])
    ax0.set_ylim([0, system.box_length])
    ax0.set_xticks([])
    ax0.set_yticks([])
    if (T != []):
        ax1 = plt.subplot(gs[0, 1])
        #ax1.plot(x_T, T, label = 'Av. Temperature = {:.3e}'.format(np.average(T[-10:])))
        #ax1.set_xlabel('Step')
        #ax1.set_ylabel('Temperature (arbitrary units)')
        #ax1.legend(loc='upper right')
        #ax1.set_xlim(0, np.amax(x_T) + 5)
        bin_width = 0.1
        hist, bin_edges = np.histogram(system.distances, bins=np.arange(0, 12.5, bin_width))
        gr = hist/(system.number_of_particles * (system.number_of_particles / system.box_length ** 2) * np.pi *
                   (bin_edges[:-1]+bin_width/2.) * bin_width)
        #a, b, c = ax1.hist(system.distances, histtype='step', normed=True, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        ax1.plot(bin_edges[:-1]+bin_width/2, gr)
        ax1.set_xlim([0, 8])
        ax1.set_xlabel('$r$')
        ax1.set_ylabel('$g(r)$')
        ax1.set_ylim([0, np.amax(gr)+0.5])
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.close()


def run(number_of_particles, kinetic_energy, number_steps):
    if number_of_particles > 285:
        raise ValueError("Density too high!")
    time = 0
    number_vel_bins = 500
    vel_bins = np.zeros(number_vel_bins)
    system = System(number_of_particles, kinetic_energy, 16., 0.01, 500, 4)
    T_arr = []
    P_arr = []
    particles, T = initialise(system)
    plot_particles(particles, system, T_arr, P_arr)
    for i in range(0, number_steps):
        particles, time, T, P, system = time_step(particles, system, time)
        if i % 10 == 0:
            T_arr.append(T)
            P_arr.append(P)
            plot_particles(particles, system, T_arr, P_arr)
            system.step0 = reset_histogram(system)