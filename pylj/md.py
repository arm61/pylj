import numpy as np
from pylj import force, sample


class System:
    def __init__(self, number_of_particles, kinetic_energy, box_length, timestep_length,
                 max_vel):
        #if number_of_particles > 324:
        #    raise ValueError("Density too high!")
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
        self.forces = np.zeros(int(pairs))
        self.temp_array = []
        self.press_array = [0]
        self.bin_width = 0.1
        self.time = 0


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


def initialise(number_of_particles, kinetic_energy):
    system = System(number_of_particles, kinetic_energy, 16., 0.01, 4)
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
    particles, system = force.compute_forces(particles, system)
    system.step0 = reset_histogram(system)
    return particles, system


def update_pos(particles, system, i):
    particles[i].xpos += particles[i].xvel * system.timestep_length + 0.5 * particles[i].xacc * system.timestep_length * system.timestep_length
    particles[i].ypos += particles[i].yvel * system.timestep_length + 0.5 * particles[i].yacc * system.timestep_length * system.timestep_length
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


def calculate_temperature(particles, system):
    k = 0
    for i in range(0, system.number_of_particles):
        system, v = update_velocity_bins(particles, system, i)
        k += 0.5 * v * v
    system.temp_sum += k / system.number_of_particles
    temp = system.temp_sum / (system.step - system.step0)
    system.temp_array.append(temp)
    particles = force.scale_velo(particles, system)
    return particles, system

def calculate_pressure(system):
    w = (-1. / 3. * np.sum(system.distances * -1 * system.forces))
    system.press_array.append(system.number_of_particles * system.kinetic_energy + w)
    return system


def update_positions(particles, system):
    system.step0 = reset_histogram(system)
    system.step += 1
    for i in range(0, system.number_of_particles):
        particles = update_pos(particles, system, i)
        particles = update_velocities(particles, system, i)
    particles, system = calculate_temperature(particles, system)
    system = calculate_pressure(system)
    return particles, system


def scale_velocities(particles, system):
    for i in range(0, len(particles)):
        particles[i].xvel *= 1 / np.average(system.temp_array)
        particles[i].yvel *= 1 / np.average(system.temp_array)
    return particles


def run(number_of_particles, kinetic_energy, number_steps):
    particles, system = initialise(number_of_particles, kinetic_energy)
    plot_ob = sample.Show(system)
    system.time = 0
    for i in range(0, number_steps):
        particles, system = force.compute_forces(particles, system)
        particles, system = update_positions(particles, system)
        system.time += system.timestep_length
        if system.step % 10 == 0:
            plot_ob.update(particles, system)