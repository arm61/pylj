import numpy as np
from pylj import util, force, sample


def initialise(number_of_particles, arrangement, threshold, max_steps):
    """Initial particle positions.

    Parameters
    ----------
    number_of_particles: int
        Number of particles in the system.
    arrangement: func
        A function to set the arrangement of the particles.

    Returns
    -------
    Particle array
        An array containing the initial particle positions, velocities and accelerations set.
    System
        System information.
    """
    system = util.System(number_of_particles, 0, 16., 0, threshold=threshold, max_steps=max_steps)
    particles = arrangement(system)
    return particles, system


def steepest_descent(particles, system, sample_system, alpha):
    previous_energy = system.threshold + 1
    current_energy = 0
    counter = 0
    while counter < system.max_steps:
        if np.abs(previous_energy - current_energy) < system.threshold:
            break
        particles = force.calculate_energy_and_force(particles, system)
        energy = []
        for i in range(0, len(particles)):
            energy.append(particles[i].energy)
        k = np.argmax(energy)
        xforce = particles[k].xforce
        yforce = particles[k].yforce
        previous_energy = current_energy
        current_energy = np.sum(energy)
        if np.isnan(current_energy) and sample_system:
            sample_system.update(particles, system, '\nNAN ERROR'.format(counter))
            break
        if counter % 10 == 0 and sample_system:
            sample_system.update(particles, system, 'EM - Steps = {:d}\nEnergy change = {:.3e}'.format(counter, np.abs(
                previous_energy - current_energy)))
        particles[k].xpos = particles[k].xpos - (xforce * alpha)
        particles[k].xpos = particles[k].xpos % system.box_length
        particles[k].ypos = particles[k].ypos - (yforce * alpha)
        particles[k].ypos = particles[k].ypos % system.box_length
        counter += 1
    if sample_system and not np.isnan(current_energy):
        sample_system.update(particles, system, 'EM - Steps = {:d}\nEnergy change = {:.3e}'.format(counter,np.abs(
            previous_energy - current_energy)))
    return particles


def newton_raphson(particles, system, sample_system):
    previous_energy = system.threshold + 1
    current_energy = 0
    counter = 0
    while np.abs(previous_energy - current_energy) > system.threshold or counter < system.max_steps:
        energy = force.calculate_energy_and_force(particles, system)
        k = np.argmax(energy[0])
        xforce = energy[1][k]
        yforce = energy[2][k]
        xforcedash = energy[3][k]
        yforcedash = energy[4][k]
        previous_energy = current_energy
        current_energy = np.sum(energy[0])
        if counter % 10 == 0:
            sample_system.update(particles, 'EM - Steps = {:d}'.format(counter))
        particles[k].xpos = particles[k].xpos - (xforce / xforcedash)
        particles[k].xpos = particles[k].xpos % system.box_length
        particles[k].ypos = particles[k].ypos - (yforce / yforcedash)
        particles[k].ypos = particles[k].ypos % system.box_length
        counter += 1
    if sample_system:
        sample_system.update(particles, 'EM - Steps = {:d}'.format(counter))
    return particles