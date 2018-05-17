from pylj import md, comp, sample

def md_nvt(number_of_particles, temperature, box_length, number_of_steps, sample_frequency):
    """This is an example NVT (constant number of particles, system volume and temperature) simulation. The temperature
    is controlled using a velocity rescaling.

    Parameters
    ----------
    number_of_particles: int
        Number of particles to simulate.
    temperature: float
        Initial temperature of the particles and the temperature of the heat bath, in Kelvin.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    number_of_steps: int
        Number of integration steps to be performed.
    sample_frequency: int
        Frequency with which the visualisation environment is to be updated.

    Returns
    -------
    System
        System information.
    """
    system = md.initialise(number_of_particles, temperature, box_length, 'square')
    sample_system = sample.Interactions(system)
    system.time = 0
    for i in range(0, number_of_steps):
        system.particles, system.distances, system.forces = comp.compute_forces(system.particles,
                                                                                system.distances,
                                                                                system.forces, system.box_length)
        system.particles = md.velocity_verlet(system.particles, system.timestep_length, system.box_length)
        system = md.sample(system.particles, system.box_length, system.initial_particles, system)
        system.particles = comp.heat_bath(system.particles, system.temperature_sample, temperature)
        system.time += system.timestep_length
        system.step += 1
        if system.step % sample_frequency == 0:
            sample_system.update(system)
    return system


def md_nve(number_of_particles, temperature, box_length, number_of_steps, sample_frequency):
    """This is an example NVE (constant number of particles, system volume and system energy) simulation.

    Parameters
    ----------
    number_of_particles: int
        Number of particles to simulate.
    temperature: float
        Initial temperature of the particles, in Kelvin.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    number_of_steps: int
        Number of integration steps to be performed.
    sample_frequency: int
        Frequency with which the visualisation environment is to be updated.

    Returns
    -------
    System
        System information.
    """
    system = md.initialise(number_of_particles, temperature, box_length, 'square')
    sample_system = sample.Interactions(system)
    system.time = 0
    for i in range(0, number_of_steps):
        system.particles, system.distances, system.forces = comp.compute_forces(system.particles,
                                                                                system.distances,
                                                                                system.forces, system.box_length)
        system.particles = md.velocity_verlet(system.particles, system.timestep_length, system.box_length)
        system = md.sample(system.particles, system.box_length, system.initial_particles, system)
        system.time += system.timestep_length
        system.step += 1
        if system.step % sample_frequency == 0:
            sample_system.update(system)
    return system


def periodic_boundary(number_of_steps, temperature):
    """This is a piece of exemplary code to show a single particle traveling across the periodic boundary.

    Parameters
    ----------
    number_of_steps: int
        Number of step in simulation.
    temperature: float
        Temperature of simulation.
    """
    number_of_particles = 1
    sample_freq = 10
    system = md.initialise(number_of_particles, temperature, 50, 'square')
    sampling = sample.JustCell(system)
    system.time = 0
    for i in range(0, number_of_steps):
        system.particles, system.distances, system.forces = comp.compute_forces(system.particles,
                                                                                system.distances,
                                                                                system.forces, system.box_length)
        system.particles = md.velocity_verlet(system.particles, system.timestep_length, system.box_length)
        system = md.sample(system.particles, system.box_length, system.initial_particles, system)
        system.particles = comp.heat_bath(system.particles, system.temperature_sample, temperature)
        system.time += system.timestep_length
        system.step += 1
        if system.step % sample_freq == 0:
            sampling.update(system)