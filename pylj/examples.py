from pylj import md, comp, sample, util


def periodic_boundary(number_of_steps, temperature):
    """This is an examplary piece of code to show a single particle travelling across the periodic boundary.

    Parameters
    ----------
    number_of_steps: int
        Number of steps to be taken in the md simulation.
    temperature: float
        Temperature of the simulation
    """
    number_of_particles = 1
    sample_freq = 10
    particles, system = md.initialise(number_of_particles, temperature, 0.01, util.set_particles_random)
    sample_system = sample.JustCell(system)
    for i in range(0, number_of_steps):
        particles, system = comp.compute_forces(particles, system)
        particles, system = md.velocity_verlet(particles, system)
        system.time += system.timestep_length
        if system.step % sample_freq == 0:
            sample_system.update(particles, system, '')


def md_nvt(number_of_particles, temperature, box_length, number_of_steps, sample_frequency):
    # Creates the visualisation environment
    %matplotlib notebook
    # Initialise the system
    system = md.initialise(number_of_particles, temperature, 0.001, box_length, 'square')
    # This sets the sampling class
    sample_system = sample.Interactions(system)
    # Start at time 0
    system.time = 0
    # Begin the molecular dynamics loop
    for i in range(0, number_of_steps):
        # At each step, calculate the forces on each particle
        # and get acceleration
        system = comp.compute_forces(system)
        # Run the equations of motion integrator algorithm
        system = md.velocity_verlet(system)
        # Allow the system to interact with a heat bath
        system = comp.heat_bath(system, temperature)
        # Iterate the time
        system.time += system.timestep_length
        system.step += 1
        # At a given frequency sample the positions and plot the RDF
        if system.step % sample_frequency == 0:
            sample_system.update(system)
    return system


def md_nve(number_of_particles, temperature, box_length, number_of_steps, sample_frequency):
    # Creates the visualisation environment
    %matplotlib notebook
    # Initialise the system
    system = md.initialise(number_of_particles, temperature, 0.001, box_length, 'square')
    # This sets the sampling class
    sample_system = sample.Interactions(system)
    # Start at time 0
    system.time = 0
    # Begin the molecular dynamics loop
    for i in range(0, number_of_steps):
        # At each step, calculate the forces on each particle
        # and get acceleration
        system = comp.compute_forces(system)
        # Run the equations of motion integrator algorithm
        system = md.velocity_verlet(system)
        # Iterate the time
        system.time += system.timestep_length
        system.step += 1
        # At a given frequency sample the positions and plot the RDF
        if system.step % sample_frequency == 0:
            sample_system.update(system)
    return system

