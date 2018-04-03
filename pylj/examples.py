from pylj import md, force, sample, util


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
        particles, system = force.compute_forces(particles, system)
        particles, system = md.velocity_verlet(particles, system)
        system.time += system.timestep_length
        if system.step % sample_freq == 0:
            sample_system.update(particles, system, '')