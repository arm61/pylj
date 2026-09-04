import numpy as np

from pylj.constants import BOLTZMANN


def initialise(
    number_of_particles,
    temperature,
    box_length,
    init_conf,
    *,
    species,
    pair_potentials,
    seed=None,
):
    """Initialise the particle positions (square or random arrangement) and
    calculate the initial pair energies.

    Parameters
    ----------
    number_of_particles: int
        Number of particles to simulate.
    temperature: float
        Initial temperature of the particles, in kelvin.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    init_conf: string
        The way that the particles are initially positioned. Should be one of:
        - 'square'
        - 'random'
        Both raise ``ValueError`` if the particles cannot be placed without
        their repulsive cores overlapping.
    species: sequence of Species
        Required, keyword only. The species in the system; particles are
        assigned to them in turn.
    pair_potentials: mapping of (Species, Species) to PairPotential
        Required, keyword only. The potential acting between each pair of
        species, including each species with itself. Only the pair energies
        are evaluated, so a potential with no finite force, such as the
        square well, can be used.
    seed: int (optional)
        Seed for the random number generator used to place a random initial
        configuration and make the Monte Carlo moves. The same seed
        reproduces the same run.

    Returns
    -------
    System
        System information.
    """
    from pylj import util

    system = util.System(
        number_of_particles,
        temperature,
        box_length,
        species=species,
        pair_potentials=pair_potentials,
        simulation="mc",
        init_conf=init_conf,
        seed=seed,
    )
    system.compute_energy()
    system.old_energy = system.energies.sum()
    return system


initialize = initialise  # US spelling


def sample(total_energy, system):
    """Sample parameters of interest in the simulation.

    Parameters
    ----------
    total_energy: float
        The total system energy.
    system: System
        Details about the whole system

    Returns
    -------
    System:
        Details about the whole system, with the new step and energy
        appended to the appropriate arrays.
    """
    system.energy_sample = np.append(system.energy_sample, total_energy)
    system.step_sample = np.append(system.step_sample, system.step)
    return system


def select_random_particle(particles, rng):
    """Selects a random particle from the system and return its index and
    current position.

    Parameters
    ----------
    particles: util.particle.dt, array_like
        Information about the particles.
    rng: numpy.random.Generator
        The random number generator to draw the particle from.

    Returns
    -------
    int:
        Index of the random particle that is selected.
    float, array_like:
        The current position of the chosen particle.
    """
    random_particle = int(rng.integers(particles.size))
    position_store = [
        particles["xposition"][random_particle],
        particles["yposition"][random_particle],
    ]
    return random_particle, position_store


def get_new_particle(particles, random_particle, box_length, rng):
    """Generates a new position for the particle.

    Parameters
    ----------
    particles: util.particle.dt, array_like
        Information about the particles.
    random_particle: int
        Index of the random particle that is selected.
    box_length: float
        Length of a single dimension of the simulation square, in metres.
    rng: numpy.random.Generator
        The random number generator to draw the position from.

    Returns
    -------
    util.particle.dt, array_like
        Information about the particles, updated to account for the change of
        selected particle position.
    """
    particles["xposition"][random_particle] = rng.uniform(0, box_length)
    particles["yposition"][random_particle] = rng.uniform(0, box_length)
    return particles


def accept(new_energy):
    """Accept the move.

    Parameters
    ----------
    new_energy: float
        A new total energy for the system.

    Returns
    -------
    float:
        A new total energy for the system.
    """
    return new_energy


def reject(position_store, particles, random_particle):
    """Reject the move and return the particle to the original place.

    Parameters
    ----------
    position_store: float, array_like
        The x and y positions previously held by the particle that has moved.
    particles: util.particle.dt, array_like
        Information about the particles.
    random_particle: int
        Index of the random particle that is selected.

    Returns
    -------
    util.particle.dt, array_like
        Information about the particles, with the particle returned to the
        original position
    """
    particles["xposition"][random_particle] = position_store[0]
    particles["yposition"][random_particle] = position_store[1]
    return particles


def metropolis(temperature, old_energy, new_energy, n=None, rng=None):
    """Determines if the move is accepted or rejected based on the metropolis
    condition. A move that does not raise the energy is always accepted
    without drawing a random number; ``n`` and ``rng`` apply only to moves
    that raise it.

    Parameters
    ----------
    temperature: float
        Simulation temperature, in kelvin.
    old_energy: float
        The total energy of the simulation in the previous configuration.
    new_energy: float
        The total energy of the simulation in the current configuration.
    n: float, optional
        The random number against which the Metropolis condition is tested.
        By default one is drawn from ``rng``.
    rng: numpy.random.Generator, optional
        The random number generator to draw ``n`` from. By default an
        unseeded generator is used, so each call draws afresh.

    Returns
    -------
    bool
        True if the move should be accepted.
    """
    energy_difference = new_energy - old_energy
    if energy_difference <= 0:
        return True
    if n is None:
        if rng is None:
            rng = np.random.default_rng()
        n = rng.random()
    beta = 1 / (BOLTZMANN * temperature)
    metropolis_factor = np.exp(-beta * energy_difference)
    return bool(n < metropolis_factor)
