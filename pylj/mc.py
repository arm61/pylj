from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pylj.constants import BOLTZMANN


@dataclass(frozen=True)
class Proposal:
    """A proposed configuration for a Monte Carlo move.

    Args:
        xposition: The proposed x position of every particle, in metres.
        yposition: The proposed y position of every particle, in metres.
        energy_change: The energy of the proposed configuration minus that
            of the current one, in joules.
    """

    xposition: NDArray[np.float64]
    yposition: NDArray[np.float64]
    energy_change: float


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
    calculate the initial pair energies and their total.

    Parameters
    ----------
    number_of_particles: int
        Number of particles to simulate.
    temperature: float
        Temperature of the simulation, in kelvin.
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
        System information, with ``energy`` set to the total pair energy.
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
    system.energy = float(system.energies.sum())
    return system


initialize = initialise  # US spelling


def accept(delta_energy, temperature, *, n=None, rng=None):
    """Apply the Metropolis criterion to an energy change.

    A change that does not raise the energy is always accepted, without
    drawing a random number. A change that raises it by ``delta_energy`` is
    accepted with probability ``exp(-delta_energy / (k_B temperature))``.

    Parameters
    ----------
    delta_energy: float
        The energy of the proposed configuration minus that of the current
        one, in joules.
    temperature: float
        Temperature of the simulation, in kelvin.
    n: float, optional
        The random number against which the acceptance probability is
        tested. By default one is drawn from ``rng``.
    rng: numpy.random.Generator, optional
        The random number generator to draw ``n`` from. By default an
        unseeded generator is used, so each call draws afresh.

    Returns
    -------
    bool
        True if the proposed configuration should be accepted.
    """
    if delta_energy <= 0:
        return True
    if n is None:
        if rng is None:
            rng = np.random.default_rng()
        n = rng.random()
    return bool(n < np.exp(-delta_energy / (BOLTZMANN * temperature)))


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
