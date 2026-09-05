from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pylj.constants import BOLTZMANN


@dataclass(frozen=True)
class Proposal:
    """A proposed configuration for a Monte Carlo move.

    The energy change is relative to the configuration that was current when
    the proposal was made, so a proposal is applied to that configuration.

    Attributes:
        xposition: The proposed x position of every particle, in metres.
        yposition: The proposed y position of every particle, in metres.
        energy_change: The energy of the proposed configuration minus that
            of the configuration it was proposed from, in joules.
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
    placement_temperature=None,
    seed=None,
):
    """Initialise the particle positions (square lattice or Metropolis
    insertion) and calculate the initial pair energies and their total.

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
        - 'square', a square lattice
        - 'metropolis', sequential Metropolis insertion at
          ``placement_temperature``
    species: sequence of Species
        Required, keyword only. The species in the system; particles are
        assigned to them in turn.
    pair_potentials: mapping of (Species, Species) to PairPotential
        Required, keyword only. The potential acting between each pair of
        species, including each species with itself. Only the pair energies
        are evaluated, so a potential with no finite force, such as the
        square well, can be used.
    placement_temperature: float (optional)
        Temperature, in kelvin, of the Metropolis acceptance used by
        ``init_conf='metropolis'``; by default the run temperature. A
        parameter of the placement, not a thermodynamic temperature: raising
        it tolerates more overlap, lowering it packs more tightly.
    seed: int (optional)
        Seed for the random number generator used to place an initial
        configuration and make the Monte Carlo moves. The same seed
        reproduces the same run.

    Returns
    -------
    System
        System information; ``energy`` is the total pair energy.

    Raises
    ------
    ValueError
        If the initial pair energy is not finite, as when particles of a
        hard-core potential sit inside its diameter.
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
        placement_temperature=placement_temperature,
        seed=seed,
    )
    if not np.isfinite(system.energy):
        raise ValueError(
            "The initial pair energy is not finite: particles sit inside a hard core. "
            "Use init_conf='metropolis', fewer particles or a larger box."
        )
    return system


initialize = initialise  # US spelling


def accept(
    energy_change: float,
    temperature: float,
    *,
    random_number: float | None = None,
    rng: np.random.Generator | None = None,
) -> bool:
    """Apply the Metropolis criterion to an energy change.

    A change that does not raise the energy is always accepted, without
    drawing a random number. A change that raises it by ``energy_change`` is
    accepted with probability ``exp(-energy_change / (k_B temperature))``.

    Args:
        energy_change: The energy of the proposed configuration minus that
            of the current one, in joules.
        temperature: Temperature of the simulation, in kelvin.
        random_number: The uniform random number the acceptance probability
            is tested against. By default one is drawn from ``rng``.
        rng: The generator to draw from; pass the system's ``rng`` for a
            reproducible run. By default an unseeded generator is used.

    Returns:
        True if the proposed configuration should be accepted.
    """
    if energy_change <= 0:
        return True
    if random_number is None:
        if rng is None:
            rng = np.random.default_rng()
        random_number = rng.random()
    return bool(random_number < np.exp(-energy_change / (BOLTZMANN * temperature)))


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
