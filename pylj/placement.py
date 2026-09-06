"""The placement of initial configurations."""

from collections.abc import Sequence

import numpy as np

from pylj.configuration import Configuration
from pylj.pairwise import PairPotentials
from pylj.potentials import Species
from pylj.simulation import (
    _check_pair_potentials,
    _check_positive_finite,
    _check_potentials_at_the_cut_off,
    _resolve_cut_off,
)

#: Number of trial positions tried for a single particle by Metropolis
#: placement before it gives up and raises ``ValueError``.
PLACEMENT_ATTEMPTS = 1000


def place_square(
    number_of_particles: int, species: tuple[Species, ...], box: float
) -> Configuration:
    """Place particles on a square lattice.

    The lattice has ``ceil(sqrt(number_of_particles))`` sites along each side of
    the box, and the particles fill those sites in order, taking the species in
    turn. On a lattice with an even number of sites per side, a mixture
    therefore starts out in stripes of one species and then the other.
    Diffusion mixes them over the course of the run. No check is made for
    overlapping particles. A lattice packed too tightly for the potential
    stores a large potential energy, and for a potential with a hard core that
    energy is infinite.

    Args:
        number_of_particles: The number of particles.
        species: The species, assigned to the particles in turn.
        box: The side length of the box, in metres.

    Returns:
        The configuration.
    """
    m = int(np.ceil(np.sqrt(number_of_particles)))
    spacing = box / m
    sites = [((i + 0.5) * spacing, (j + 0.5) * spacing) for i in range(m) for j in range(m)]
    position = np.array(sites[:number_of_particles], dtype=float).reshape(-1, 2)
    species_index = np.arange(number_of_particles) % len(species)
    return Configuration(position, species, species_index, box)


def place_metropolis(
    number_of_particles: int,
    species: tuple[Species, ...],
    box: float,
    pair_potentials: PairPotentials,
    cut_off: float,
    placement_temperature: float,
    rng: np.random.Generator,
) -> Configuration:
    """Place particles one at a time by Metropolis insertion.

    Each particle in turn is given a uniform trial position in the box,
    accepted by :func:`mc.accept` at ``placement_temperature`` on its
    interaction energy with the particles already placed, and redrawn on
    rejection. Before the particle is added it interacts with nothing, so its
    interaction energy with the particles already placed is exactly the energy
    change the insertion causes.

    Placing the particles one after another gives a reasonable starting point,
    not a configuration drawn from equilibrium. The particles avoid the close
    contacts the potential penalises at the placement temperature, and raising
    that temperature makes closer contacts more likely. The run itself brings
    the configuration to equilibrium.

    Args:
        number_of_particles: The number of particles.
        species: The species, assigned to the particles in turn.
        box: The side length of the box, in metres.
        pair_potentials: The potential between each pair of species.
        cut_off: The cut-off, in metres.
        placement_temperature: The temperature of the acceptance, in kelvin.
        rng: The generator to draw trial positions and acceptances from.

    Returns:
        The configuration.

    Raises:
        ValueError: If :data:`PLACEMENT_ATTEMPTS` trial positions are all
            rejected for a single particle. At the highest densities this many
            attempts can reach, success depends on the positions that happen to
            be drawn, so the same call may succeed with one seed and raise with
            another.
    """
    # mc imports this module for place, so the acceptance criterion is
    # imported here rather than at the top of the module.
    from pylj.mc import accept

    species_index = np.arange(number_of_particles) % len(species)
    placed = Configuration(np.zeros((0, 2)), species, species_index[:0], box)
    for i in range(number_of_particles):
        for _attempt in range(PLACEMENT_ATTEMPTS):
            trial = rng.uniform(0, box, size=2)
            energy = placed.insertion_energy(trial, int(species_index[i]), pair_potentials, cut_off)
            if accept(energy, placement_temperature, rng=rng):
                placed = placed.replace(
                    position=np.vstack([placed.position, trial]),
                    species_index=species_index[: i + 1],
                )
                break
        else:
            raise ValueError(
                f"Could not place particle {i + 1} of {number_of_particles} in a "
                f"{box * 1e10:.1f} Angstrom box at a placement temperature of "
                f"{placement_temperature:g} K after {PLACEMENT_ATTEMPTS} attempts; "
                "reduce the number of particles or use a larger box; for a soft "
                "potential, raising placement_temperature tolerates closer contacts."
            )
    return placed


def place(
    number_of_particles: int,
    temperature: float,
    box: float,
    *,
    species: Sequence[Species],
    pair_potentials: PairPotentials,
    init_conf: str,
    placement_temperature: float | None,
    cut_off: float | None,
    rng: np.random.Generator,
) -> tuple[Configuration, float]:
    """Build the initial configuration for a simulation factory.

    Takes the box and cut-off in Angstrom, as the factories do, validates
    the model, checks the potentials at the cut-off before any placement is
    attempted, and places the particles.

    Args:
        number_of_particles: The number of particles.
        temperature: The temperature of the run, in kelvin.
        box: The side length of the box, in Angstrom, from 4 to 600.
        species: The species, assigned to the particles in turn.
        pair_potentials: The potential between each pair of species.
        init_conf: ``'square'`` for a lattice or ``'metropolis'`` for
            sequential Metropolis insertion.
        placement_temperature: The temperature of the Metropolis acceptance
            used by ``'metropolis'``, in kelvin; ``None`` for the run
            temperature.
        cut_off: The cut-off, in Angstrom; ``None`` for 15 Angstrom or half
            the box, whichever is smaller.
        rng: The generator for Metropolis placement.

    Returns:
        The configuration and the cut-off, both in metres.

    Raises:
        ValueError: If no particles are requested, a temperature is
            not positive and finite, the box is outside 4 to 600 Angstrom,
            the cut-off exceeds half the box, the model is incomplete, a
            potential has not died away at the cut-off, ``init_conf`` is
            unknown, or Metropolis placement exhausts its trial budget.
        TypeError: If a pair potential is not a ``PairPotential`` instance.
    """
    if number_of_particles < 1:
        raise ValueError("A simulation needs at least one particle")
    _check_positive_finite("temperature", temperature)
    if placement_temperature is None:
        placement_temperature = temperature
    _check_positive_finite("placement_temperature", placement_temperature)
    if not 4 <= box <= 600:
        raise ValueError(
            f"box must be between 4 and 600 Angstrom, not {box}: below 4 the cell cannot "
            "hold more than one particle, and above 600 the particles are too small to be "
            "seen in the viewer."
        )
    species = tuple(species)
    _check_pair_potentials(species, pair_potentials)
    box_m = box * 1e-10
    cut_off_m = _resolve_cut_off(box_m, None if cut_off is None else cut_off * 1e-10)
    _check_potentials_at_the_cut_off(species, pair_potentials, cut_off_m, temperature, box_m)
    if init_conf == "square":
        configuration = place_square(number_of_particles, species, box_m)
    elif init_conf == "metropolis":
        configuration = place_metropolis(
            number_of_particles,
            species,
            box_m,
            pair_potentials,
            cut_off_m,
            placement_temperature,
            rng,
        )
    else:
        raise ValueError(f"init_conf must be 'square' or 'metropolis', not {init_conf!r}")
    return configuration, cut_off_m
