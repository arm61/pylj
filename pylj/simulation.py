"""The checks a simulation makes on its model, the placement of initial
configurations, and the base class of the simulations."""

import itertools
from collections.abc import Sequence

import numpy as np

from pylj import pairwise
from pylj.configuration import Configuration
from pylj.constants import BOLTZMANN
from pylj.pairwise import PairPotentials
from pylj.potentials import PairPotential, Species

#: Number of trial positions tried for a single particle by Metropolis
#: placement before it gives up and raises ``ValueError``.
PLACEMENT_ATTEMPTS = 1000

#: Largest potential energy per particle, in units of k_B T, accepted for an
#: initial configuration.
INITIAL_ENERGY_LIMIT = 10.0


def _check_positive_finite(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is positive and finite."""
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, not {value}")


def _check_pair_potentials(species: Sequence[Species], pair_potentials: PairPotentials) -> None:
    """Check that a model is complete.

    Args:
        species: The species in the model.
        pair_potentials: The potential between each pair of species.

    Raises:
        ValueError: If ``species`` is empty, a pair of species has no entry
            in ``pair_potentials`` in either order, or a cross pair has one
            in both orders.
        TypeError: If a value in ``pair_potentials`` is not a
            ``PairPotential`` instance, such as the class itself.
    """
    if not species:
        raise ValueError("species must name at least one Species")
    for one, other in itertools.combinations_with_replacement(species, 2):
        if (one, other) not in pair_potentials and (other, one) not in pair_potentials:
            raise ValueError(f"pair_potentials has no entry for the pair {one} and {other}")
    for one, other in itertools.combinations(species, 2):
        if (one, other) in pair_potentials and (other, one) in pair_potentials:
            raise ValueError(
                f"pair_potentials has the pair {one} and {other} in both orders; "
                "give each unordered pair once"
            )
    for pair, potential in pair_potentials.items():
        if not isinstance(potential, PairPotential):
            raise TypeError(
                f"pair_potentials[{pair}] must be a PairPotential instance, such as "
                f"LennardJones(epsilon=..., sigma=...), not {potential!r}"
            )


def _check_potentials_at_the_cut_off(
    species: Sequence[Species],
    pair_potentials: PairPotentials,
    cut_off: float,
    temperature: float,
    box: float,
) -> None:
    """Check that every pair potential has died away at the cut-off.

    Truncating the interaction at the cut-off assumes it is negligible
    there. The check is that each pair energy at the cut-off is finite and
    within ``k_B T`` of zero.

    Args:
        species: The species in the model.
        pair_potentials: The potential between each pair of species.
        cut_off: The cut-off, in metres.
        temperature: The temperature, in kelvin.
        box: The box side, in metres; a cut-off of half the box cannot be
            raised, so the remedy is then a larger box.

    Raises:
        ValueError: If any pair potential's energy at the cut-off is not
            finite or larger in magnitude than ``k_B T``.
    """
    from_box = cut_off >= box / 2
    where = f"the cut-off of {cut_off * 1e10:.1f} Angstrom"
    if from_box:
        where += " (half the box)"
    remedy = "Use a larger box" if from_box else "Use a larger box or cut-off"
    for one, other in itertools.combinations_with_replacement(species, 2):
        potential = pairwise.pair_potential(pair_potentials, one, other)
        energy = float(potential.energies(np.array([cut_off]))[0])
        pair = (
            f"{type(potential).__name__} between {one.name or 'particles'} and "
            f"{other.name or 'particles'}"
        )
        if not np.isfinite(energy):
            raise ValueError(
                f"{pair} is infinite at {where}: its hard core is wider than the cut-off. {remedy}."
            )
        if abs(energy) > BOLTZMANN * temperature:
            raise ValueError(
                f"{pair} is still {energy / (BOLTZMANN * temperature):+.3g} k_B T at {where}; "
                "the cut-off assumes the interaction has died away there. "
                f"{remedy}, or check the parameter units: metres and joules are expected."
            )


def _check_initial_energy(energy: float, number_of_particles: int, temperature: float) -> None:
    """Refuse a starting configuration that stores far more potential energy
    than thermal energy.

    Potential energy stored in an initial configuration is released as
    motion over the first steps and heats the run. A configuration holding
    more than :data:`INITIAL_ENERGY_LIMIT` k_B T per particle has particles
    too close together for its temperature.

    Args:
        energy: The total pair energy of the configuration, in joules.
        number_of_particles: The number of particles.
        temperature: The temperature of the run, in kelvin.

    Raises:
        ValueError: If the energy is not finite, or exceeds
            :data:`INITIAL_ENERGY_LIMIT` k_B T per particle.
    """
    remedy = (
        "Use fewer particles or a larger box; init_conf='metropolis' places particles by "
        "energy, and a lower placement_temperature there keeps them further apart."
    )
    if not np.isfinite(energy):
        raise ValueError(
            f"The initial pair energy is not finite: particles sit inside a hard core. {remedy}"
        )
    per_particle = energy / (number_of_particles * BOLTZMANN * temperature)
    if per_particle > INITIAL_ENERGY_LIMIT:
        raise ValueError(
            f"The initial configuration stores {per_particle:.3g} k_B T of potential energy "
            f"per particle, above the limit of {INITIAL_ENERGY_LIMIT:g}: its particles are "
            f"too close together for {temperature:g} K. {remedy}"
        )


def _resolve_cut_off(box: float, cut_off: float | None) -> float:
    """The cut-off in metres: as given, or 15 Angstrom or half the box,
    whichever is smaller.

    Raises:
        ValueError: If a given cut-off is not positive and finite, or exceeds
            half the box, beyond which the minimum image convention fails.
    """
    if cut_off is None:
        return min(15e-10, box / 2)
    _check_positive_finite("cut_off", cut_off)
    if cut_off > box / 2:
        raise ValueError(
            f"The cut-off of {cut_off * 1e10:.1f} Angstrom exceeds half the box of "
            f"{box * 1e10:.1f} Angstrom; the minimum image convention needs a cut-off of at "
            "most half the box."
        )
    return cut_off


def place_square(
    number_of_particles: int, species: tuple[Species, ...], box: float
) -> Configuration:
    """Place particles on a square lattice.

    The lattice has ``ceil(sqrt(number_of_particles))`` sites along each
    side of the box and the particles fill it in order, each assigned to the
    species in turn. No overlap check is made: a lattice too dense for the
    potential stores a large, or for a hard core infinite, energy.

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
    rejection. Inserting from vacuum makes that energy the energy change of
    the insertion.

    Sequential insertion is an initialiser, not an equilibrium sample: the
    placed configuration avoids the close contacts the potential penalises
    at the placement temperature, closer contacts become likelier as that
    temperature rises, and the run equilibrates it.

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
        ValueError: If :data:`PLACEMENT_ATTEMPTS` trial positions are
            rejected for a single particle. Near the density the budget
            allows, whether that happens depends on the draw, so the same
            call can place with one seed and raise with another.
    """
    # mc imports this module for its base classes, so the criterion is
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
        ValueError: If there are fewer than one particle, a temperature is
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
