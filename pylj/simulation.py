"""The checks a simulation makes on its model, and the base class of the
simulations."""

import itertools
from collections.abc import Sequence

import numpy as np

from pylj import pairwise
from pylj.constants import BOLTZMANN
from pylj.pairwise import PairPotentials
from pylj.potentials import PairPotential, Species

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
