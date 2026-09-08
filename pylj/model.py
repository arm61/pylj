"""The physical model a simulation runs: the species and the potential
between each pair of them."""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from pylj.potentials import PairPotential, Species


def _check_complete(
    species: tuple[Species, ...], pair_potentials: Mapping[tuple[Species, Species], PairPotential]
) -> None:
    """Check that every pair of species has exactly one potential.

    Raises:
        ValueError: If ``species`` is empty or repeats a species, a pair of
            species has no entry in ``pair_potentials`` in either order, or
            a cross pair has one in both orders.
        TypeError: If a value in ``pair_potentials`` is not a
            ``PairPotential`` instance, such as the class itself.
    """
    if not species:
        raise ValueError("species must name at least one Species")
    if len(set(species)) != len(species):
        raise ValueError("species must not repeat: two Species that compare equal are one species")
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


@dataclass(frozen=True)
class Model:
    """The species in a simulation and the potential between each pair of them.

    Every pair of species, including each species with itself, has one
    entry in ``pair_potentials``, keyed by the two species in either order.
    ``single`` builds the model for one species.

    Args:
        species: The species. Atoms are assigned to them in turn when a
            configuration is placed.
        pair_potentials: The potential between each pair of species.

    Raises:
        ValueError: If ``species`` is empty or repeats a species, a pair of
            species has no potential, or a cross pair is given in both
            orders.
        TypeError: If a value in ``pair_potentials`` is not a
            ``PairPotential`` instance.
    """

    species: tuple[Species, ...]
    pair_potentials: Mapping[tuple[Species, Species], PairPotential]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_potentials", dict(self.pair_potentials))
        _check_complete(self.species, self.pair_potentials)

    @classmethod
    def single(cls, species: Species, potential: PairPotential) -> Self:
        """The model for one species interacting through one potential."""
        return cls((species,), {(species, species): potential})

    def potential(self, one: Species, other: Species) -> PairPotential:
        """Return the potential acting between two species, in either order.

        Raises:
            KeyError: If either species is not in the model.
        """
        for species in (one, other):
            if species not in self.species:
                raise KeyError(f"{species.name or species} is not a species in this model")
        if (one, other) in self.pair_potentials:
            return self.pair_potentials[(one, other)]
        return self.pair_potentials[(other, one)]
