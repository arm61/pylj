"""The physical model a simulation runs: the species and the potential
between each pair of them."""

import itertools
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Self

from pylj.potentials import PairPotential, Species


def _check_complete(
    species: tuple[Species, ...], pair_potentials: Mapping[tuple[Species, Species], PairPotential]
) -> None:
    """Check that every pair of the species has exactly one potential and that
    no other pair has one."""
    for pair in pair_potentials:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                f"pair_potentials keys must be a pair of species, such as (argon, argon), "
                f"not {pair!r}"
            )
        for one in pair:
            if one not in species:
                raise ValueError(
                    f"pair_potentials has an entry for {pair}, but {one!r} is not one of "
                    f"the species {species}"
                )
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


class Model:
    """The species in a simulation and the potential between each pair of them.

    Every pair of species, including each species with itself, has one
    entry in ``pair_potentials``, keyed by the two species in either order.
    ``single`` builds the model for one species. ``species`` and
    ``pair_potentials`` are read-only.

    Args:
        species: The species, as any sequence of ``Species``. Atoms are
            assigned to them in turn when a configuration is placed.
        pair_potentials: The potential between each pair of species.

    Raises:
        ValueError: If ``species`` is empty or repeats a species, a key of
            ``pair_potentials`` is not a pair, an entry of
            ``pair_potentials`` names something that is not one of the
            species, a pair of species has no potential, or a cross pair is
            given in both orders.
        TypeError: If ``species`` is a single ``Species`` rather than a
            sequence, an item of ``species`` is not a ``Species``,
            ``pair_potentials`` is not a mapping, or a value in
            ``pair_potentials`` is not a ``PairPotential`` instance.
    """

    def __init__(
        self,
        species: Sequence[Species],
        pair_potentials: Mapping[tuple[Species, Species], PairPotential],
    ) -> None:
        if isinstance(species, Species | str):
            raise TypeError(
                "species must be a sequence of Species, such as "
                f"(Species(mass=39.948, name='argon'),), not {species!r}; "
                "for one species, Model.single(species, potential) builds the model"
            )
        self._species = tuple(species)
        for one in self._species:
            if not isinstance(one, Species):
                raise TypeError(
                    f"species must be Species instances, such as Species(mass=39.948, "
                    f"name='argon'), not {one!r}"
                )
        if not isinstance(pair_potentials, Mapping):
            raise TypeError(
                f"pair_potentials must map each pair of species to its potential, such as "
                f"{{(argon, argon): potential}}, not {pair_potentials!r}; for one species, "
                "Model.single(species, potential) builds the model"
            )
        if not self._species:
            raise ValueError("species must name at least one Species")
        if len(set(self._species)) != len(self._species):
            raise ValueError(
                "species must not repeat: two Species that compare equal are one species"
            )
        # A plain dict on the instance, so a model can be pickled and sent
        # to another process; the property hands out a read-only view.
        self._pair_potentials = dict(pair_potentials)
        _check_complete(self._species, self._pair_potentials)

    @property
    def species(self) -> tuple[Species, ...]:
        """The species, as a tuple, in the order atoms are assigned to them."""
        return self._species

    @property
    def pair_potentials(self) -> Mapping[tuple[Species, Species], PairPotential]:
        """The potential between each pair of species, as a read-only mapping."""
        return MappingProxyType(self._pair_potentials)

    @classmethod
    def single(cls, species: Species, potential: PairPotential) -> Self:
        """Build the model for one species.

        Args:
            species: The one species.
            potential: The potential between two atoms of it.
        """
        return cls((species,), {(species, species): potential})

    def potential(self, one: Species, other: Species) -> PairPotential:
        """Return the potential between two species.

        The pair may be given in either order.

        Args:
            one: One species of the pair.
            other: The other species; the same one for a pair of like atoms.

        Raises:
            KeyError: If either species is not in the model.
        """
        for species in (one, other):
            if species not in self._species:
                raise KeyError(f"{species!r} is not a species in this model")
        if (one, other) in self._pair_potentials:
            return self._pair_potentials[(one, other)]
        return self._pair_potentials[(other, one)]

    def __repr__(self) -> str:
        return f"Model(species={self._species!r}, pair_potentials={self._pair_potentials!r})"
