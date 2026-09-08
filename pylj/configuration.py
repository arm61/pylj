"""Configurations: the physical state a simulation evolves."""

import dataclasses
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pylj import pairwise
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN
from pylj.model import Model
from pylj.potentials import Species


@dataclass(frozen=True, eq=False)
class PairData:
    """The distance, separation, energy and force of every pair of atoms.

    Each pair appears once, in the order :func:`pairwise.dist` returns them:
    the atom with the lower index first.

    Attributes:
        distance: The minimum-image distance between each pair, in metres.
        separation: The minimum-image separation ``r_i - r_j`` of each pair,
            shape ``(M, 2)``, in metres.
        energy: The energy of each pair, in joules; zero beyond the cut-off.
        radial_force: The radial force on each pair, in newtons, positive
            where repulsive and zero beyond the cut-off; ``None`` when the
            forces were not evaluated.
    """

    distance: NDArray[np.float64]
    separation: NDArray[np.float64]
    energy: NDArray[np.float64]
    radial_force: NDArray[np.float64] | None

    @property
    def virial(self) -> float:
        """The sum over pairs of the radial force times the distance, in joules."""
        return float(np.sum(_radial_force(self) * self.distance))


def _radial_force(pairs: PairData) -> NDArray[np.float64]:
    """Return the radial forces, raising if the pair data was evaluated without them."""
    if pairs.radial_force is None:
        raise ValueError("The pair forces were not evaluated; call pairs(..., forces=True)")
    return pairs.radial_force


@dataclass(frozen=True, eq=False)
class Configuration:
    """Where the atoms are: the state a Monte Carlo simulation evolves.

    A configuration cannot be changed once it is made, and it holds no
    description of how the atoms interact. Each method that evaluates the
    interactions between atoms is given a ``Model`` and a ``cut_off`` when it
    is called, so the same configuration can be evaluated under different
    potentials.
    Everything is in SI units.

    Attributes:
        position: The position of each atom, shape ``(N, 2)``, in
            metres; a simulation keeps them wrapped into the box.
        species: The species in the configuration.
        species_index: The index in ``species`` of each atom's species,
            shape ``(N,)``.
        box: The side length of the square periodic box, in metres.

    Raises:
        ValueError: If the array shapes disagree, ``species`` is empty, an
            entry of ``species_index`` does not correspond to one of the
            species, or the box is not positive and finite.
    """

    position: NDArray[np.float64]
    species: tuple[Species, ...]
    species_index: NDArray[np.int64]
    box: float

    def __post_init__(self) -> None:
        if self.position.ndim != 2 or self.position.shape[1] != 2:
            raise ValueError(f"position must have shape (N, 2), not {self.position.shape}")
        n = self.position.shape[0]
        integer = np.issubdtype(self.species_index.dtype, np.integer)
        if self.species_index.shape != (n,) or not integer:
            raise ValueError(
                f"species_index must be an integer array of shape ({n},), one entry per "
                f"atom, not {self.species_index.dtype} of shape {self.species_index.shape}"
            )
        if not self.species:
            raise ValueError("species must name at least one Species")
        if n and not (
            0 <= self.species_index.min() and self.species_index.max() < len(self.species)
        ):
            raise ValueError(f"species_index must index the {len(self.species)} species")
        if not (np.isfinite(self.box) and self.box > 0):
            raise ValueError(f"box must be positive and finite, not {self.box}")

    @property
    def number_of_atoms(self) -> int:
        """The number of atoms."""
        return self.position.shape[0]

    @property
    def masses(self) -> NDArray[np.float64]:
        """The mass of each atom, in kilograms, from its species."""
        masses = np.array([one.mass for one in self.species], dtype=float) * ATOMIC_MASS_UNIT
        return masses[self.species_index]

    def replace(self, **changes: Any) -> Self:
        """Return a copy with the given fields replaced. The copy is checked in the same
        way as any other configuration."""
        return dataclasses.replace(self, **changes)

    def without(self, index: int) -> Self:
        """Return a copy with one atom removed.

        Args:
            index: The index of the atom to remove.

        Returns:
            The configuration without that atom.
        """
        arrays: dict[str, Any] = {
            field.name: np.delete(getattr(self, field.name), index, axis=0)
            for field in dataclasses.fields(self)
            if isinstance(getattr(self, field.name), np.ndarray)
        }
        return dataclasses.replace(self, **arrays)

    def pairs(self, model: Model, cut_off: float, *, forces: bool = False) -> PairData:
        """Evaluate every pair of atoms under the model.

        Each pair is separated by its minimum-image distance, and the potential
        for the two species it joins gives its energy. A pair further apart
        than the cut-off contributes nothing. A pair closer than its
        potential's ``min_separation`` is forbidden: its energy is infinite,
        and if forces are requested the evaluation raises, since the
        simulation has entered a region where the model is unphysical. The
        forces are evaluated only when ``forces`` is requested, so a
        potential that has no finite force, such as the square well, can
        still be used here.

        Args:
            model: The species and the potential between each pair of them.
            cut_off: The separation beyond which a pair contributes nothing,
                in metres.
            forces: Whether to evaluate the radial forces as well.

        Returns:
            The pair distances, separations, energies and, if requested,
            forces.

        Raises:
            ValueError: If forces are requested and a pair is closer than
                its potential's ``min_separation``.
        """
        distance, separation = pairwise.dist(self.position, self.box)
        energy = np.zeros(distance.size)
        force = np.zeros(distance.size) if forces else None
        for mask, type_1, type_2 in pairwise.species_pairs(self.species_index):
            potential = model.potential(self.species[type_1], self.species[type_2])
            energy[mask] = potential.energies(distance[mask])
            forbidden = mask & (distance < potential.min_separation)
            if forbidden.any():
                if force is not None:
                    raise ValueError(
                        f"A pair of {self.species[type_1].name or 'atoms'} and "
                        f"{self.species[type_2].name or 'atoms'} is "
                        f"{distance[forbidden].min() * 1e10:.2f} Angstrom apart, closer than the "
                        f"{potential.min_separation * 1e10:.2f} Angstrom below which "
                        f"{type(potential).__name__} is unphysical: the simulation has collapsed."
                    )
                energy[forbidden] = np.inf
            if force is not None:
                force[mask] = potential.forces(distance[mask])
        beyond = distance > cut_off
        energy[beyond] = 0.0
        if force is not None:
            force[beyond] = 0.0
        return PairData(distance, separation, energy, force)

    def potential_energy(self, model: Model, cut_off: float) -> float:
        """The total pair energy, in joules."""
        return float(self.pairs(model, cut_off).energy.sum())

    def forces(self, model: Model, cut_off: float) -> NDArray[np.float64]:
        """The net force on each atom, shape ``(N, 2)``, in newtons."""
        pairs = self.pairs(model, cut_off, forces=True)
        radial = _radial_force(pairs)
        i, j = np.triu_indices(self.number_of_atoms, 1)
        # Each pair's radial force acts along its separation, pushing
        # atom i one way and atom j the other.
        pair_force = (radial / pairs.distance)[:, None] * pairs.separation
        force = np.zeros((self.number_of_atoms, 2))
        np.add.at(force, i, pair_force)
        np.add.at(force, j, -pair_force)
        return force

    def virial(self, model: Model, cut_off: float) -> float:
        """The sum over pairs of the radial force times the distance, in joules."""
        return self.pairs(model, cut_off, forces=True).virial

    def insertion_energy(
        self,
        position: ArrayLike,
        species_index: int,
        model: Model,
        cut_off: float,
    ) -> float:
        """Return the interaction energy of one added atom with the atoms already
        in the configuration.

        Args:
            position: The ``(x, y)`` position of the added atom, in metres.
            species_index: The index in ``species`` of its species.
            model: The species and the potential between each pair of them.
            cut_off: The separation beyond which a pair contributes nothing,
                in metres.

        Returns:
            The sum of its pair energies, in joules; zero for an empty
            configuration.
        """
        separation = pairwise.minimum_image(
            np.asarray(position, dtype=float) - self.position, self.box
        )
        distance = np.linalg.norm(separation, axis=1)
        energy = np.zeros(distance.size)
        for other in np.unique(self.species_index):
            mask = self.species_index == other
            potential = model.potential(self.species[species_index], self.species[int(other)])
            energy[mask] = potential.energies(distance[mask])
            energy[mask & (distance < potential.min_separation)] = np.inf
        energy[distance > cut_off] = 0.0
        return float(energy.sum())


@dataclass(frozen=True, eq=False)
class MDConfiguration(Configuration):
    """Where the atoms are and how fast they move: the state a molecular
    dynamics simulation evolves.

    Attributes:
        velocity: The velocity of each atom, shape ``(N, 2)``, in
            metres per second.
        unwrapped: The position of each atom without periodic wrapping,
            shape ``(N, 2)``, in metres, for the mean squared displacement.

    Raises:
        ValueError: If ``velocity`` or ``unwrapped`` is not the shape of
            ``position``, or for anything :class:`Configuration` rejects.
    """

    velocity: NDArray[np.float64]
    unwrapped: NDArray[np.float64]

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("velocity", "unwrapped"):
            if getattr(self, name).shape != self.position.shape:
                raise ValueError(
                    f"{name} must have the shape of position, {self.position.shape}, "
                    f"not {getattr(self, name).shape}"
                )

    def kinetic_energy(self) -> float:
        """The total kinetic energy, in joules."""
        return float(0.5 * np.sum(self.masses * np.sum(self.velocity**2, axis=1)))

    def temperature(self) -> float:
        """The instantaneous temperature, in kelvin.

        ``MDSimulation.initialise`` sets the centre of mass at rest, and the
        pair forces cannot set it moving. Two of the ``2N`` velocity
        components are therefore fixed by that condition, leaving ``2N - 2``
        components to carry thermal energy. The temperature is the kinetic
        energy divided by ``(N - 1) k_B``.

        Raises:
            ValueError: If there are fewer than two atoms.
        """
        if self.number_of_atoms < 2:
            raise ValueError(
                "The temperature needs at least two atoms: with one atom there "
                "is no thermal motion once the centre-of-mass velocity is removed."
            )
        return self.kinetic_energy() / ((self.number_of_atoms - 1) * BOLTZMANN)

    def msd(self, initial: "MDConfiguration") -> float:
        """Return the mean squared displacement since an earlier configuration.

        The unwrapped positions are used, so an atom that crosses the edge
        of the box and reappears on the other side counts as having travelled
        the whole way.

        Args:
            initial: The configuration to measure the displacement from.

        Returns:
            The mean squared displacement, in metres squared.
        """
        displacement = self.unwrapped - initial.unwrapped
        return float(np.mean(np.sum(displacement**2, axis=1)))
