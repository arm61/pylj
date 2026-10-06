"""Atom configurations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pylj import pairwise
from pylj.constants import BOLTZMANN, KJ_PER_MOL
from pylj.model import Model
from pylj.potentials import Species
from pylj.scattering import check_q_max, default_q_max, shell_average, wavevectors


@dataclass(frozen=True, eq=False)
class PairData:
    """The result of :meth:`Configuration.pairs`.

    Attributes:
        distances: The minimum-image distance between each pair, in Angstrom.
        separations: The minimum-image separation of each pair, ``r_i - r_j``
            with ``i < j``, shape ``(M, 2)``, in Angstrom.
        energies: The energy of each pair, in kJ/mol.
        radial_forces: The radial force on each pair, in kJ/mol/Angstrom, positive
            where repulsive; ``None`` if it was not evaluated.
    """

    distances: NDArray[np.float64]
    separations: NDArray[np.float64]
    energies: NDArray[np.float64]
    radial_forces: NDArray[np.float64] | None

    @property
    def virial(self) -> float:
        """The sum over pairs of the radial force times the distance, in kJ/mol."""
        return float(np.sum(_radial_forces(self) * self.distances))


def _radial_forces(pairs: PairData) -> NDArray[np.float64]:
    """Returns the radial forces.

    Raises:
        ValueError: If the pair data was evaluated without forces.
    """
    if pairs.radial_forces is None:
        raise ValueError("The pair forces were not evaluated; call pairs(..., forces=True)")
    return pairs.radial_forces


_AtomPairs = tuple[NDArray[np.intp], NDArray[np.intp], list[tuple[NDArray[np.bool_], int, int]]]


class Configuration:
    """A single configuration of atoms and the simulation cell.

    The constructor copies the arrays it is given.

    Args:
        positions: The position of each atom, shape ``(N, 2)``, in Angstrom.
        species: The distinct species, indexed by ``species_index``.
        species_index: The index in ``species`` of each atom's species,
            shape ``(N,)``.
        box: The side length of the square periodic box, in Angstrom.

    Raises:
        ValueError: If the array shapes disagree, ``species`` is empty, an
            entry of ``species_index`` does not correspond to one of the
            species, or the box is not positive and finite.
    """

    __slots__ = ("positions", "species", "species_index", "box", "_atom_pairs_cache")

    def __init__(
        self,
        positions: ArrayLike,
        species: Sequence[Species],
        species_index: ArrayLike,
        box: float,
    ) -> None:
        positions = np.array(positions, dtype=float)
        species = tuple(species)
        species_index = np.array(species_index)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"positions must have shape (N, 2), not {positions.shape}")
        n = positions.shape[0]
        integer = np.issubdtype(species_index.dtype, np.integer)
        if species_index.shape != (n,) or not integer:
            raise ValueError(
                f"species_index must be an integer array of shape ({n},), one entry per "
                f"atom, not {species_index.dtype} of shape {species_index.shape}"
            )
        if not species:
            raise ValueError("species must name at least one Species")
        if n and not (0 <= species_index.min() and species_index.max() < len(species)):
            raise ValueError(f"species_index must index the {len(species)} species")
        if not (np.isfinite(box) and box > 0):
            raise ValueError(f"box must be positive and finite, not {box}")
        self.positions: NDArray[np.float64] = positions
        self.species: tuple[Species, ...] = species
        self.species_index: NDArray[np.int64] = species_index
        self.box: float = box
        self._atom_pairs_cache: tuple[NDArray[np.int64], _AtomPairs] | None = None

    @property
    def number_of_atoms(self) -> int:
        """The number of atoms."""
        return self.positions.shape[0]

    @property
    def masses(self) -> NDArray[np.float64]:
        """Atomic masses, in atomic mass units."""
        masses = np.array([one.mass for one in self.species], dtype=float)
        return masses[self.species_index]

    def copy(self) -> Self:
        """Returns an independent copy, with its own copy of every array."""
        return type(self)(self.positions, self.species, self.species_index, self.box)

    def without(self, index: int) -> Self:
        """Returns a copy of this Configuration with one atom removed.

        Args:
            index: The index of the atom to remove.

        Returns:
            The configuration without that atom.
        """
        return type(self)(
            np.delete(self.positions, index, axis=0),
            self.species,
            np.delete(self.species_index, index),
            self.box,
        )

    def _atom_pairs(self) -> _AtomPairs:
        """Returns the indices ``i < j`` of every pair of atoms and the pairs
        grouped by the species they join, rebuilding them when
        ``species_index`` has changed."""
        cache = self._atom_pairs_cache
        if cache is None or not np.array_equal(cache[0], self.species_index):
            i, j = np.triu_indices(self.number_of_atoms, 1)
            pairs = (i, j, list(pairwise.species_pairs(self.species_index)))
            cache = (self.species_index.copy(), pairs)
            self._atom_pairs_cache = cache
        return cache[1]

    def pairs(self, model: Model, cut_off: float, *, forces: bool = False) -> PairData:
        """Evaluates the energy and optionally forces for each pair of atoms.

        A pair further apart than the cut-off has zero energy and force, and
        a pair closer than its potential's ``min_separation`` has infinite
        energy and raises if forces are requested.

        Args:
            model: The model.
            cut_off: The cut-off, in Angstrom.
            forces: Whether to evaluate the radial forces.

        Returns:
            The pair data.

        Raises:
            ValueError: If forces are requested and a pair is closer than
                its potential's ``min_separation``.
        """
        i, j, by_species = self._atom_pairs()
        separations = pairwise.minimum_image(self.positions[i] - self.positions[j], self.box)
        distances = np.linalg.norm(separations, axis=1)
        energies = np.zeros(distances.size)
        radial_forces = np.zeros(distances.size) if forces else None
        for mask, type_1, type_2 in by_species:
            potential = model.potential(self.species[type_1], self.species[type_2])
            energies[mask] = potential.energies(distances[mask])
            forbidden = mask & (distances < potential.min_separation)
            if forbidden.any():
                if radial_forces is not None:
                    raise ValueError(
                        f"A pair of {self.species[type_1].name or 'atoms'} and "
                        f"{self.species[type_2].name or 'atoms'} is "
                        f"{distances[forbidden].min():.2f} Angstrom apart, closer than the "
                        f"{potential.min_separation:.2f} Angstrom below which "
                        f"{type(potential).__name__} is unphysical: the simulation has collapsed."
                    )
                energies[forbidden] = np.inf
            if radial_forces is not None:
                radial_forces[mask] = potential.forces(distances[mask])
        beyond = distances > cut_off
        energies[beyond] = 0.0
        if radial_forces is not None:
            radial_forces[beyond] = 0.0
        return PairData(distances, separations, energies, radial_forces)

    def potential_energy(self, model: Model, cut_off: float) -> float:
        """Computes the total pair energy, in kJ/mol."""
        return float(self.pairs(model, cut_off).energies.sum())

    def forces(self, model: Model, cut_off: float) -> NDArray[np.float64]:
        """Computes the net force on each atom, shape ``(N, 2)``, in kJ/mol/Angstrom."""
        pairs = self.pairs(model, cut_off, forces=True)
        radial = _radial_forces(pairs)
        i, j, _ = self._atom_pairs()
        # Each pair's radial force acts along its separation, pushing
        # atom i one way and atom j the other.
        pair_forces = (radial / pairs.distances)[:, None] * pairs.separations
        forces = np.zeros((self.number_of_atoms, 2))
        np.add.at(forces, i, pair_forces)
        np.add.at(forces, j, -pair_forces)
        return forces

    def virial(self, model: Model, cut_off: float) -> float:
        """Computes the sum over pairs of the radial force times the distance,
        in kJ/mol."""
        return self.pairs(model, cut_off, forces=True).virial

    def insertion_energy(
        self,
        position: ArrayLike,
        species_index: int,
        model: Model,
        cut_off: float,
    ) -> float:
        """Computes the interaction energy of one added atom with the atoms already
        in the configuration.

        Args:
            position: The ``(x, y)`` position of the added atom, in Angstrom.
            species_index: The index in ``species`` of the added atom's species.
            model: The model.
            cut_off: The cut-off, in Angstrom.

        Returns:
            The sum of its pair energies, in kJ/mol; zero for an empty
            configuration.
        """
        separations = pairwise.minimum_image(
            np.asarray(position, dtype=float) - self.positions, self.box
        )
        distances = np.linalg.norm(separations, axis=1)
        energies = np.zeros(distances.size)
        for other in np.unique(self.species_index):
            mask = self.species_index == other
            potential = model.potential(self.species[species_index], self.species[int(other)])
            energies[mask] = potential.energies(distances[mask])
            energies[mask & (distances < potential.min_separation)] = np.inf
        energies[distances > cut_off] = 0.0
        return float(energies.sum())

    def rdf(
        self, bins: int = 100, r_max: float | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Computes the radial distribution function g(r) of this configuration.

        The pair distances are binned from zero to ``r_max`` and divided by
        the counts an ideal gas of the same density would give.

        Args:
            bins: The number of bins.
            r_max: The largest distance binned, in Angstrom. By default half
                the box, beyond which the ideal-gas normalisation no longer
                holds.

        Returns:
            The bin centres, in Angstrom, and g(r) in each bin.
            Returns zero everywhere for a configuration containing one atom.
        """
        if r_max is None:
            r_max = self.box / 2
        edges = np.linspace(0, r_max, bins + 1)
        dr = edges[1] - edges[0]
        r = edges[:-1] + dr / 2
        n = self.number_of_atoms
        pairs = n * (n - 1) / 2
        if pairs == 0:
            return r, np.zeros(bins)
        distances, _ = pairwise.dist(self.positions, self.box)
        counts, _ = np.histogram(distances, bins=edges)
        ideal = pairs * 2 * np.pi * r * dr / self.box**2
        return r, counts / ideal

    def structure_factor(
        self, q_max: float | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Computes the structure factor S(q) of this configuration.

        S(q) is evaluated at the wavevectors commensurate with the box,
        ``2 pi (h, k) / L`` for integers ``h`` and ``k`` not both zero.
        Wavevectors of equal magnitude are averaged together, so the result
        holds one value per magnitude. Every atom counts alike, whatever its
        species.

        Args:
            q_max: The largest wavevector magnitude, in 1/Angstrom; the default
                comes from :func:`~pylj.scattering.default_q_max`.

        Returns:
            The wavevector magnitudes, in 1/Angstrom, and S(q) at each magnitude.

        Raises:
            ValueError: If ``q_max`` is below ``2 pi / L``.
        """
        if q_max is None:
            q_max = default_q_max(self.number_of_atoms, self.box)
        check_q_max(q_max, self.box)
        q, index, shell = wavevectors(self.box, q_max)
        return q, shell_average(self.positions, self.box, index, shell)


class MDConfiguration(Configuration):
    """A configuration with atom velocities and box crossings.

    It takes the arguments of :class:`Configuration`, followed by these two.

    Args:
        velocities: The velocity of each atom, shape ``(N, 2)``, in
            Angstrom per picosecond.
        images: The number of times each atom has crossed the box along x
            and along y, shape ``(N, 2)``, negative for crossings in the
            negative direction. By default zero.

    Raises:
        ValueError: If ``velocities`` or ``images`` is not the shape of
            ``positions``, or ``images`` is not an integer array.
    """

    __slots__ = ("velocities", "images")

    def __init__(
        self,
        positions: ArrayLike,
        species: Sequence[Species],
        species_index: ArrayLike,
        box: float,
        velocities: ArrayLike,
        images: ArrayLike | None = None,
    ) -> None:
        super().__init__(positions, species, species_index, box)
        velocities = np.array(velocities, dtype=float)
        if images is None:
            images = np.zeros(self.positions.shape, dtype=np.int64)
        images = np.array(images)
        for name, array in (("velocities", velocities), ("images", images)):
            if array.shape != self.positions.shape:
                raise ValueError(
                    f"{name} must have the shape of positions, {self.positions.shape}, "
                    f"not {array.shape}"
                )
        if not np.issubdtype(images.dtype, np.integer):
            raise ValueError(f"images must be an integer array, not {images.dtype}")
        self.velocities: NDArray[np.float64] = velocities
        self.images: NDArray[np.int64] = images

    @property
    def unwrapped(self) -> NDArray[np.float64]:
        """The position of each atom without periodic wrapping, shape ``(N, 2)``,
        in Angstrom: ``positions + images * box``."""
        return self.positions + self.images * self.box

    def copy(self) -> Self:
        """Returns an independent copy, with its own copy of every array."""
        return type(self)(
            self.positions,
            self.species,
            self.species_index,
            self.box,
            self.velocities,
            self.images,
        )

    def without(self, index: int) -> Self:
        """Returns a copy of this MDConfiguration with one atom removed.

        Args:
            index: The index of the atom to remove.

        Returns:
            The configuration without that atom.
        """
        return type(self)(
            np.delete(self.positions, index, axis=0),
            self.species,
            np.delete(self.species_index, index),
            self.box,
            np.delete(self.velocities, index, axis=0),
            np.delete(self.images, index, axis=0),
        )

    def kinetic_energy(self) -> float:
        """Computes the total kinetic energy, in kJ/mol."""
        return float(0.5 * np.sum(self.masses * np.sum(self.velocities**2, axis=1)) / KJ_PER_MOL)

    def temperature(self) -> float:
        """Computes the instantaneous temperature, in kelvin.

        The kinetic energy divided by ``(N - 1) k_B``, the centre of mass
        being at rest.

        Raises:
            ValueError: If there are fewer than two atoms.
        """
        if self.number_of_atoms < 2:
            raise ValueError("The temperature is undefined for a single atom.")
        return self.kinetic_energy() / ((self.number_of_atoms - 1) * BOLTZMANN)
