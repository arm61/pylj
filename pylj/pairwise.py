"""Calculations over every pair of atoms at once: grouping the atom pairs by
the species they join, applying the minimum image convention, and getting the
pressure from the virial."""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


def species_pairs(
    species_index: NDArray[np.int64],
) -> Iterator[tuple[NDArray[np.bool_], int, int]]:
    """Group the atom pairs by the two species they join.

    Each pair of species present is yielded once, because species 0 with
    species 1 is the same pair as species 1 with species 0. Each comes with a
    mask, which picks out the entries of the pair arrays returned by
    :func:`dist` that join those two species.

    Args:
        species_index: The species index of each atom.

    Yields:
        The mask, the lower species index and the upper species index.
    """
    i, j = np.triu_indices(species_index.size, 1)
    lower = np.minimum(species_index[i], species_index[j])
    upper = np.maximum(species_index[i], species_index[j])
    for type_1, type_2 in sorted(set(zip(lower.tolist(), upper.tolist(), strict=True))):
        yield (lower == type_1) & (upper == type_2), type_1, type_2


def minimum_image(separation: NDArray[np.float64], box: float) -> NDArray[np.float64]:
    """Return separations wrapped to the nearest periodic image.

    Args:
        separation: Separation vectors, shape ``(..., 2)``, in metres.
        box: The side length of the square periodic box, in metres.

    Returns:
        The separations with each component brought into ``[-box / 2,
        box / 2]``, so each is the shortest of the periodic copies.
    """
    return separation - box * np.round(separation / box)


def dist(
    position: NDArray[np.float64], box: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the minimum-image distance and separation of every pair.

    Args:
        position: The position of each atom, shape ``(N, 2)``, in
            metres.
        box: The side length of the square periodic box, in metres.

    Returns:
        The distance between each pair, shape ``(M,)``, and the separation
        ``r_i - r_j`` of each pair, shape ``(M, 2)``, both in metres. Each of
        the ``M = N (N - 1) / 2`` pairs appears once, ordered by the lower
        atom index and then the higher.
    """
    i, j = np.triu_indices(position.shape[0], 1)
    separation = minimum_image(position[i] - position[j], box)
    return np.linalg.norm(separation, axis=1), separation


def calculate_pressure(virial: float, box: float, kinetic_energy: float) -> float:
    r"""Return the instantaneous pressure of the cell in two dimensions.

    .. math::
        p = \frac{1}{2 L^2} \left( 2 K + \sum_{i} \sum_{j > i} f_{ij} r_{ij} \right)

    The kinetic term is the momentum the atoms carry across a line in
    the cell. The centre of mass is held at rest, so over a run at
    temperature ``T`` the kinetic energy averages ``(N - 1) k_B T`` and this
    term averages ``(N - 1) k_B T / L^2``, one atom short of the
    ideal-gas pressure ``N k_B T / L^2``.

    Args:
        virial: The sum over pairs of the radial force times the distance,
            in joules.
        box: The side length of the square periodic box, in metres.
        kinetic_energy: The total kinetic energy, in joules.

    Returns:
        The pressure, in newtons per metre (a two-dimensional pressure).
    """
    return (2 * kinetic_energy + virial) / (2 * box * box)
