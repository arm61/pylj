"""Vectorised calculations over atom pairs."""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


def species_pairs(
    species_index: NDArray[np.int64],
) -> Iterator[tuple[NDArray[np.bool_], int, int]]:
    """Groups the atom pairs by the two species they join.

    Each pair of species present is yielded once, with a mask picking out
    the entries of the pair arrays returned by :func:`dist` that join those
    two species.

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
    """Wraps separations to the nearest periodic image.

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
    """Computes the minimum-image distance and separation of every pair.

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
    r"""Computes the instantaneous pressure of the cell in two dimensions.

    .. math::
        p = \frac{1}{2 L^2} \left( 2 K + \sum_{i} \sum_{j > i} f_{ij} r_{ij} \right)

    Args:
        virial: The sum over pairs of the radial force times the distance,
            in joules.
        box: The side length of the square periodic box, in metres.
        kinetic_energy: The total kinetic energy, in joules.

    Returns:
        The pressure, in newtons per metre.
    """
    return (2 * kinetic_energy + virial) / (2 * box * box)
