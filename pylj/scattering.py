"""The structure factor of a configuration, evaluated at the wavevectors
commensurate with its box."""

import numpy as np
from numpy.typing import NDArray


def default_q_max(number_of_atoms: int, box: float) -> float:
    """Return a wavevector magnitude that covers the first few peaks, in 1/m.

    A square box of side ``L`` holding ``N`` atoms leaves a mean spacing of
    ``L / sqrt(N)`` between them, and the wavevector matching that spacing is
    ``2 pi sqrt(N) / L``. The magnitude returned is six times that, so it
    grows with the density of the configuration.

    Args:
        number_of_atoms: The number of atoms.
        box: The side length of the square box, in metres.

    Returns:
        The wavevector magnitude, in 1/m.
    """
    return 6 * 2 * np.pi * np.sqrt(number_of_atoms) / box


def wavevectors(
    box: float, q_max: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Return the wavevectors commensurate with a box, grouped by magnitude.

    A square box of side ``L`` that repeats in both directions has the
    wavevectors ``2 pi (h, k) / L``, for integer ``h`` and ``k``. The pair
    where both are zero is left out. The wavevectors that share a magnitude
    make up a shell.

    Args:
        box: The side length of the square box, in metres.
        q_max: The largest magnitude to return, in 1/m.

    Returns:
        The distinct magnitudes in increasing order, in 1/m; the wavevectors
        themselves, of shape ``(M, 2)``, in 1/m; and, for each wavevector,
        the position in that list of magnitudes of the shell it belongs to.
    """
    unit = 2 * np.pi / box
    limit = int(np.floor(q_max / unit))
    index = np.arange(-limit, limit + 1)
    h, k = np.meshgrid(index, index, indexing="ij")
    square = (h**2 + k**2).ravel()
    inside = (square > 0) & (square <= limit**2)
    square = square[inside]
    order = np.argsort(square, kind="stable")
    square = square[order]
    pair = np.stack([h.ravel()[inside][order], k.ravel()[inside][order]], axis=1)
    magnitude, shell = np.unique(square, return_inverse=True)
    return unit * np.sqrt(magnitude), unit * pair.astype(float), shell.astype(np.int64)


def shell_average(
    position: NDArray[np.float64],
    wavevector: NDArray[np.float64],
    shell: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return the structure factor of a configuration, one value per shell.

    Each wavevector ``q`` has an amplitude ``sum_j exp(i q . r_j)``, summed
    over the atom positions ``r_j``. The structure factor at that wavevector
    is the square of the modulus of the amplitude, divided by the number of
    atoms. The wavevectors that share a magnitude are averaged together, so
    the result holds one value per shell.

    Args:
        position: The atom positions, shape ``(N, 2)``, in metres.
        wavevector: The wavevectors, shape ``(M, 2)``, in 1/m.
        shell: The index of the shell each wavevector belongs to.

    Returns:
        The structure factor at each shell magnitude.
    """
    amplitude = np.exp(1j * (position @ wavevector.T)).sum(axis=0)
    intensity = np.abs(amplitude) ** 2 / len(position)
    return np.bincount(shell, weights=intensity) / np.bincount(shell)
