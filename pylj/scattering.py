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


def check_q_max(q_max: float, box: float) -> None:
    """Check that a wavevector magnitude reaches the box.

    The smallest wavevector a box of side ``L`` has is ``2 pi / L``, so a
    ``q_max`` below that leaves nothing to evaluate.

    Args:
        q_max: The largest wavevector magnitude, in 1/m.
        box: The side length of the square box, in metres.

    Raises:
        ValueError: If ``q_max`` is below ``2 pi / L``.
    """
    smallest = 2 * np.pi / box
    if q_max < smallest:
        raise ValueError(
            f"q_max of {q_max:g} 1/m is below {smallest:g} 1/m, the smallest wavevector a box "
            f"of {box * 1e10:.1f} Angstrom has. q_max is in 1/m, and one inverse Angstrom is "
            "1e10 1/m, so a value meant in inverse Angstrom lands far below the box."
        )


def wavevectors(
    box: float, q_max: float
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Return the wavevectors commensurate with a box, grouped by magnitude.

    A square box of side ``L`` that repeats in both directions has the
    wavevectors ``2 pi (h, k) / L``, for integer ``h`` and ``k``. The pair
    where both are zero is left out. The wavevectors that share a magnitude
    make up a shell.

    Args:
        box: The side length of the square box, in metres.
        q_max: The largest magnitude to return, in 1/m.

    Returns:
        The distinct magnitudes in increasing order, in 1/m; the pair of
        integers ``(h, k)`` of each wavevector, of shape ``(M, 2)``; and,
        for each wavevector, the position in that list of magnitudes of the
        shell it belongs to.
    """
    unit = 2 * np.pi / box
    # No single component can exceed q_max, so that bounds the search, and
    # the magnitude of the pair is what decides whether it is inside. A whole
    # number of units is the common request, and dividing by unit can land a
    # hair under it, so the ratio is nudged up before it is floored and
    # before it is squared. The price is that a shell within a part in 1e12
    # above q_max is kept.
    ratio = q_max / unit * (1 + 1e-12)
    limit = int(np.floor(ratio))
    index = np.arange(-limit, limit + 1)
    h, k = np.meshgrid(index, index, indexing="ij")
    square = (h**2 + k**2).ravel()
    inside = (square > 0) & (square <= ratio**2)
    square = square[inside]
    order = np.argsort(square, kind="stable")
    square = square[order]
    pair = np.stack([h.ravel()[inside][order], k.ravel()[inside][order]], axis=1)
    magnitude, shell = np.unique(square, return_inverse=True)
    return unit * np.sqrt(magnitude), pair.astype(np.int64), shell.astype(np.int64)


def _phase_rows(
    coordinate: NDArray[np.float64], unit: float, limit: int
) -> NDArray[np.complex128]:
    """Return ``exp(i unit h x)`` for every atom and every ``h``.

    The rows run from ``-limit`` to ``limit``. Negative ``h`` gives the
    complex conjugate of positive ``h``, so only the non-negative rows are
    evaluated.

    Args:
        coordinate: One coordinate of each atom, in metres.
        unit: The smallest wavevector of the box, in 1/m.
        limit: The largest ``h`` to return.

    Returns:
        The phase factors, shape ``(2 * limit + 1, N)``.
    """
    nonneg = np.exp(1j * unit * np.outer(np.arange(limit + 1), coordinate))
    return np.vstack([np.conj(nonneg[:0:-1]), nonneg])


def shell_average(
    position: NDArray[np.float64],
    box: float,
    index: NDArray[np.int64],
    shell: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return the structure factor of a configuration, one value per shell.

    The wavevector of a pair of integers ``(h, k)`` is ``2 pi (h, k) / L``,
    and its amplitude is ``sum_j exp(i q . r_j)``, summed over the atom
    positions ``r_j``. The structure factor at that wavevector is the square
    of the modulus of the amplitude, divided by the number of atoms. Every
    atom counts alike, whatever its species. The wavevectors that share a
    magnitude are averaged together, so the result holds one value per shell.

    A phase factor separates into one term per axis, ``exp(i q . r) =
    exp(i q_x x) exp(i q_y y)``, so the amplitudes of every ``(h, k)`` are
    the product of a matrix of phase factors along x with one along y.

    Args:
        position: The atom positions, shape ``(N, 2)``, in metres.
        box: The side length of the square box, in metres.
        index: The pair of integers of each wavevector, shape ``(M, 2)``.
        shell: The index of the shell each wavevector belongs to.

    Returns:
        The structure factor at each shell magnitude.
    """
    limit = int(np.abs(index).max())
    unit = 2 * np.pi / box
    along_x = _phase_rows(position[:, 0], unit, limit)
    along_y = _phase_rows(position[:, 1], unit, limit)
    grid = along_x @ along_y.T
    amplitude = grid[index[:, 0] + limit, index[:, 1] + limit]
    intensity = np.abs(amplitude) ** 2 / len(position)
    return np.bincount(shell, weights=intensity) / np.bincount(shell)
