"""Structure factor calculations."""

import numpy as np
from numpy.typing import NDArray

#: Most wavevectors a structure factor is evaluated at. Building them and
#: the amplitudes takes a few tens of bytes each, so this bounds the working
#: memory to under a hundred megabytes. The default range asks for about a
#: hundred thousand of them for a thousand atoms.
MOST_WAVEVECTORS = 1_000_000


def default_q_max(number_of_atoms: int, box: float) -> float:
    """Chooses the default largest wavevector magnitude, in 1/m.

    ``2 pi sqrt(N) / L`` matches the mean spacing between ``N`` atoms in a
    box of side ``L``; the default is six times it.

    Args:
        number_of_atoms: The number of atoms.
        box: The side length of the square box, in metres.

    Returns:
        The wavevector magnitude, in 1/m.
    """
    return 6 * 2 * np.pi * np.sqrt(number_of_atoms) / box


def check_q_max(q_max: float, box: float) -> None:
    """Checks a wavevector magnitude against the box and the wavevector limit.

    Args:
        q_max: The largest wavevector magnitude, in 1/m.
        box: The side length of the square box, in metres.

    Raises:
        ValueError: If ``q_max`` is below ``2 pi / L``, or needs more than
            :data:`MOST_WAVEVECTORS`.
    """
    smallest = 2 * np.pi / box
    if q_max < smallest:
        raise ValueError(
            f"q_max of {q_max:g} 1/m is below {smallest:g} 1/m, the smallest wavevector a box "
            f"of {box * 1e10:.1f} Angstrom has. q_max is in 1/m, and one inverse Angstrom is "
            "1e10 1/m, so a value meant in inverse Angstrom lands far below the box."
        )
    across = 2 * int(np.floor(q_max / smallest)) + 1
    if across**2 > MOST_WAVEVECTORS:
        largest = smallest * (np.sqrt(MOST_WAVEVECTORS) - 1) / 2
        raise ValueError(
            f"q_max of {q_max:g} 1/m needs {across**2} wavevectors, above the limit of "
            f"{MOST_WAVEVECTORS}. A box of {box * 1e10:.1f} Angstrom reaches {largest:g} 1/m "
            "within it. q_max is in 1/m, and one inverse Angstrom is 1e10 1/m."
        )


def wavevectors(
    box: float, q_max: float
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Enumerates the wavevectors commensurate with a box, grouped by magnitude.

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
    pair = np.stack([h.ravel()[inside], k.ravel()[inside]], axis=1)
    magnitude, shell = np.unique(square[inside], return_inverse=True)
    return unit * np.sqrt(magnitude), pair.astype(np.int64), shell.astype(np.int64)


def _phase_rows(
    coordinate: NDArray[np.float64], unit: float, limit: int
) -> NDArray[np.complex128]:
    """Computes ``exp(i unit h x)`` for every atom and every ``h``.

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
    positions: NDArray[np.float64],
    box: float,
    index: NDArray[np.int64],
    shell: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Computes the structure factor of a configuration.

    The amplitude at a wavevector is ``sum_j exp(i q . r_j)`` over the atom
    positions, and the structure factor is the square of its modulus divided
    by the number of atoms. Every atom counts alike, whatever its species,
    and the wavevectors of a shell are averaged together.

    Args:
        positions: The atom positions, shape ``(N, 2)``, in metres.
        box: The side length of the square box, in metres.
        index: The pair of integers of each wavevector, shape ``(M, 2)``.
        shell: The index of the shell each wavevector belongs to.

    Returns:
        The structure factor at each shell magnitude.
    """
    # exp(i q . r) = exp(i q_x x) exp(i q_y y), so the amplitudes of every
    # (h, k) are one matrix product of the per-axis phase factors.
    limit = int(np.abs(index).max())
    unit = 2 * np.pi / box
    along_x = _phase_rows(positions[:, 0], unit, limit)
    along_y = _phase_rows(positions[:, 1], unit, limit)
    grid = along_x @ along_y.T
    amplitude = grid[index[:, 0] + limit, index[:, 1] + limit]
    intensity = np.abs(amplitude) ** 2 / len(positions)
    return np.bincount(shell, weights=intensity) / np.bincount(shell)
