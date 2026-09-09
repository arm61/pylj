"""The Debye sum, which turns a set of pair distances into a scattering
profile, and the binning that lets one sum stand for many pairs."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import j0


def max_separation(box: float) -> float:
    """Return the largest separation two atoms can have in a periodic box.

    Each component of a minimum-image separation is at most half the box, so
    the largest separation is the half-diagonal. The value returned is one
    floating-point step above that, because a distance computed from the two
    components can land a step above a half-diagonal computed directly, and
    a distance above the largest bin edge would be dropped from a histogram.

    Args:
        box: The side length of the square box, in metres.

    Returns:
        The largest separation, in metres.
    """
    return float(np.nextafter(box / np.sqrt(2), np.inf))


def bin_centres(bins: int, r_max: float) -> NDArray[np.float64]:
    """Return the centre of each bin, in metres.

    The bins divide zero to ``r_max`` evenly. They depend on nothing but
    these two numbers, so one set of centres serves every frame of a run.

    Args:
        bins: The number of bins.
        r_max: The largest distance binned, in metres.

    Returns:
        The centre of each bin, in metres.
    """
    edges = np.linspace(0, r_max, bins + 1)
    return edges[:-1] + (edges[1] - edges[0]) / 2


def bin_counts(distance: NDArray[np.float64], bins: int, r_max: float) -> NDArray[np.float64]:
    """Return how many distances fall in each bin of :func:`bin_centres`.

    Args:
        distance: The pair distances, in metres.
        bins: The number of bins.
        r_max: The largest distance binned, in metres. Any distance beyond
            it is dropped, so a sum over every pair needs an ``r_max`` of at
            least :func:`max_separation`.

    Returns:
        How many distances fall in each bin.
    """
    counts, _ = np.histogram(distance, bins=np.linspace(0, r_max, bins + 1))
    return counts.astype(float)


def debye_sum(
    distance: ArrayLike,
    q: ArrayLike,
    number_of_atoms: int,
    weight: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return the two-dimensional Debye sum over a set of pair distances.

    Each atom contributes one for scattering on its own, giving
    ``number_of_atoms`` in total, and each pair at distance ``r`` adds
    ``2 J0(q r)``.

    Args:
        distance: The pair distances to sum over, in metres.
        q: The magnitudes of the scattering vector, in 1/m.
        number_of_atoms: The number of atoms.
        weight: How many pairs each distance stands for; by default one
            each. Binning gives one distance per bin and the count in it,
            and an average over several frames divides those counts by the
            number of frames.

    Returns:
        I(q) at each value of ``q``, in units of one atom's scattering.
    """
    distance = np.asarray(distance, dtype=float)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    weight = None if weight is None else np.asarray(weight, dtype=float)
    intensity = np.empty_like(q)
    # A block of q values at a time; the outer product with the pair
    # distances is what takes the memory.
    block = 16
    for start in range(0, q.size, block):
        bessel = j0(np.outer(q[start : start + block], distance))
        if weight is not None:
            bessel = bessel * weight
        intensity[start : start + block] = bessel.sum(axis=1)
    return number_of_atoms + 2 * intensity
