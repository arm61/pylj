from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Species:
    """A particle species.

    Args:
        mass: The particle mass, in atomic mass units.
        name: A label for the species, such as "argon".

    Raises:
        ValueError: If the mass is not positive and finite.
    """

    mass: float
    name: str = ""

    def __post_init__(self) -> None:
        if not (np.isfinite(self.mass) and self.mass > 0):
            raise ValueError(f"mass must be positive and finite, not {self.mass}")


class PairPotential(ABC):
    """The interface every pair potential implements.

    A pair potential is central: the pair energy, and the radial force
    derived from it, depend only on the separation magnitude. Both
    ``energies`` and ``forces`` take an array of separations ``dr``, in
    metres, and return an array of the same shape.
    """

    @abstractmethod
    def energies(self, dr: ArrayLike) -> NDArray[np.float64]:
        """Return the pair energy for each separation in ``dr``."""

    @abstractmethod
    def forces(self, dr: ArrayLike) -> NDArray[np.float64]:
        """Return the signed radial force for each separation in ``dr``.

        The value is minus the derivative of the energy with respect to the
        separation, so it is positive where the interaction is repulsive and
        negative where it is attractive.
        """


class LennardJones(PairPotential):
    r"""The 12-6 Lennard-Jones pair potential.

    .. math::
        E = 4 \epsilon \left[ (\sigma / r)^{12} - (\sigma / r)^{6} \right]

    Args:
        epsilon: The well depth, in joules.
        sigma: The separation at which the pair energy is zero, in metres.
    """

    def __init__(self, *, epsilon: float, sigma: float):
        self.epsilon = epsilon
        self.sigma = sigma

    def energies(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        return 4 * self.epsilon * (self.sigma**12 / dr**12 - self.sigma**6 / dr**6)

    def forces(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        return 4 * self.epsilon * (12 * self.sigma**12 / dr**13 - 6 * self.sigma**6 / dr**7)


class Buckingham(PairPotential):
    r"""The Buckingham pair potential.

    .. math::
        E = A e^{-B r} - C / r^{6}

    Args:
        a: The A parameter, an energy scale, in joules.
        b: The B parameter, an inverse length, in reciprocal metres.
        c: The C parameter, the dispersion coefficient, in joule metre^6.
    """

    def __init__(self, *, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def energies(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        return self.a * np.exp(-self.b * dr) - self.c / dr**6

    def forces(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        return self.a * self.b * np.exp(-self.b * dr) - 6 * self.c / dr**7


class SquareWell(PairPotential):
    r"""The square-well pair potential.

    The energy is ``max_val`` inside the hard core of diameter sigma, minus
    epsilon in the well out to lambda times sigma, and zero beyond. The force
    is impulsive at the two walls, so the potential drives Monte Carlo only.

    Args:
        epsilon: The well depth, in joules.
        sigma: The hard-core diameter, in metres.
        lambda_: The outer edge of the well, in units of sigma.
        max_val: The value used in place of the infinite hard core.
    """

    def __init__(self, *, epsilon: float, sigma: float, lambda_: float, max_val: float = np.inf):
        self.epsilon = epsilon
        self.sigma = sigma
        self.lambda_ = lambda_
        self.max_val = max_val

    def energies(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        return np.where(
            dr < self.sigma,
            self.max_val,
            np.where(dr < self.lambda_ * self.sigma, -self.epsilon, 0.0),
        )

    def forces(self, dr: ArrayLike) -> NDArray[np.float64]:
        raise ValueError(
            "The square-well force is impulsive and cannot drive molecular "
            "dynamics; use a Monte Carlo simulation."
        )
