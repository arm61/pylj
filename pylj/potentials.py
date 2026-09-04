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
    """

    mass: float
    name: str = ""


class PairPotential(ABC):
    """The interface every pair potential implements.

    A pair potential is central: the pair energy depends only on the
    separation magnitude, so both quantities are functions of an array of
    separations.
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


class lennard_jones(PairPotential):
    r"""The 12-6 Lennard-Jones pair potential.

    .. math::
        E = 4 \epsilon \left[ (\sigma / r)^{12} - (\sigma / r)^{6} \right]

    Args:
        epsilon: The well depth, in Joules.
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


class buckingham(PairPotential):
    r"""The Buckingham pair potential.

    .. math::
        E = A e^{-B r} - C / r^{6}

    Args:
        a: The A parameter, an energy scale, in Joules.
        b: The B parameter, an inverse length, in reciprocal metres.
        c: The C parameter, the dispersion coefficient, in Joule metre^6.
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
