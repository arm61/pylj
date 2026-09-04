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
