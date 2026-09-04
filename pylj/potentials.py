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
