from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq


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

    A pair potential is a central potential: the energy of a pair of particles, and
    the force that follows from it, depend only on how far apart the two
    particles are, and not on the direction from one to the other. Both
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
        negative where the interaction is attractive.
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
        with np.errstate(divide="ignore"):
            x = (self.sigma / dr) ** 6
        return 4 * self.epsilon * x * (x - 1)

    def forces(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        with np.errstate(divide="ignore"):
            x = (self.sigma / dr) ** 6
            return 24 * self.epsilon * x * (2 * x - 1) / dr


class Buckingham(PairPotential):
    r"""The Buckingham pair potential.

    .. math::
        E = A e^{-B r} - C / r^{6}

    At short range the attractive term, minus C over r to the sixth, grows
    faster than the exponential repulsion. The energy therefore rises to a
    barrier as the particles approach and then falls to minus infinity as
    the separation goes to zero. That collapse is a defect of the formula,
    not real physics, so this potential treats the barrier as an
    impenetrable wall: at any separation smaller than ``turnover``, the
    separation at the top of the barrier, the energy and the force are
    infinite.

    Args:
        a: The A parameter, an energy scale, in joules.
        b: The B parameter, an inverse length, in reciprocal metres.
        c: The C parameter, the dispersion coefficient, in joule metre^6.

    Attributes:
        turnover: The separation of the top of the short-range barrier, in
            metres; zero when there is no barrier.
    """

    def __init__(self, *, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c
        self.turnover = self._find_turnover()

    def _form(self, dr: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.a * np.exp(-self.b * dr) - self.c / dr**6

    def _slope(self, dr: float) -> float:
        return float(-self.a * self.b * np.exp(-self.b * dr) + 6 * self.c / dr**7)

    def _find_turnover(self) -> float:
        """Locate the top of the short-range barrier. The slope of the energy is zero
        there, between the fall to minus infinity at short range and the well
        beyond."""
        dr = np.geomspace(1e-13, 1e-8, 4000)
        with np.errstate(over="ignore"):
            barrier = int(np.argmax(self._form(dr)))
        if barrier == 0:
            return 0.0
        return float(brentq(self._slope, dr[barrier - 1], dr[barrier + 1], xtol=1e-16))

    def energies(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(dr < self.turnover, np.inf, self._form(dr))

    def forces(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            force = self.a * self.b * np.exp(-self.b * dr) - 6 * self.c / dr**7
        return np.where(dr < self.turnover, np.inf, force)


class SquareWell(PairPotential):
    r"""The square-well pair potential: a hard core surrounded by a well of
    constant depth.

    The energy takes three values. When the separation is smaller than
    ``sigma`` the particles overlap and the energy is ``max_val``, infinite
    by default. Between ``sigma`` and ``lambda_`` times ``sigma`` the
    particles sit in the well and the energy is minus ``epsilon``. Beyond
    the well the energy is zero. Because the energy changes only in steps,
    the force is zero everywhere except at the two walls, where it is
    infinite. A potential without a finite force cannot drive molecular
    dynamics, so the square well is for Monte Carlo, which uses energies
    only.

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
            "The square-well energy changes in steps, so its force is zero everywhere except "
            "at the two walls, where it is infinite. Molecular dynamics cannot integrate that; "
            "use a Monte Carlo simulation."
        )
