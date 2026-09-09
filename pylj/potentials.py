"""The atom species and the pair potentials that act between them:
Lennard-Jones, Buckingham and the square well."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq


def check_positive_finite(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is positive and finite.

    Args:
        name: The name of the parameter, for the error message.
        value: The value to check.
    """
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, not {value}")


@dataclass(frozen=True)
class Species:
    """An atom species.

    Args:
        mass: The atom mass, in atomic mass units.
        name: A label for the species, such as "argon".

    Raises:
        ValueError: If the mass is not positive and finite.
    """

    mass: float
    name: str = ""

    def __post_init__(self) -> None:
        check_positive_finite("mass", self.mass)


class PairPotential(ABC):
    """The interface every pair potential implements.

    A pair potential is a central potential: the energy of a pair of atoms,
    and the force that follows from it, depend only on how far apart the two
    atoms are, and not on the direction from one to the other. Both
    ``energies`` and ``forces`` take an array of separations ``dr``, in metres,
    and return an array of the same shape.

    Attributes:
        min_separation: The separation, in metres, below which the potential
            is unphysical. Zero, the default, means the potential is physical
            at every separation. A configuration treats any pair closer
            than this as forbidden: its energy is infinite, and asking for
            its force raises an error.
    """

    min_separation: float = 0.0

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

    Raises:
        ValueError: If ``epsilon`` or ``sigma`` is not positive and finite.
    """

    def __init__(self, *, epsilon: float, sigma: float):
        check_positive_finite("epsilon", epsilon)
        check_positive_finite("sigma", sigma)
        self.epsilon = epsilon
        self.sigma = sigma

    def __repr__(self) -> str:
        return f"LennardJones(epsilon={self.epsilon!r}, sigma={self.sigma!r})"

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
    barrier as the atoms approach and then falls to minus infinity as
    the separation goes to zero. That collapse is a defect of the formula,
    not real physics. ``energies`` and ``forces`` return the formula at
    every separation, and ``min_separation`` is set to the separation at the
    top of the barrier, so a simulation never lets two atoms pass it.

    Args:
        a: The A parameter, an energy scale, in joules.
        b: The B parameter, an inverse length, in reciprocal metres.
        c: The C parameter, the dispersion coefficient, in joule metre^6.

    Attributes:
        min_separation: The separation of the top of the short-range
            barrier, in metres; zero when there is no barrier.

    Raises:
        ValueError: If ``a`` or ``b`` is not positive and finite, if ``c`` is
            negative or not finite, or if the barrier lies beyond 100
            Angstrom, so that the formula collapses at every separation a
            simulation could reach.
    """

    def __init__(self, *, a: float, b: float, c: float):
        check_positive_finite("a", a)
        check_positive_finite("b", b)
        if not (np.isfinite(c) and c >= 0):
            raise ValueError(f"c must be non-negative and finite, not {c}")
        self.a = a
        self.b = b
        self.c = c
        self.min_separation = self._find_barrier()

    def __repr__(self) -> str:
        return f"Buckingham(a={self.a!r}, b={self.b!r}, c={self.c!r})"

    def _form(self, dr: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.a * np.exp(-self.b * dr) - self.c / dr**6

    def _slope(self, dr: float) -> float:
        return float(-self.a * self.b * np.exp(-self.b * dr) + 6 * self.c / dr**7)

    def _find_barrier(self) -> float:
        """Locate the top of the short-range barrier. The slope of the energy is zero
        there, between the fall to minus infinity at short range and the well
        beyond."""
        dr = np.geomspace(1e-13, 1e-8, 4000)
        with np.errstate(over="ignore"):
            barrier = int(np.argmax(self._form(dr)))
        if barrier == 0:
            return 0.0
        if barrier == dr.size - 1:
            raise ValueError(
                "The Buckingham energy is still rising at 100 Angstrom: the repulsion "
                "A exp(-B r) is too weak to hold atoms apart at any separation a "
                "simulation could reach. Increase a or b, or reduce c."
            )
        return float(brentq(self._slope, dr[barrier - 1], dr[barrier + 1], xtol=1e-16))

    def energies(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        with np.errstate(divide="ignore"):
            return self._form(dr)

    def forces(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        with np.errstate(divide="ignore"):
            return self.a * self.b * np.exp(-self.b * dr) - 6 * self.c / dr**7


class SquareWell(PairPotential):
    r"""The square-well pair potential: a hard core surrounded by a well of
    constant depth.

    The energy takes three values. When the separation is smaller than
    ``sigma`` the atoms overlap and the energy is ``max_val``, infinite
    by default. Between ``sigma`` and ``lambda_`` times ``sigma`` the
    atoms sit in the well and the energy is minus ``epsilon``. Beyond
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

    Raises:
        ValueError: If ``epsilon`` or ``sigma`` is not positive and finite,
            ``lambda_`` is not greater than one, or ``max_val`` is not
            positive.
    """

    def __init__(self, *, epsilon: float, sigma: float, lambda_: float, max_val: float = np.inf):
        check_positive_finite("epsilon", epsilon)
        check_positive_finite("sigma", sigma)
        if not (np.isfinite(lambda_) and lambda_ > 1):
            raise ValueError(
                f"lambda_ must be greater than 1, not {lambda_}: the well lies outside "
                "the hard core"
            )
        if not max_val > 0:
            raise ValueError(
                f"max_val must be positive, not {max_val}: a hard core that lowers the "
                "energy would draw atoms into it"
            )
        self.epsilon = epsilon
        self.sigma = sigma
        self.lambda_ = lambda_
        self.max_val = max_val

    def __repr__(self) -> str:
        return (
            f"SquareWell(epsilon={self.epsilon!r}, sigma={self.sigma!r}, "
            f"lambda_={self.lambda_!r}, max_val={self.max_val!r})"
        )

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
