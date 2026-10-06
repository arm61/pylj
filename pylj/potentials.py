"""Atom species and pair potentials."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq


def check_positive_finite(name: str, value: float) -> None:
    """Raises ``ValueError`` unless ``value`` is positive and finite.

    Args:
        name: The name of the parameter, for the error message.
        value: The value to check.
    """
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, not {value}")


def check_non_negative_finite(name: str, value: float) -> None:
    """Raises ``ValueError`` unless ``value`` is zero or positive, and finite.

    Args:
        name: The name of the parameter, for the error message.
        value: The value to check.
    """
    if not (np.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be non-negative and finite, not {value}")


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

    Both ``energies`` and ``forces`` take an array of separations ``dr``,
    in Angstrom, and return an array of the same shape: energies in kJ/mol
    and forces in kJ/mol per Angstrom.

    Attributes:
        min_separation: The separation, in Angstrom, below which the potential
            is unphysical. Zero, the default, means the potential is physical
            at every separation. A configuration treats any pair closer
            than this as forbidden: its energy is infinite, and asking for
            its force raises an error.
    """

    min_separation: float = 0.0

    @abstractmethod
    def energies(self, dr: ArrayLike) -> NDArray[np.float64]:
        """Evaluates the pair energy at each separation in ``dr``."""

    @abstractmethod
    def forces(self, dr: ArrayLike) -> NDArray[np.float64]:
        """Evaluates the signed radial force at each separation in ``dr``.

        Minus the derivative of the energy with respect to the separation,
        so positive where the interaction is repulsive.
        """


@dataclass(frozen=True, kw_only=True)
class LennardJones(PairPotential):
    r"""The 12-6 Lennard-Jones pair potential.

    .. math::
        E = 4 \epsilon \left[ (\sigma / r)^{12} - (\sigma / r)^{6} \right]

    Args:
        epsilon: The well depth, in kJ/mol.
        sigma: The separation at which the pair energy is zero, in Angstrom.

    Raises:
        ValueError: If ``epsilon`` or ``sigma`` is not positive and finite.
    """

    epsilon: float
    sigma: float

    def __post_init__(self) -> None:
        check_positive_finite("epsilon", self.epsilon)
        check_positive_finite("sigma", self.sigma)

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


@dataclass(frozen=True, kw_only=True)
class Buckingham(PairPotential):
    r"""The Buckingham pair potential.

    .. math::
        E = A e^{-B r} - C / r^{6}

    The formula falls to minus infinity at short range, beyond a barrier.
    ``energies`` and ``forces`` return it at every separation, and
    ``min_separation`` is the separation at the top of that barrier.

    Args:
        a: The A parameter, an energy scale, in kJ/mol.
        b: The B parameter, an inverse length, in reciprocal Angstrom.
        c: The C parameter, the dispersion coefficient, in kJ/mol Angstrom^6.

    Attributes:
        min_separation: The separation of the top of the short-range
            barrier, in Angstrom; zero when there is no barrier.

    Raises:
        ValueError: If ``a`` or ``b`` is not positive and finite, if ``c`` is
            negative or not finite, or if the barrier lies beyond 100
            Angstrom.
    """

    a: float
    b: float
    c: float
    min_separation: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        check_positive_finite("a", self.a)
        check_positive_finite("b", self.b)
        check_non_negative_finite("c", self.c)
        object.__setattr__(self, "min_separation", self._find_barrier())

    def _form(self, dr: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.a * np.exp(-self.b * dr) - self.c / dr**6

    def _slope(self, dr: float) -> float:
        return float(-self.a * self.b * np.exp(-self.b * dr) + 6 * self.c / dr**7)

    def _find_barrier(self) -> float:
        """Locates the top of the short-range barrier."""
        dr = np.geomspace(1e-3, 1e2, 4000)
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
        return float(brentq(self._slope, dr[barrier - 1], dr[barrier + 1]))

    def energies(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        with np.errstate(divide="ignore"):
            return self._form(dr)

    def forces(self, dr: ArrayLike) -> NDArray[np.float64]:
        dr = np.asarray(dr, dtype=float)
        with np.errstate(divide="ignore"):
            return self.a * self.b * np.exp(-self.b * dr) - 6 * self.c / dr**7


@dataclass(frozen=True, kw_only=True)
class SquareWell(PairPotential):
    r"""The square-well pair potential.

    The energy is ``max_val`` below ``sigma``, minus ``epsilon`` between
    ``sigma`` and ``lambda_`` times ``sigma``, and zero beyond. It has no
    finite force, so it is for Monte Carlo, which uses energies only.

    Args:
        epsilon: The well depth, in kJ/mol.
        sigma: The hard-core diameter, in Angstrom.
        lambda_: The outer edge of the well, in units of sigma.
        max_val: The value used in place of the infinite hard core, in kJ/mol.

    Raises:
        ValueError: If ``epsilon`` or ``sigma`` is not positive and finite,
            ``lambda_`` is not greater than one, or ``max_val`` is not
            positive.
    """

    epsilon: float
    sigma: float
    lambda_: float
    max_val: float = np.inf

    def __post_init__(self) -> None:
        check_positive_finite("epsilon", self.epsilon)
        check_positive_finite("sigma", self.sigma)
        if not (np.isfinite(self.lambda_) and self.lambda_ > 1):
            raise ValueError(
                "lambda_ must be greater than 1: the well lies outside the hard core"
            )
        if not self.max_val > 0:
            raise ValueError(
                "max_val must be positive: a hard core that lowers the energy would "
                "draw atoms into it"
            )

    def __repr__(self) -> str:
        call = (
            f"SquareWell(epsilon={self.epsilon!r}, sigma={self.sigma!r}, "
            f"lambda_={self.lambda_!r}"
        )
        if np.isfinite(self.max_val):
            call += f", max_val={self.max_val!r}"
        return call + ")"

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
