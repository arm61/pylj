"""Shared simulation base class and sample records."""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Self

import numpy as np
from numpy.typing import NDArray

from pylj.configuration import Configuration
from pylj.constants import BOLTZMANN
from pylj.model import Model
from pylj.potentials import check_positive_finite
from pylj.trajectory import Trajectory

#: The cut-off used when none is given, in Angstrom, or half the box if
#: that is smaller.
DEFAULT_CUT_OFF = 15

#: Largest potential energy per atom, in units of k_B T, accepted for an
#: initial configuration.
INITIAL_ENERGY_LIMIT = 10.0


def _check_potentials_at_the_cut_off(
    model: Model, cut_off: float, temperature: float, box: float
) -> None:
    """Checks that every pair potential has died away at the cut-off.

    The check is that each pair energy at the cut-off is finite and within
    ``k_B T`` of zero.

    Args:
        model: The model.
        cut_off: The cut-off, in metres.
        temperature: The temperature, in kelvin.
        box: The side length of the box, in metres.

    Raises:
        ValueError: If any pair potential's energy at the cut-off is not
            finite or larger in magnitude than ``k_B T``.
    """
    from_box = cut_off >= box / 2
    where = f"the cut-off of {cut_off * 1e10:.1f} Angstrom"
    if from_box:
        where += " (half the box)"
    remedy = "Use a larger box" if from_box else "Use a larger box or cut-off"
    for (one, other), potential in model.pair_potentials.items():
        energy = float(potential.energies(np.array([cut_off]))[0])
        pair = (
            f"{type(potential).__name__} between {one.name or 'atoms'} and {other.name or 'atoms'}"
        )
        if not np.isfinite(energy):
            raise ValueError(
                f"{pair} is infinite at {where}: its hard core is wider than the cut-off. {remedy}."
            )
        if abs(energy) > BOLTZMANN * temperature:
            raise ValueError(
                f"{pair} is still {energy / (BOLTZMANN * temperature):+.3g} k_B T at {where}; "
                "the cut-off assumes the interaction has died away there. "
                f"{remedy}, or check the parameter units: metres and joules are expected."
            )


def _check_initial_energy(energy: float, number_of_atoms: int, temperature: float) -> None:
    """Refuses a starting configuration that stores far more potential energy
    than thermal energy.

    A configuration holding more than :data:`INITIAL_ENERGY_LIMIT` k_B T
    per atom has atoms too close together for its temperature.

    Args:
        energy: The total pair energy of the configuration, in joules.
        number_of_atoms: The number of atoms.
        temperature: The temperature of the run, in kelvin.

    Raises:
        ValueError: If the energy is not finite, or exceeds
            :data:`INITIAL_ENERGY_LIMIT` k_B T per atom.
    """
    remedy = (
        "Use fewer atoms or a larger box; init_conf='metropolis' places atoms by "
        "energy, and a lower placement_temperature there keeps them further apart."
    )
    if number_of_atoms == 0:
        return
    if not np.isfinite(energy):
        raise ValueError(
            "The initial pair energy is not finite: atoms sit inside a hard core, or a "
            f"position is not a number. {remedy}"
        )
    per_atom = energy / (number_of_atoms * BOLTZMANN * temperature)
    if per_atom > INITIAL_ENERGY_LIMIT:
        raise ValueError(
            f"The initial configuration stores {per_atom:.3g} k_B T of potential energy "
            f"per atom, above the limit of {INITIAL_ENERGY_LIMIT:g}: its atoms are "
            f"too close together for {temperature:g} K. {remedy}"
        )


def _resolve_cut_off(box: float, cut_off: float | None) -> float:
    """Resolves the cut-off to metres.

    A ``cut_off`` of ``None`` becomes :data:`DEFAULT_CUT_OFF` Angstrom, or
    half the box if that is smaller.

    Raises:
        ValueError: If a given cut-off is not positive and finite, or is
            larger than half the box.
    """
    if cut_off is None:
        return min(DEFAULT_CUT_OFF * 1e-10, box / 2)
    check_positive_finite("cut_off", cut_off)
    if cut_off > box / 2:
        raise ValueError(
            f"The cut-off of {cut_off * 1e10:.1f} Angstrom exceeds half the box of "
            f"{box * 1e10:.1f} Angstrom; the minimum image convention needs a cut-off of at "
            "most half the box."
        )
    return cut_off


def _empty() -> NDArray[np.float64]:
    return np.array([])


@dataclass
class Samples:
    """The record a simulation's ``sample`` appends to.

    Every array holds one entry per call of ``sample``, in order.

    Attributes:
        step: The step at which each sample was taken.
    """

    step: NDArray[np.int64] = field(default_factory=lambda: np.array([], dtype=np.int64))

    def add(self, **values: float) -> None:
        """Appends one value to every array.

        Args:
            **values: One value for each of this record's arrays, by name.

        Raises:
            ValueError: If the names do not match this record's arrays
                exactly.
        """
        expected = {f.name for f in fields(self)}
        if set(values) != expected:
            raise ValueError(
                f"{type(self).__name__}.add needs one value for each of "
                f"{sorted(expected)}, not {sorted(values)}"
            )
        for name, value in values.items():
            setattr(self, name, np.append(getattr(self, name), value))


class Simulation(ABC):
    """The base class molecular dynamics and Monte Carlo share.

    The constructor takes a configuration that is already built, in SI
    units; the ``initialise`` method of either subclass builds one from a
    model and a number of atoms instead.

    Args:
        configuration: The starting configuration.
        model: The model.
        cut_off: The separation, in metres, beyond which a pair's energy
            and force are zero. By default :data:`DEFAULT_CUT_OFF` Angstrom
            or half the box,
            whichever is smaller; it may not exceed half the box.
        seed: Seed for the random number generator; the same seed
            reproduces the run, and without one the run differs each time.

    Attributes:
        configuration: The current configuration.
        rng: The random number generator for this simulation.
        steps: The number of steps taken.
        samples: The record ``sample`` appends to.
        trajectory: The configurations sampled so far, a
            :class:`~pylj.trajectory.Trajectory`.

    Raises:
        ValueError: If a species in the configuration is not in the model,
            or the cut-off exceeds half the box.
    """

    def __init__(
        self,
        configuration: Configuration,
        model: Model,
        *,
        cut_off: float | None = None,
        seed: int | None = None,
    ) -> None:
        for one in configuration.species:
            if one not in model.species:
                raise ValueError(f"The configuration has species {one}, which is not in the model")
        self.configuration = configuration
        self.model = model
        self.cut_off = _resolve_cut_off(configuration.box, cut_off)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.samples = Samples()
        self.trajectory = Trajectory()

    @abstractmethod
    def step(self) -> None:
        """Advances the simulation by one step."""

    @abstractmethod
    def sample(self) -> None:
        """Records the current step in ``samples``."""

    def restart(self) -> Self:
        """Returns a new simulation continuing from the current configuration.

        The new simulation copies the model, the numerical choices and the
        state of the random number generator, and starts with ``steps`` at
        zero, no samples and an empty trajectory. This simulation is
        unchanged. Use it to start a production run after equilibration::

            for _ in range(1000):
                simulation.step()
            production = simulation.restart()
            for _ in range(5000):
                production.step()
                production.sample()

        Returns:
            The new simulation.
        """
        new = copy.copy(self)
        new.rng = copy.deepcopy(self.rng)
        new.steps = 0
        new.samples = type(self.samples)()
        new.trajectory = Trajectory()
        return new
