"""Shared simulation base class and sample records."""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Self

import numpy as np
from numpy.typing import NDArray

from pylj.configuration import Configuration
from pylj.model import Model
from pylj.potentials import check_positive_finite
from pylj.trajectory import Trajectory

#: The cut-off used when none is given, in Angstrom, or half the box if
#: that is smaller.
DEFAULT_CUT_OFF = 15


def _resolve_cut_off(box: float, cut_off: float | None) -> float:
    """Resolves the cut-off.

    A ``cut_off`` of ``None`` becomes :data:`DEFAULT_CUT_OFF`, or half the
    box if that is smaller.

    Raises:
        ValueError: If a given cut-off is not positive and finite, or is
            larger than half the box.
    """
    if cut_off is None:
        return min(DEFAULT_CUT_OFF, box / 2)
    check_positive_finite("cut_off", cut_off)
    if cut_off > box / 2:
        raise ValueError(
            f"The cut-off of {cut_off:.1f} Angstrom exceeds half the box of {box:.1f} "
            "Angstrom; the minimum image convention needs a cut-off of at most half the box."
        )
    return cut_off


def _check_species(configuration: Configuration, model: Model) -> None:
    """Raises ``ValueError`` if the configuration has a species the model lacks."""
    for one in configuration.species:
        if one not in model.species:
            raise ValueError(f"The configuration has species {one}, which is not in the model")


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

    The constructor takes a configuration that is already built; the
    ``initialise`` method of either subclass builds one from a model and a
    number of atoms instead.

    Args:
        configuration: The starting configuration. The simulation runs on
            its own copy.
        model: The model.
        cut_off: The separation, in Angstrom, beyond which a pair's energy
            and force are zero. By default :data:`DEFAULT_CUT_OFF` Angstrom
            or half the box, whichever is smaller; it may not exceed half the
            box.
        seed: Seed for the random number generator; the same seed
            reproduces the run, and without one the run differs each time.

    Attributes:
        configuration: The current configuration. Its arrays can be
            changed, or another configuration assigned; the simulation
            recomputes what it keeps from it before it next uses it.
        rng: The random number generator for this simulation.
        steps: The number of steps taken.
        samples: The record ``sample`` appends to.
        trajectory: The configurations sampled so far, a
            :class:`~pylj.trajectory.Trajectory`.

    Raises:
        ValueError: If a species in the configuration is not in the model,
            or the cut-off exceeds half the box.
    """

    __slots__ = (
        "configuration",
        "model",
        "cut_off",
        "rng",
        "steps",
        "samples",
        "trajectory",
        "_kept_for",
    )

    def __init__(
        self,
        configuration: Configuration,
        model: Model,
        *,
        cut_off: float | None = None,
        seed: int | None = None,
    ) -> None:
        _check_species(configuration, model)
        self.configuration = configuration.copy()
        self.model = model
        self.cut_off = _resolve_cut_off(configuration.box, cut_off)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.samples = Samples()
        self.trajectory = Trajectory()
        self._recompute_from_configuration()
        self._keep()

    def _keep(self) -> None:
        """Records the state that the kept forces or energy came from."""
        c = self.configuration
        self._kept_for = (
            c.positions.copy(),
            c.species_index.copy(),
            c.species,
            c.box,
            self.model,
            self.cut_off,
        )

    def _bring_up_to_date(self) -> None:
        """Recomputes the kept forces or energy if the state they came from has
        changed.

        Raises:
            ValueError: If the number of atoms or the box has changed, or the
                configuration has a species the model lacks.
        """
        c = self.configuration
        positions, species_index, species, box, model, cut_off = self._kept_for
        if (
            np.array_equal(c.positions, positions)
            and np.array_equal(c.species_index, species_index)
            and c.species == species
            and c.box == box
            and self.model is model
            and self.cut_off == cut_off
        ):
            return
        if c.number_of_atoms != len(positions):
            raise ValueError(
                f"The number of atoms has changed from {len(positions)} to "
                f"{c.number_of_atoms}. To run with the {c.number_of_atoms}-atom "
                f"configuration, build a new {type(self).__name__}."
            )
        if c.box != box:
            raise ValueError(
                f"The box has changed from {box:g} to {c.box:g} Angstrom. To run with the "
                f"configuration in the {c.box:g} Angstrom box, build a new "
                f"{type(self).__name__}."
            )
        _check_species(c, self.model)
        self._recompute_from_configuration()
        self._keep()

    @abstractmethod
    def _recompute_from_configuration(self) -> None:
        """Recomputes what the simulation keeps from its configuration."""

    @abstractmethod
    def step(self) -> None:
        """Advances the simulation by one step."""

    @abstractmethod
    def sample(self) -> None:
        """Records the current step in ``samples``."""

    def restart(self) -> Self:
        """Returns a new simulation continuing from the current configuration.

        The new simulation has its own copy of the configuration and of the
        state of the random number generator, the same model and numerical
        choices, ``steps`` at zero, no samples and an empty trajectory. This
        simulation is unchanged. Use it to start a production run after
        equilibration::

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
        new.configuration = self.configuration.copy()
        new.rng = copy.deepcopy(self.rng)
        new.steps = 0
        new.samples = type(self.samples)()
        # The new trajectory records times only if the old one did.
        new.trajectory = Trajectory(times=None if self.trajectory.times is None else [])
        return new
