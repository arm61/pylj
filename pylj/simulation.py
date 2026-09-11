"""The checks a simulation makes on its model, the record it keeps of its
samples, and the base class the simulations share."""

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

#: Largest potential energy per atom, in units of k_B T, accepted for an
#: initial configuration.
INITIAL_ENERGY_LIMIT = 10.0


def _check_potentials_at_the_cut_off(
    model: Model, cut_off: float, temperature: float, box: float
) -> None:
    """Check that every pair potential has died away at the cut-off.

    Truncating the interaction at the cut-off assumes it is negligible
    there. The check is that each pair energy at the cut-off is finite and
    within ``k_B T`` of zero.

    Args:
        model: The species and the potential between each pair of them.
        cut_off: The cut-off, in metres.
        temperature: The temperature, in kelvin.
        box: The side length of the box, in metres. The cut-off can be no
            larger than half the box, so when it already is, the only remedy is
            a larger box.

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
    """Refuse a starting configuration that stores far more potential energy
    than thermal energy.

    Potential energy stored in an initial configuration is released as
    motion over the first steps and heats the run. A configuration holding
    more than :data:`INITIAL_ENERGY_LIMIT` k_B T per atom has atoms
    too close together for its temperature.

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
    """Return the cut-off in metres.

    A cut-off given by the caller is used as it stands. Without one, the
    cut-off is 15 Angstrom, or half the box if that is smaller.

    Raises:
        ValueError: If a given cut-off is not positive and finite, or is larger
            than half the box. A cut-off beyond half the box breaks the minimum
            image convention.
    """
    if cut_off is None:
        return min(15e-10, box / 2)
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

    Every array holds one entry per call of ``sample``, in order, so the
    arrays line up: ``step[i]`` is the step at which the other arrays'
    ``i``-th entries were measured. Subclasses add the quantities their
    simulation measures.

    Attributes:
        step: The step at which each sample was taken.
    """

    step: NDArray[np.int64] = field(default_factory=lambda: np.array([], dtype=np.int64))

    def add(self, **values: float) -> None:
        """Append one value to every array, keeping them aligned.

        Args:
            **values: One value for each of this record's arrays, by name.

        Raises:
            ValueError: If the names do not match this record's arrays
                exactly, since a missing or unknown name would leave the
                arrays out of step.
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
    """A simulation: a configuration, the model, the numerical choices, and
    the machinery that evolves the configuration and measures it.

    ``MDSimulation`` and ``MCSimulation`` add ``step`` and ``sample`` to this
    class. The constructor takes a configuration that has already been built,
    in SI units. To start from a model and a number of atoms instead, use the
    ``initialise`` method of one of those subclasses, which builds the
    configuration for you.

    Args:
        configuration: The starting configuration.
        model: The species and the potential between each pair of them, a
            :class:`~pylj.model.Model`.
        cut_off: The separation, in metres, beyond which a pair's
            interaction is taken as negligible. By default 15 Angstrom or
            half the box, whichever is smaller; it may not exceed half the
            box.
        seed: Seed for the random number generator. The same seed
            reproduces the same run; without one the run differs each time.

    Attributes:
        configuration: The current configuration.
        model: The model.
        cut_off: The cut-off, in metres.
        rng: The random number generator for this simulation.
        steps: The number of steps taken.
        samples: The record ``sample`` appends to; subclasses replace it
            with the record of what they measure.
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
        """Advance the simulation by one step."""

    @abstractmethod
    def sample(self) -> None:
        """Record the quantities of interest at the current step."""

    def restart(self) -> Self:
        """A new simulation that continues from the current configuration.

        The new simulation shares the model, the numerical choices
        and every other attribute with this one. Its random number generator
        starts from a copy of this one's state, so what this simulation draws
        next has no effect on the new one. The new simulation starts with
        ``steps`` at zero, an empty record of samples and an empty trajectory.
        A subclass that holds other per-run state extends this method to reset
        it. This simulation is not changed. Use it to start a production run
        after equilibration::

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
