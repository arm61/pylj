"""Metropolis Monte Carlo simulation."""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

from pylj.configuration import Configuration
from pylj.constants import BOLTZMANN
from pylj.model import Model
from pylj.placement import place
from pylj.potentials import check_non_negative_finite, check_positive_finite
from pylj.simulation import Samples, Simulation, _empty


@dataclass
class MCSamples(Samples):
    """The record a Monte Carlo simulation appends to at each sample.

    Attributes:
        potential_energy: The total pair energy, in kJ/mol.
    """

    potential_energy: NDArray[np.float64] = field(default_factory=_empty)


@dataclass(frozen=True, eq=False)
class Proposal:
    """A proposed configuration for a Monte Carlo move, and the corresponding
    energy change.

    Attributes:
        positions: The proposed position of every atom, shape ``(N, 2)``,
            in Angstrom.
        energy_change: The change in energy under the proposed move, in
            kJ/mol.
        source: A copy of the positions the proposal was made from, shape
            ``(N, 2)``, in Angstrom.
    """

    positions: NDArray[np.float64]
    energy_change: float
    source: NDArray[np.float64]


def accept(
    energy_change: float,
    temperature: float,
    *,
    rng: np.random.Generator | None = None,
) -> bool:
    """Applies the Metropolis criterion to an energy change.

    Args:
        energy_change: The change in energy under the proposed move, in
            kJ/mol.
        temperature: The temperature the acceptance is judged at, in kelvin.
        rng: The generator to draw from; pass the simulation's ``rng`` for
            a reproducible run. By default an unseeded generator is used.

    Returns:
        True if the proposed configuration should be accepted.

    Raises:
        ValueError: If the energy change is not a number.
    """
    if np.isnan(energy_change):
        raise ValueError(
            "The energy change is not a number, so the Metropolis criterion is "
            "undefined. The atom's current and trial positions are both inside a "
            "hard core, or a position is not finite."
        )
    if energy_change <= 0:
        return True
    if temperature == 0:
        return False
    if rng is None:
        rng = np.random.default_rng()
    return bool(rng.random() < np.exp(-energy_change / (BOLTZMANN * temperature)))


class MCSimulation(Simulation):
    """A Monte Carlo simulation.

    Args:
        configuration: The starting configuration.
        model: The model.
        temperature: The temperature of the simulation, in kelvin.
        cut_off: The cut-off, in Angstrom; see :class:`Simulation`.
        max_displacement: The largest distance an atom is moved along each
            axis in one step, in Angstrom.
        seed: Seed for the random number generator.

    Attributes:
        temperature: The temperature, in kelvin.
        max_displacement: The largest distance an atom is moved along each
            axis in one step, in Angstrom.
        accepted: The number of moves accepted so far.
        samples: The :class:`MCSamples` record that ``sample`` appends to.

    Raises:
        ValueError: If the temperature is negative or not finite, or the
            maximum displacement is not positive and finite.
    """

    samples: MCSamples

    __slots__ = ("temperature", "max_displacement", "accepted", "_energy")

    def __init__(
        self,
        configuration: Configuration,
        model: Model,
        temperature: float,
        *,
        cut_off: float | None = None,
        max_displacement: float = 0.5,
        seed: int | None = None,
    ) -> None:
        check_non_negative_finite("temperature", temperature)
        check_positive_finite("max_displacement", max_displacement)
        self.temperature = temperature
        self.max_displacement = max_displacement
        self.accepted = 0
        super().__init__(configuration, model, cut_off=cut_off, seed=seed)
        self.samples = MCSamples()

    @classmethod
    def initialise(
        cls,
        model: Model,
        *,
        number_of_atoms: int,
        temperature: float,
        box: float,
        init_conf: str = "square",
        placement_temperature: float | None = None,
        max_strain: float = 0.05,
        max_displacement: float = 0.5,
        cut_off: float | None = None,
        seed: int | None = None,
    ) -> Self:
        """Builds a simulation from a model.

        Places the atoms in the box.

        Args:
            model: The model.
            number_of_atoms: The number of atoms.
            temperature: The temperature of the simulation, in kelvin.
            box: The side length of the box, in Angstrom, at least
                :data:`~pylj.placement.SMALLEST_BOX`.
            init_conf: How the atoms are placed. ``'square'`` puts them on a
                grid of columns and rows filling the box, ``'triangular'`` on
                a triangular lattice filling the box, which constrains the
                number of atoms, and ``'metropolis'`` inserts them at random
                positions for a disordered start.
            placement_temperature: The temperature of the Metropolis
                acceptance used by ``'metropolis'``, in kelvin; by default
                the run temperature.
            max_strain: How far a triangular lattice may sit from
                ``sqrt(3) / 2``, as a fraction of that ratio; used when
                ``init_conf`` is ``'triangular'``.
            max_displacement: The largest distance an atom is moved along
                each axis in one step, in Angstrom.
            cut_off: The cut-off, in Angstrom; by default
                :data:`~pylj.simulation.DEFAULT_CUT_OFF` Angstrom or half the
                box, whichever is smaller.
            seed: Seed for the random number generator used to place the
                configuration and make the moves; without one the run differs
                each time.

        Returns:
            The simulation, with its energy evaluated.

        Raises:
            ValueError: For anything :func:`placement.place` or the
                constructor rejects.
        """
        rng = np.random.default_rng(seed)
        configuration = place(
            number_of_atoms,
            temperature,
            box,
            model=model,
            init_conf=init_conf,
            placement_temperature=placement_temperature,
            max_strain=max_strain,
            cut_off=cut_off,
            rng=rng,
        )
        simulation = cls(
            configuration,
            model,
            temperature,
            cut_off=cut_off,
            max_displacement=max_displacement,
        )
        simulation.rng = rng
        return simulation

    def _recompute_from_configuration(self) -> None:
        self._energy = self.configuration.potential_energy(self.model, self.cut_off)

    @property
    def energy(self) -> float:
        """The total pair energy of the current configuration, in kJ/mol."""
        self._bring_up_to_date()
        return self._energy

    def propose(self) -> Proposal:
        """Proposes moving one atom, chosen at random, by a random distance of up
        to ``max_displacement`` along each axis.

        Returns:
            The proposal.
        """
        configuration = self.configuration
        atom = int(self.rng.integers(configuration.number_of_atoms))
        current = configuration.positions[atom]
        displacement = self.rng.uniform(-self.max_displacement, self.max_displacement, size=2)
        trial = (current + displacement) % configuration.box
        species_index = int(configuration.species_index[atom])
        others = configuration.without(atom)
        energy_change = others.insertion_energy(
            trial, species_index, self.model, self.cut_off
        ) - others.insertion_energy(current, species_index, self.model, self.cut_off)
        source = configuration.positions.copy()
        positions = source.copy()
        positions[atom] = trial
        return Proposal(positions, energy_change, source)

    def apply(self, proposal: Proposal) -> None:
        """Applies a proposal, moving the atom to its proposed position.

        Args:
            proposal: The proposal.

        Raises:
            ValueError: If the proposal was made from positions other than
                the current ones.
        """
        if not np.array_equal(proposal.source, self.configuration.positions):
            raise ValueError(
                "This proposal was made from a configuration that is no longer the current "
                "one, so its energy change no longer applies. Propose again from the current "
                "configuration."
            )
        energy = self.energy
        self.configuration.positions = proposal.positions
        self._energy = energy + proposal.energy_change
        if not np.isfinite(self._energy):
            # A hard-core overlap makes the energy infinite, and adding an
            # energy change to infinity cannot tell when the overlap clears.
            self._energy = self.configuration.potential_energy(self.model, self.cut_off)
        self._keep()

    def step(self) -> None:
        """Proposes a move and accepts or rejects it by the Metropolis criterion."""
        proposal = self.propose()
        if accept(proposal.energy_change, self.temperature, rng=self.rng):
            self.apply(proposal)
            self.accepted += 1
        self.steps += 1

    def sample(self) -> None:
        """Records a copy of the configuration in the trajectory and measures it.

        The energy is recomputed from the configuration, so the recorded
        value is exact.
        """
        self._bring_up_to_date()
        self.trajectory.append(self.configuration.copy())
        self._energy = self.configuration.potential_energy(self.model, self.cut_off)
        self._keep()
        self.samples.add(step=self.steps, potential_energy=self._energy)

    def restart(self) -> Self:
        """Returns a new simulation continuing from the current configuration.

        See :meth:`Simulation.restart`.
        """
        new = super().restart()
        new.accepted = 0
        return new
