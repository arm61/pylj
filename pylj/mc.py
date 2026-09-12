"""Metropolis Monte Carlo simulation."""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

from pylj.configuration import Configuration
from pylj.constants import BOLTZMANN
from pylj.model import Model
from pylj.placement import place
from pylj.potentials import check_positive_finite
from pylj.simulation import (
    Samples,
    Simulation,
    _check_initial_energy,
    _check_potentials_at_the_cut_off,
    _empty,
)


@dataclass
class MCSamples(Samples):
    """The record a Monte Carlo simulation appends to at each sample.

    Attributes:
        potential_energy: The total pair energy, in joules.
    """

    potential_energy: NDArray[np.float64] = field(default_factory=_empty)


@dataclass(frozen=True, eq=False)
class Proposal:
    """A proposed configuration for a Monte Carlo move, and the corresponding
    energy change.

    Attributes:
        positions: The proposed position of every atom, shape ``(N, 2)``,
            in metres.
        energy_change: The change in energy under the proposed move, in
            joules.
        source: The configuration the proposal was made from.
    """

    positions: NDArray[np.float64]
    energy_change: float
    source: Configuration


def accept(
    energy_change: float,
    temperature: float,
    *,
    rng: np.random.Generator | None = None,
) -> bool:
    """Applies the Metropolis criterion to an energy change.

    Args:
        energy_change: The change in energy under the proposed move, in
            joules.
        temperature: The temperature the acceptance is judged at, in kelvin.
        rng: The generator to draw from; pass the simulation's ``rng`` for
            a reproducible run. By default an unseeded generator is used.

    Returns:
        True if the proposed configuration should be accepted.
    """
    if energy_change <= 0:
        return True
    if rng is None:
        rng = np.random.default_rng()
    return bool(rng.random() < np.exp(-energy_change / (BOLTZMANN * temperature)))


class MCSimulation(Simulation):
    """A Monte Carlo simulation.

    Args:
        configuration: The starting configuration.
        model: The model.
        temperature: The temperature of the simulation, in kelvin.
        cut_off: The cut-off, in metres; see :class:`Simulation`.
        seed: Seed for the random number generator.

    Attributes:
        temperature: The temperature, in kelvin.
        energy: The total pair energy of the current configuration, in
            joules.
        accepted: The number of moves accepted so far.
        samples: The :class:`MCSamples` record that ``sample`` appends to.

    Raises:
        ValueError: If the temperature is not positive and finite, a pair
            potential has not died away at the cut-off, or the configuration
            stores more than :data:`simulation.INITIAL_ENERGY_LIMIT` k_B T of
            potential energy per atom.
    """

    samples: MCSamples

    def __init__(
        self,
        configuration: Configuration,
        model: Model,
        temperature: float,
        *,
        cut_off: float | None = None,
        seed: int | None = None,
    ) -> None:
        check_positive_finite("temperature", temperature)
        super().__init__(configuration, model, cut_off=cut_off, seed=seed)
        self.temperature = temperature
        _check_potentials_at_the_cut_off(self.model, self.cut_off, temperature, configuration.box)
        self.energy = configuration.potential_energy(self.model, self.cut_off)
        _check_initial_energy(self.energy, configuration.number_of_atoms, temperature)
        self.accepted = 0
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
                square grid, ``'triangular'`` on a triangular lattice filling
                the box, which constrains the number of atoms, and
                ``'metropolis'`` inserts them at random positions for a
                disordered start.
            placement_temperature: The temperature of the Metropolis
                acceptance used by ``'metropolis'``, in kelvin; by default
                the run temperature.
            max_strain: If ``init_conf`` is ``'triangular'``, how far the
                fitted lattice may sit from
                ``sqrt(3) / 2``, as a fraction of that ratio.
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
        configuration, cut_off_metres = place(
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
        simulation = cls(configuration, model, temperature, cut_off=cut_off_metres)
        simulation.rng = rng
        return simulation

    def propose(self) -> Proposal:
        """Proposes moving one atom to a random positions.

        Returns:
            The proposal.
        """
        configuration = self.configuration
        atom = int(self.rng.integers(configuration.number_of_atoms))
        trial = self.rng.uniform(0, configuration.box, size=2)
        current = configuration.positions[atom]
        species_index = int(configuration.species_index[atom])
        others = configuration.without(atom)
        energy_change = others.insertion_energy(
            trial, species_index, self.model, self.cut_off
        ) - others.insertion_energy(current, species_index, self.model, self.cut_off)
        positions = configuration.positions.copy()
        positions[atom] = trial
        return Proposal(positions, energy_change, configuration)

    def apply(self, proposal: Proposal) -> None:
        """Applies a proposal, replacing the configuration with its positions.

        Args:
            proposal: The proposal.

        Raises:
            ValueError: If the proposal was made from a configuration other
                than the current one.
        """
        if proposal.source is not self.configuration:
            raise ValueError(
                "This proposal was made from a configuration that is no longer the current "
                "one, so its energy change no longer applies. Propose again from the current "
                "configuration."
            )
        self.configuration = self.configuration.replace(positions=proposal.positions)
        self.energy += proposal.energy_change

    def step(self) -> None:
        """Proposes a move and accept or reject it by the Metropolis criterion."""
        proposal = self.propose()
        if accept(proposal.energy_change, self.temperature, rng=self.rng):
            self.apply(proposal)
            self.accepted += 1
        self.steps += 1

    def sample(self) -> None:
        """Records the configuration in the trajectory and measures it.

        The energy is recomputed from the configuration, so the recorded
        value is exact.
        """
        self.trajectory.append(self.configuration)
        self.energy = self.configuration.potential_energy(self.model, self.cut_off)
        self.samples.add(step=self.steps, potential_energy=self.energy)

    def restart(self) -> Self:
        """Returns a new simulation continuing from the current configuration.

        See :meth:`Simulation.restart`.
        """
        new = super().restart()
        new.accepted = 0
        return new
