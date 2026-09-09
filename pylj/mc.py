"""Monte Carlo: the simulation that samples configurations by the Metropolis
criterion, the criterion itself, and the proposed move the criterion accepts or
rejects."""

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
    """What a Monte Carlo simulation records at each sample.

    Attributes:
        potential_energy: The total pair energy, in joules.
    """

    potential_energy: NDArray[np.float64] = field(default_factory=_empty)


@dataclass(frozen=True, eq=False)
class Proposal:
    """A proposed configuration for a Monte Carlo move.

    The energy change is relative to the configuration the proposal was made
    from, so a proposal can only be applied while that configuration is
    still the current one.

    Attributes:
        position: The proposed position of every atom, shape ``(N, 2)``,
            in metres.
        energy_change: The energy of the proposed configuration minus that
            of ``source``, in joules.
        source: The configuration the proposal was made from.
    """

    position: NDArray[np.float64]
    energy_change: float
    source: Configuration


def accept(
    energy_change: float,
    temperature: float,
    *,
    random_number: float | None = None,
    rng: np.random.Generator | None = None,
) -> bool:
    """Apply the Metropolis criterion to an energy change.

    A change that does not raise the energy is always accepted, without
    drawing a random number. A change that raises it by ``energy_change`` is
    accepted with probability ``exp(-energy_change / (k_B temperature))``.

    Args:
        energy_change: The energy of the proposed configuration minus that
            of the current one, in joules.
        temperature: The temperature the acceptance is judged at, in kelvin.
        random_number: The uniform random number the acceptance probability
            is tested against. By default one is drawn from ``rng``.
        rng: The generator to draw from; pass the simulation's ``rng`` for
            a reproducible run. By default an unseeded generator is used.

    Returns:
        True if the proposed configuration should be accepted.
    """
    if energy_change <= 0:
        return True
    if random_number is None:
        if rng is None:
            rng = np.random.default_rng()
        random_number = rng.random()
    return bool(random_number < np.exp(-energy_change / (BOLTZMANN * temperature)))


class MCSimulation(Simulation):
    """A Monte Carlo simulation at a temperature.

    ``step`` proposes a move, accepts it by the Metropolis criterion or
    leaves the configuration as it is, and advances the step count;
    ``sample`` records the exact energy.

    Args:
        configuration: The starting configuration. An ``MDConfiguration``
            is accepted; its velocities are ignored.
        model: The species and the potential between each pair of them.
            Only the pair energies are evaluated, so a potential with no
            finite force, such as the square well, can be used.
        temperature: The temperature of the simulation, in kelvin.
        cut_off: The cut-off, in metres; see :class:`Simulation`.
        seed: Seed for the random number generator.

    Attributes:
        temperature: The temperature, in kelvin.
        energy: The total pair energy of the current configuration, in joules.
            It is computed when the simulation is built, updated by ``apply``
            each time a move is accepted, and recomputed exactly each time
            ``sample`` is called.
        accepted: The number of moves accepted so far.
        samples: The :class:`MCSamples` record that ``sample`` appends to.

    Raises:
        ValueError: If the temperature is not positive and finite, a pair
            potential has not died away at the cut-off, the configuration
            stores more than :data:`simulation.INITIAL_ENERGY_LIMIT` k_B T of
            potential energy per atom, or for anything
            :class:`Simulation` rejects.
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
        cut_off: float | None = None,
        seed: int | None = None,
    ) -> Self:
        """Build a simulation from a model: place the atoms and set the
        temperature.

        Args:
            model: The species, assigned to the atoms in turn, and the
                potential between each pair of them.
            number_of_atoms: The number of atoms.
            temperature: The temperature of the simulation, in kelvin.
            box: The side length of the box, in Angstrom, from 4 to 600.
            init_conf: ``'square'`` for a lattice or ``'metropolis'`` for
                sequential Metropolis insertion.
            placement_temperature: The temperature of the Metropolis
                acceptance used by ``'metropolis'``, in kelvin; by default
                the run temperature. Raising it tolerates closer contacts,
                lowering it rejects them more strictly and can exhaust the
                trial budget. Ignored by ``'square'``.
            cut_off: The cut-off, in Angstrom; by default 15 Angstrom or
                half the box, whichever is smaller.
            seed: Seed for the random number generator used to place the
                configuration and make the moves.

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
            cut_off=cut_off,
            rng=rng,
        )
        simulation = cls(configuration, model, temperature, cut_off=cut_off_metres)
        simulation.rng = rng
        return simulation

    def propose(self) -> Proposal:
        """Propose a move: one atom relocated at random.

        An atom is chosen at random and given a uniform trial position in
        the box. The energy change is that atom's interaction energy at
        the trial position minus that at its current position, with every
        other atom. The configuration is not changed.

        Returns:
            The proposed configuration and its energy change.
        """
        configuration = self.configuration
        atom = int(self.rng.integers(configuration.number_of_atoms))
        trial = self.rng.uniform(0, configuration.box, size=2)
        current = configuration.position[atom]
        species_index = int(configuration.species_index[atom])
        others = configuration.without(atom)
        energy_change = others.insertion_energy(
            trial, species_index, self.model, self.cut_off
        ) - others.insertion_energy(current, species_index, self.model, self.cut_off)
        position = configuration.position.copy()
        position[atom] = trial
        return Proposal(position, energy_change, configuration)

    def apply(self, proposal: Proposal) -> None:
        """Make a proposed configuration the current one.

        The configuration takes the proposal's positions, and the proposal's
        energy change is added to ``energy``.

        Args:
            proposal: The proposal to apply, from :meth:`propose`.

        Raises:
            ValueError: If the proposal was made from a configuration other
                than the current one, so its energy change no longer
                applies. This happens when two proposals are made and both
                are applied: the second must be proposed after the first is
                applied.
        """
        if proposal.source is not self.configuration:
            raise ValueError(
                "This proposal was made from a configuration that is no longer the current "
                "one, so its energy change no longer applies. Propose again from the current "
                "configuration."
            )
        self.configuration = self.configuration.replace(position=proposal.position)
        self.energy += proposal.energy_change

    def step(self) -> None:
        """Propose a move, accept it by the Metropolis criterion or leave
        the configuration as it is, and advance the step count."""
        proposal = self.propose()
        if accept(proposal.energy_change, self.temperature, rng=self.rng):
            self.apply(proposal)
            self.accepted += 1
        self.steps += 1

    def sample(self) -> None:
        """Record the configuration in the trajectory and the step and energy
        in the samples.

        The energy is recomputed from the configuration first, so the
        recorded value is exact; between samples ``apply`` keeps a running
        total.
        """
        self.trajectory.append(self.configuration)
        self.energy = self.configuration.potential_energy(self.model, self.cut_off)
        self.samples.add(step=self.steps, potential_energy=self.energy)

    def restart(self) -> Self:
        """A new simulation that continues from the current configuration.

        See :meth:`Simulation.restart`.
        """
        new = super().restart()
        new.accepted = 0
        return new
