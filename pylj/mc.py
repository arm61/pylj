"""Monte Carlo: the simulation that samples configurations by the Metropolis
criterion, the criterion itself, and the proposal it decides on."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

from pylj.configuration import Configuration
from pylj.constants import BOLTZMANN
from pylj.pairwise import PairPotentials
from pylj.placement import place
from pylj.potentials import Species
from pylj.simulation import (
    Samples,
    Simulation,
    _check_initial_energy,
    _check_positive_finite,
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


@dataclass(frozen=True)
class Proposal:
    """A proposed configuration for a Monte Carlo move.

    The energy change is relative to the configuration that was current when
    the proposal was made, so a proposal is applied to that configuration.

    Attributes:
        position: The proposed position of every particle, shape ``(N, 2)``,
            in metres.
        energy_change: The energy of the proposed configuration minus that
            of the configuration it was proposed from, in joules.
    """

    position: NDArray[np.float64]
    energy_change: float


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
        pair_potentials: The potential between each pair of species. Only
            the pair energies are evaluated, so a potential with no finite
            force, such as the square well, can be used.
        temperature: The temperature of the simulation, in kelvin.
        cut_off: The cut-off, in metres; see :class:`Simulation`.
        seed: Seed for the random number generator.

    Attributes:
        temperature: The temperature, in kelvin.
        energy: The total pair energy of the current configuration, in
            joules: computed at construction, kept current by ``apply`` and
            set exactly on each ``sample``.
        accepted: The number of moves accepted so far.
        samples: The :class:`MCSamples` record that ``sample`` appends to.

    Raises:
        ValueError: If the temperature is not positive and finite, a pair
            potential has not died away at the cut-off, the configuration
            stores more than :data:`simulation.INITIAL_ENERGY_LIMIT` k_B T of
            potential energy per particle, or for anything
            :class:`Simulation` rejects.
    """

    samples: MCSamples

    def __init__(
        self,
        configuration: Configuration,
        pair_potentials: PairPotentials,
        temperature: float,
        *,
        cut_off: float | None = None,
        seed: int | None = None,
    ) -> None:
        _check_positive_finite("temperature", temperature)
        super().__init__(configuration, pair_potentials, cut_off=cut_off, seed=seed)
        self.temperature = temperature
        _check_potentials_at_the_cut_off(
            configuration.species,
            self.pair_potentials,
            self.cut_off,
            temperature,
            configuration.box,
        )
        self.energy = configuration.potential_energy(self.pair_potentials, self.cut_off)
        _check_initial_energy(self.energy, configuration.number_of_particles, temperature)
        self.accepted = 0
        self.samples = MCSamples()

    @classmethod
    def initialise(
        cls,
        number_of_particles: int,
        temperature: float,
        box: float,
        *,
        species: Sequence[Species],
        pair_potentials: PairPotentials,
        init_conf: str = "square",
        placement_temperature: float | None = None,
        cut_off: float | None = None,
        seed: int | None = None,
    ) -> Self:
        """Build a simulation from a model: place the particles and set the
        temperature.

        Args:
            number_of_particles: The number of particles.
            temperature: The temperature of the simulation, in kelvin.
            box: The side length of the box, in Angstrom, from 4 to 600.
            species: The species; particles are assigned to them in turn.
            pair_potentials: The potential between each pair of species.
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
            number_of_particles,
            temperature,
            box,
            species=species,
            pair_potentials=pair_potentials,
            init_conf=init_conf,
            placement_temperature=placement_temperature,
            cut_off=cut_off,
            rng=rng,
        )
        simulation = cls(configuration, pair_potentials, temperature, cut_off=cut_off_metres)
        simulation.rng = rng
        return simulation

    def propose(self) -> Proposal:
        """Propose a move: one particle relocated at random.

        A particle is chosen at random and given a uniform trial position in
        the box. The energy change is that particle's interaction energy at
        the trial position minus that at its current position, with every
        other particle. The configuration is not changed.

        Returns:
            The proposed configuration and its energy change.
        """
        configuration = self.configuration
        particle = int(self.rng.integers(configuration.number_of_particles))
        trial = self.rng.uniform(0, configuration.box, size=2)
        current = configuration.position[particle]
        species_index = int(configuration.species_index[particle])
        others = configuration.without(particle)
        energy_change = others.insertion_energy(
            trial, species_index, self.pair_potentials, self.cut_off
        ) - others.insertion_energy(current, species_index, self.pair_potentials, self.cut_off)
        position = configuration.position.copy()
        position[particle] = trial
        return Proposal(position, energy_change)

    def apply(self, proposal: Proposal) -> None:
        """Make a proposed configuration the current one.

        The positions become the proposal's and ``energy`` gains its energy
        change.

        Args:
            proposal: The proposal to apply, from :meth:`propose`.
        """
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
        """Record the step and the energy.

        The energy is recomputed from the configuration first, so the
        recorded value is exact; between samples ``apply`` keeps a running
        total.
        """
        self.energy = self.configuration.potential_energy(self.pair_potentials, self.cut_off)
        self.samples.add(step=self.steps, potential_energy=self.energy)

    def restart(self) -> Self:
        """A new simulation that continues from the current configuration.

        See :meth:`Simulation.restart`.
        """
        new = super().restart()
        new.accepted = 0
        return new
