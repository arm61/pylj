"""Molecular dynamics: the simulation that integrates Newton's equations of
motion, the Velocity-Verlet integrator, and the velocity-rescaling
thermostat."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

from pylj import pairwise
from pylj.configuration import MDConfiguration
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
class MDSamples(Samples):
    """What a molecular dynamics simulation records at each sample.

    Attributes:
        temperature: The instantaneous temperature, in kelvin.
        pressure: The pressure, in newtons per metre.
        potential_energy: The total pair energy, in joules.
        kinetic_energy: The total kinetic energy, in joules.
        msd: The mean squared displacement from the initial configuration,
            in metres squared.
    """

    temperature: NDArray[np.float64] = field(default_factory=_empty)
    pressure: NDArray[np.float64] = field(default_factory=_empty)
    potential_energy: NDArray[np.float64] = field(default_factory=_empty)
    kinetic_energy: NDArray[np.float64] = field(default_factory=_empty)
    msd: NDArray[np.float64] = field(default_factory=_empty)

    @property
    def total_energy(self) -> NDArray[np.float64]:
        """The potential plus the kinetic energy at each sample, in joules."""
        return self.potential_energy + self.kinetic_energy


class MDSimulation(Simulation):
    """A molecular dynamics simulation.

    The state between steps is the configuration and the forces at it, the
    two registers the Velocity-Verlet integrator needs. ``step`` integrates
    one timestep and advances the clock; ``sample`` measures the
    configuration.

    Args:
        configuration: The starting configuration, with velocities.
        pair_potentials: The potential between each pair of species.
        cut_off: The cut-off, in metres; see :class:`Simulation`.
        timestep: The length of each integration step, in seconds.
        seed: Seed for the random number generator.

    Attributes:
        configuration: The current configuration.
        forces: The net force on each particle at the current
            configuration, shape ``(N, 2)``, in newtons.
        timestep: The length of each step, in seconds.
        initial_configuration: The configuration the mean squared
            displacement is measured from.
        samples: The :class:`MDSamples` record that ``sample`` appends to.

    Raises:
        TypeError: If ``configuration`` is not an ``MDConfiguration``.
        ValueError: If the timestep is not positive and finite, the
            configuration is at rest or has a non-finite temperature, a pair
            potential has
            not died away at the cut-off at the configuration's
            temperature, the configuration stores more than
            :data:`simulation.INITIAL_ENERGY_LIMIT` k_B T of potential
            energy per particle, or for anything :class:`Simulation`
            rejects.
    """

    configuration: MDConfiguration
    samples: MDSamples

    def __init__(
        self,
        configuration: MDConfiguration,
        pair_potentials: PairPotentials,
        *,
        cut_off: float | None = None,
        timestep: float = 1e-14,
        seed: int | None = None,
    ) -> None:
        if not isinstance(configuration, MDConfiguration):
            raise TypeError(
                "MDSimulation needs an MDConfiguration, which carries velocities; build one "
                "with MDSimulation.initialise(...) or construct an MDConfiguration."
            )
        super().__init__(configuration, pair_potentials, cut_off=cut_off, seed=seed)
        _check_positive_finite("timestep", timestep)
        self.timestep = timestep
        temperature = configuration.temperature()
        if temperature == 0:
            raise ValueError(
                "The configuration is at rest: molecular dynamics needs velocities. "
                "MDSimulation.initialise draws them at a temperature."
            )
        if not np.isfinite(temperature):
            raise ValueError(
                f"The configuration's temperature is {temperature}: the simulation it came "
                "from has diverged."
            )
        _check_potentials_at_the_cut_off(
            configuration.species,
            self.pair_potentials,
            self.cut_off,
            temperature,
            configuration.box,
        )
        _check_initial_energy(
            configuration.potential_energy(self.pair_potentials, self.cut_off),
            configuration.number_of_particles,
            temperature,
        )
        self.forces = configuration.forces(self.pair_potentials, self.cut_off)
        self.initial_configuration = configuration
        self.samples = MDSamples()

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
        timestep: float = 1e-14,
        cut_off: float | None = None,
        seed: int | None = None,
    ) -> Self:
        """Build a simulation from a model: place the particles and draw
        their velocities at a temperature.

        Each velocity component is drawn from a normal distribution with
        the thermal width for the particle's mass at the temperature, the
        centre-of-mass velocity is removed, and the velocities are rescaled
        so that the instantaneous temperature is exactly the one requested.
        The temperature is not stored: molecular dynamics measures it.

        Args:
            number_of_particles: The number of particles, at least two.
            temperature: The initial temperature, in kelvin.
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
            timestep: The length of each integration step, in seconds.
            cut_off: The cut-off, in Angstrom; by default 15 Angstrom or
                half the box, whichever is smaller.
            seed: Seed for the random number generator used to place the
                configuration, draw the velocities and continue the run.

        Returns:
            The simulation, with its forces evaluated.

        Raises:
            ValueError: If fewer than two particles are requested, or for
                anything :func:`placement.place` or the constructor
                rejects.
        """
        if number_of_particles < 2:
            raise ValueError(
                "Molecular dynamics needs at least two particles: with one particle "
                "there is no thermal motion once the centre-of-mass velocity is removed."
            )
        rng = np.random.default_rng(seed)
        placed, cut_off_metres = place(
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
        masses = placed.masses
        thermal_speed = np.sqrt(BOLTZMANN * temperature / masses)
        velocity = rng.normal(0.0, thermal_speed[:, None], size=(number_of_particles, 2))
        velocity -= (masses[:, None] * velocity).sum(axis=0) / masses.sum()
        configuration = heat_bath(
            MDConfiguration(
                position=placed.position,
                species=placed.species,
                species_index=placed.species_index,
                box=placed.box,
                velocity=velocity,
                unwrapped=placed.position.copy(),
            ),
            temperature,
        )
        simulation = cls(configuration, pair_potentials, cut_off=cut_off_metres, timestep=timestep)
        simulation.rng = rng
        return simulation

    @property
    def time(self) -> float:
        """The simulated time, in seconds: the steps taken times the timestep."""
        return self.steps * self.timestep

    def integrate(self) -> None:
        """Move the configuration one timestep forward with Velocity-Verlet,
        replacing the configuration and the forces.

        A subclass with a different integrator overrides this method.
        """
        self.configuration, self.forces = velocity_verlet(
            self.configuration, self.forces, self.timestep, self.pair_potentials, self.cut_off
        )

    def step(self) -> None:
        """Integrate one timestep and advance the clock."""
        self.integrate()
        self.steps += 1

    def heat_bath(self, bath_temperature: float) -> None:
        """Rescale the velocities to the bath temperature.

        Args:
            bath_temperature: The desired temperature, in kelvin.

        Raises:
            ValueError: If the bath temperature is not positive, or the
                particles are at rest or the simulation has diverged.
        """
        self.configuration = heat_bath(self.configuration, bath_temperature)

    def sample(self) -> None:
        """Measure the configuration: record the step, temperature, pressure,
        potential and kinetic energies and mean squared displacement.
        """
        configuration = self.configuration
        temperature = configuration.temperature()
        pairs = configuration.pairs(self.pair_potentials, self.cut_off, forces=True)
        self.samples.add(
            step=self.steps,
            temperature=temperature,
            pressure=pairwise.calculate_pressure(
                pairs.virial, configuration.box, configuration.number_of_particles, temperature
            ),
            potential_energy=float(pairs.energy.sum()),
            kinetic_energy=configuration.kinetic_energy(),
            msd=configuration.msd(self.initial_configuration),
        )

    def restart(self) -> Self:
        """A new simulation that continues from the current configuration,
        with the mean squared displacement measured from it.

        See :meth:`Simulation.restart`.
        """
        new = super().restart()
        new.configuration = self.configuration.replace(unwrapped=self.configuration.position.copy())
        new.initial_configuration = new.configuration
        new.forces = self.forces.copy()
        return new


def velocity_verlet(
    configuration: MDConfiguration,
    forces: NDArray[np.float64],
    timestep: float,
    pair_potentials: PairPotentials,
    cut_off: float,
) -> tuple[MDConfiguration, NDArray[np.float64]]:
    """Move a configuration one timestep forward with the Velocity-Verlet
    integrator.

    The positions are advanced with the current velocities and
    accelerations, the forces are evaluated at the new positions, and the
    velocities are advanced with the mean of the old and new accelerations.

    Args:
        configuration: The configuration at time t.
        forces: The net force on each particle at that configuration, shape
            ``(N, 2)``, in newtons.
        timestep: The length of the step, in seconds.
        pair_potentials: The potential between each pair of species.
        cut_off: The cut-off, in metres.

    Returns:
        The configuration at time t + dt and the forces at it.
    """
    masses = configuration.masses[:, None]
    accelerations = forces / masses
    position, unwrapped = update_positions(configuration, accelerations, timestep)
    moved = configuration.replace(position=position, unwrapped=unwrapped)
    next_forces = moved.forces(pair_potentials, cut_off)
    next_accelerations = next_forces / masses
    velocity = update_velocities(
        configuration.velocity, accelerations, next_accelerations, timestep
    )
    return moved.replace(velocity=velocity), next_forces


def update_positions(
    configuration: MDConfiguration, accelerations: NDArray[np.float64], timestep: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance the positions by ``v dt + a dt^2 / 2``.

    Args:
        configuration: The configuration to advance.
        accelerations: The acceleration of each particle, shape ``(N, 2)``,
            in metres per second squared.
        timestep: The length of the step, in seconds.

    Returns:
        The new positions, wrapped into the box, and the new unwrapped
        positions.
    """
    displacement = configuration.velocity * timestep + 0.5 * accelerations * timestep**2
    position = (configuration.position + displacement) % configuration.box
    return position, configuration.unwrapped + displacement


def update_velocities(
    velocity: NDArray[np.float64],
    accelerations: NDArray[np.float64],
    next_accelerations: NDArray[np.float64],
    timestep: float,
) -> NDArray[np.float64]:
    """Advance the velocities by the mean acceleration times the timestep.

    Args:
        velocity: The velocity of each particle, shape ``(N, 2)``.
        accelerations: The accelerations at the start of the step.
        next_accelerations: The accelerations at the end of the step.
        timestep: The length of the step, in seconds.

    Returns:
        The new velocities.
    """
    return velocity + 0.5 * (accelerations + next_accelerations) * timestep


def heat_bath(configuration: MDConfiguration, bath_temperature: float) -> MDConfiguration:
    r"""Rescale the velocities so the instantaneous temperature equals the
    bath temperature.

    This is a velocity-rescaling thermostat: each call sets the
    instantaneous temperature to the bath temperature, scaling every
    velocity by

    .. math::
        \sqrt{T_{\text{bath}} / T_{\text{now}}}

    where :math:`T_{\text{now}}` is the temperature of the current
    velocities.

    Args:
        configuration: The configuration to thermostat.
        bath_temperature: The desired temperature, in kelvin.

    Returns:
        The configuration with the velocities rescaled.

    Raises:
        ValueError: If the bath temperature is not positive, the particles
            are at rest, or the current temperature is not finite (the
            simulation has diverged).
    """
    if not bath_temperature > 0:
        raise ValueError(f"bath_temperature must be positive, not {bath_temperature}")
    current = configuration.temperature()
    if current == 0:
        raise ValueError("Cannot rescale velocities: the particles are at rest.")
    if not (np.isfinite(current) and current > 0):
        raise ValueError(
            f"Cannot rescale velocities: the current temperature is {current}, so the "
            "simulation has diverged."
        )
    return configuration.replace(
        velocity=configuration.velocity * np.sqrt(bath_temperature / current)
    )
