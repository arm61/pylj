"""Molecular dynamics simulation."""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

from pylj import pairwise
from pylj.configuration import MDConfiguration
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
class MDSamples(Samples):
    """The record a molecular dynamics simulation appends to at each sample.

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

    Args:
        configuration: The starting configuration, with velocities. The
            simulation starts from a copy with the centre of mass at rest;
            see :func:`at_rest`.
        model: The model.
        cut_off: The cut-off, in metres; see :class:`Simulation`.
        timestep: The length of each integration step, in seconds.
        seed: Seed for the random number generator.

    Attributes:
        configuration: The current configuration.
        forces: The net force on each atom at the current
            configuration, shape ``(N, 2)``, in newtons.
        timestep: The length of each step, in seconds.
        initial_configuration: The configuration the mean squared
            displacement is measured from.
        samples: The :class:`MDSamples` record that ``sample`` appends to.

    Raises:
        TypeError: If ``configuration`` is not an ``MDConfiguration``.
        ValueError: If the timestep is not positive and finite, the
            configuration is at rest or has a non-finite temperature, a pair
            potential has not died away at the cut-off, or the configuration
            stores more than :data:`simulation.INITIAL_ENERGY_LIMIT` k_B T
            of potential energy per atom.
    """

    configuration: MDConfiguration
    samples: MDSamples

    def __init__(
        self,
        configuration: MDConfiguration,
        model: Model,
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
        configuration = at_rest(configuration)
        super().__init__(configuration, model, cut_off=cut_off, seed=seed)
        check_positive_finite("timestep", timestep)
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
        _check_potentials_at_the_cut_off(self.model, self.cut_off, temperature, configuration.box)
        _check_initial_energy(
            configuration.potential_energy(self.model, self.cut_off),
            configuration.number_of_atoms,
            temperature,
        )
        self.forces = configuration.forces(self.model, self.cut_off)
        self.initial_configuration = configuration
        self.samples = MDSamples()

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
        timestep: float = 1e-14,
        cut_off: float | None = None,
        seed: int | None = None,
    ) -> Self:
        """Builds a simulation from a model.

        Places the atoms in the box and draws their velocities at random,
        scaled so the initial temperature is exactly the one asked for. The
        temperature is not stored: a molecular dynamics run measures its own.

        Args:
            model: The model.
            number_of_atoms: The number of atoms, at least two.
            temperature: The initial temperature, in kelvin.
            box: The side length of the box, in Angstrom, from
                :data:`~pylj.placement.SMALLEST_BOX` to :data:`~pylj.placement.LARGEST_BOX`.
            init_conf: How the atoms are placed. ``'square'`` puts them on a
                square grid, ``'triangular'`` on a triangular lattice filling
                the box, which constrains the number of atoms, and
                ``'metropolis'`` inserts them at random positions for a
                disordered start.
            placement_temperature: The temperature of the Metropolis
                acceptance used by ``'metropolis'``, in kelvin; by default
                the run temperature.
            max_strain: How far the fitted lattice may sit from
                ``sqrt(3) / 2``, as a fraction of that ratio, when
                ``init_conf`` is ``'triangular'``. Used only by that
                placement.
            timestep: The length of each integration step, in seconds.
            cut_off: The cut-off, in Angstrom; by default
                :data:`~pylj.simulation.DEFAULT_CUT_OFF` Angstrom or half the
                box, whichever is smaller.
            seed: Seed for the random number generator used to place the
                configuration, draw the velocities and continue the run;
                without one the run differs each time.

        Returns:
            The simulation, with its forces evaluated.

        Raises:
            ValueError: If fewer than two atoms are requested, or for
                anything :func:`placement.place` or the constructor
                rejects.
        """
        if number_of_atoms < 2:
            raise ValueError(
                "Molecular dynamics needs at least two atoms: with one atom "
                "there is no thermal motion once the centre-of-mass velocity is removed."
            )
        rng = np.random.default_rng(seed)
        placed, cut_off_metres = place(
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
        masses = placed.masses
        thermal_speed = np.sqrt(BOLTZMANN * temperature / masses)
        velocity = rng.normal(0.0, thermal_speed[:, None], size=(number_of_atoms, 2))
        configuration = heat_bath(
            at_rest(
                MDConfiguration(
                    position=placed.position,
                    species=placed.species,
                    species_index=placed.species_index,
                    box=placed.box,
                    velocity=velocity,
                    unwrapped=placed.position.copy(),
                )
            ),
            temperature,
        )
        simulation = cls(configuration, model, cut_off=cut_off_metres, timestep=timestep)
        simulation.rng = rng
        return simulation

    @property
    def time(self) -> float:
        """The simulated time, in seconds."""
        return self.steps * self.timestep

    def integrate(self) -> None:
        """Moves the configuration one timestep forward with Velocity-Verlet,
        replacing the configuration and the forces.
        """
        self.configuration, self.forces = velocity_verlet(
            self.configuration, self.forces, self.timestep, self.model, self.cut_off
        )

    def step(self) -> None:
        """Integrates one timestep and advance the clock.

        Raises:
            ValueError: If an atom moves further than half the cut-off in
                the step, or a pair comes closer than its potential allows.
        """
        self.integrate()
        self.steps += 1

    def heat_bath(self, bath_temperature: float) -> None:
        """Rescales the velocities to the bath temperature.

        Args:
            bath_temperature: The desired temperature, in kelvin.

        Raises:
            ValueError: If the bath temperature is not positive and finite,
                the atoms are at rest, or the simulation has diverged.
        """
        self.configuration = heat_bath(self.configuration, bath_temperature)

    def sample(self) -> None:
        """Records the configuration in the trajectory and measures it into
        :class:`MDSamples`."""
        self.trajectory.append(self.configuration)
        configuration = self.configuration
        kinetic_energy = configuration.kinetic_energy()
        pairs = configuration.pairs(self.model, self.cut_off, forces=True)
        self.samples.add(
            step=self.steps,
            temperature=configuration.temperature(),
            pressure=pairwise.calculate_pressure(pairs.virial, configuration.box, kinetic_energy),
            potential_energy=float(pairs.energy.sum()),
            kinetic_energy=kinetic_energy,
            msd=configuration.msd(self.initial_configuration),
        )

    def restart(self) -> Self:
        """Returns a new simulation continuing from the current configuration,
        with the mean squared displacement measured from it again.

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
    model: Model,
    cut_off: float,
) -> tuple[MDConfiguration, NDArray[np.float64]]:
    """Moves a configuration one timestep forward with the Velocity-Verlet
    integrator.

    Args:
        configuration: The configuration at time t.
        forces: The net force on each atom at that configuration, shape
            ``(N, 2)``, in newtons.
        timestep: The length of the step, in seconds.
        model: The model.
        cut_off: The cut-off, in metres.

    Returns:
        The configuration at time t + dt and the forces at it.

    Raises:
        ValueError: If an atom moves further than half the cut-off in the
            one step.
    """
    masses = configuration.masses[:, None]
    accelerations = forces / masses
    position, unwrapped = update_positions(configuration, accelerations, timestep)
    furthest = float(np.linalg.norm(unwrapped - configuration.unwrapped, axis=1).max())
    if not furthest < cut_off / 2:
        raise ValueError(
            f"An atom moved {furthest * 1e10:.3g} Angstrom in a single step of "
            f"{timestep:.3g} s, more than half the cut-off of {cut_off * 1e10:.3g} Angstrom: "
            "the timestep is too long, or the simulation has diverged."
        )
    moved = configuration.replace(position=position, unwrapped=unwrapped)
    next_forces = moved.forces(model, cut_off)
    next_accelerations = next_forces / masses
    velocity = update_velocities(
        configuration.velocity, accelerations, next_accelerations, timestep
    )
    return moved.replace(velocity=velocity), next_forces


def update_positions(
    configuration: MDConfiguration, accelerations: NDArray[np.float64], timestep: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advances the positions by one timestep.

    Args:
        configuration: The configuration to advance.
        accelerations: The acceleration of each atom, shape ``(N, 2)``,
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
    """Advances the velocities by one timestep.

    Args:
        velocity: The velocity of each atom, shape ``(N, 2)``, in metres
            per second.
        accelerations: The accelerations at the start of the step.
        next_accelerations: The accelerations at the end of the step.
        timestep: The length of the step, in seconds.

    Returns:
        The new velocities.
    """
    return velocity + 0.5 * (accelerations + next_accelerations) * timestep


def at_rest(configuration: MDConfiguration) -> MDConfiguration:
    """Returns the configuration with its centre of mass at rest.

    The mass-weighted mean velocity is subtracted from every atom.

    Args:
        configuration: The configuration to bring to rest.

    Returns:
        A copy with the centre of mass at rest.
    """
    masses = configuration.masses[:, None]
    drift = (masses * configuration.velocity).sum(axis=0) / masses.sum()
    return configuration.replace(velocity=configuration.velocity - drift)


def heat_bath(configuration: MDConfiguration, bath_temperature: float) -> MDConfiguration:
    """Rescales the velocities so the instantaneous temperature equals the
    bath temperature.

    Args:
        configuration: The configuration to thermostat.
        bath_temperature: The desired temperature, in kelvin.

    Returns:
        The configuration with the velocities rescaled.

    Raises:
        ValueError: If the bath temperature is not positive and finite, the
            atoms are at rest, or the current temperature is not finite.
    """
    check_positive_finite("bath_temperature", bath_temperature)
    current = configuration.temperature()
    if current == 0:
        raise ValueError("Cannot rescale velocities: the atoms are at rest.")
    if not (np.isfinite(current) and current > 0):
        raise ValueError(
            f"Cannot rescale velocities: the current temperature is {current}, so the "
            "simulation has diverged."
        )
    return configuration.replace(
        velocity=configuration.velocity * np.sqrt(bath_temperature / current)
    )
