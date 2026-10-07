"""Molecular dynamics simulation."""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

from pylj import pairwise
from pylj.configuration import MDConfiguration
from pylj.constants import BOLTZMANN, KJ_PER_MOL
from pylj.model import Model
from pylj.placement import place
from pylj.potentials import check_non_negative_finite, check_positive_finite
from pylj.simulation import Samples, Simulation, _empty
from pylj.trajectory import Trajectory


@dataclass
class MDSamples(Samples):
    """The record a molecular dynamics simulation appends to at each sample.

    Attributes:
        temperature: The instantaneous temperature, in kelvin.
        pressure: The pressure, in kJ/mol per Angstrom squared.
        potential_energy: The total pair energy, in kJ/mol.
        kinetic_energy: The total kinetic energy, in kJ/mol.
    """

    temperature: NDArray[np.float64] = field(default_factory=_empty)
    pressure: NDArray[np.float64] = field(default_factory=_empty)
    potential_energy: NDArray[np.float64] = field(default_factory=_empty)
    kinetic_energy: NDArray[np.float64] = field(default_factory=_empty)

    @property
    def total_energy(self) -> NDArray[np.float64]:
        """The potential plus the kinetic energy at each sample, in kJ/mol."""
        return self.potential_energy + self.kinetic_energy


class MDSimulation(Simulation):
    """A molecular dynamics simulation.

    Args:
        configuration: The starting configuration, with velocities. The
            simulation runs on its own copy, with the centre of mass at rest.
        model: The model.
        cut_off: The cut-off, in Angstrom; see :class:`Simulation`.
        timestep: The length of each integration step, in picoseconds.
        seed: Seed for the random number generator.

    Attributes:
        configuration: The current configuration.
        timestep: The length of each step, in picoseconds.
        samples: The :class:`MDSamples` record that ``sample`` appends to.

    Raises:
        TypeError: If ``configuration`` is not an ``MDConfiguration``.
        ValueError: If the timestep is not positive and finite.
    """

    configuration: MDConfiguration
    samples: MDSamples

    __slots__ = ("timestep", "_forces")

    def __init__(
        self,
        configuration: MDConfiguration,
        model: Model,
        *,
        cut_off: float | None = None,
        timestep: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if not isinstance(configuration, MDConfiguration):
            raise TypeError(
                "MDSimulation needs an MDConfiguration, which carries velocities; build one "
                "with MDSimulation.initialise(...) or construct an MDConfiguration."
            )
        check_positive_finite("timestep", timestep)
        self.timestep = timestep
        super().__init__(configuration, model, cut_off=cut_off, seed=seed)
        self._remove_drift()
        self.trajectory = Trajectory(times=[])
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
        timestep: float = 0.01,
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
            timestep: The length of each integration step, in picoseconds.
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
                "Molecular dynamics needs at least two atoms: the temperature is "
                "undefined for a single atom."
            )
        rng = np.random.default_rng(seed)
        placed = place(
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
        thermal_speed = np.sqrt(BOLTZMANN * temperature * KJ_PER_MOL / masses)
        velocities = rng.normal(0.0, thermal_speed[:, None], size=(number_of_atoms, 2))
        configuration = MDConfiguration(
            positions=placed.positions,
            species=placed.species,
            species_index=placed.species_index,
            box=placed.box,
            velocities=velocities,
        )
        simulation = cls(configuration, model, cut_off=cut_off, timestep=timestep)
        simulation.rescale_velocities(temperature)
        simulation.rng = rng
        return simulation

    @property
    def time(self) -> float:
        """The simulated time, in picoseconds."""
        return self.steps * self.timestep

    def _remove_drift(self) -> None:
        """Subtracts the mass-weighted mean velocity from every atom, so the
        centre of mass is at rest."""
        configuration = self.configuration
        masses = configuration.masses[:, None]
        drift = (masses * configuration.velocities).sum(axis=0) / masses.sum()
        configuration.velocities = configuration.velocities - drift

    def _recompute_from_configuration(self) -> None:
        self._forces = self.configuration.forces(self.model, self.cut_off)

    @property
    def forces(self) -> NDArray[np.float64]:
        """The net force on each atom at the current configuration, shape
        ``(N, 2)``, in kJ/mol/Angstrom."""
        self._bring_up_to_date()
        return self._forces

    def integrate(self) -> None:
        """Moves the atoms one timestep forward with Velocity-Verlet.

        Raises:
            ValueError: If an atom would move further than half the cut-off
                in the step, or a pair comes closer than its potential
                allows.
        """
        configuration = self.configuration
        masses = configuration.masses[:, None]
        accelerations = self.forces / masses * KJ_PER_MOL
        displacement = (
            configuration.velocities * self.timestep + 0.5 * accelerations * self.timestep**2
        )
        furthest = float(np.linalg.norm(displacement, axis=1).max())
        if not furthest < self.cut_off / 2:
            raise ValueError(
                f"An atom moved {furthest:.3g} Angstrom in a single step of "
                f"{self.timestep:.3g} ps, more than half the cut-off of {self.cut_off:.3g} "
                "Angstrom: the timestep is too long, or the simulation has diverged."
            )
        moved = configuration.positions + displacement
        configuration.positions = moved % configuration.box
        crossings = np.floor(moved / configuration.box).astype(np.int64)
        configuration.images = configuration.images + crossings
        forces = configuration.forces(self.model, self.cut_off)
        next_accelerations = forces / masses * KJ_PER_MOL
        configuration.velocities = (
            configuration.velocities + 0.5 * (accelerations + next_accelerations) * self.timestep
        )
        self._forces = forces
        self._keep()

    def step(self) -> None:
        """Integrates one timestep and advances the clock.

        Raises:
            ValueError: If an atom moves further than half the cut-off in
                the step, or a pair comes closer than its potential allows.
        """
        self.integrate()
        self.steps += 1

    def rescale_velocities(self, temperature: float) -> None:
        """Rescales the velocities so the instantaneous temperature equals
        ``temperature``.

        Args:
            temperature: The temperature to rescale to, in kelvin.

        Raises:
            ValueError: If ``temperature`` is negative or not finite, the
                atoms are at rest and ``temperature`` is above zero, or the
                current temperature is not finite.
        """
        check_non_negative_finite("temperature", temperature)
        configuration = self.configuration
        current = configuration.temperature()
        if current == 0:
            if temperature == 0:
                return
            raise ValueError("Cannot rescale velocities: the atoms are at rest.")
        if not (np.isfinite(current) and current > 0):
            raise ValueError(
                f"Cannot rescale velocities: the current temperature is {current}, so the "
                "simulation has diverged."
            )
        configuration.velocities = configuration.velocities * np.sqrt(temperature / current)

    def sample(self) -> None:
        """Records a copy of the configuration in the trajectory and measures
        the configuration into :class:`MDSamples`."""
        self._bring_up_to_date()
        configuration = self.configuration
        self.trajectory.append(configuration.copy(), self.time)
        kinetic_energy = configuration.kinetic_energy()
        pairs = configuration.pairs(self.model, self.cut_off, forces=True)
        self.samples.add(
            step=self.steps,
            temperature=configuration.temperature(),
            pressure=pairwise.calculate_pressure(pairs.virial, configuration.box, kinetic_energy),
            potential_energy=float(pairs.energies.sum()),
            kinetic_energy=kinetic_energy,
        )

