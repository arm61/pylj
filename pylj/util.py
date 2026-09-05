import copy
import itertools
import webbrowser
from collections.abc import Sequence
from typing import Literal, Self

import numpy as np

from pylj import mc, md, pairwise
from pylj.constants import BOLTZMANN
from pylj.pairwise import PairPotentials
from pylj.potentials import PairPotential, Species

#: Number of trial positions tried for a single particle by Metropolis
#: placement before it gives up and raises ``ValueError``.
PLACEMENT_ATTEMPTS = 1000


def _check_pair_potentials(species: Sequence[Species], pair_potentials: PairPotentials) -> None:
    """Check that a system's model is complete.

    Args:
        species: The species in the system.
        pair_potentials: The potential between each pair of species.

    Raises:
        ValueError: If ``species`` is empty, or a pair of species has no
            entry in ``pair_potentials`` in either order.
        TypeError: If a value in ``pair_potentials`` is not a
            ``PairPotential`` instance, such as the class itself.
    """
    if not species:
        raise ValueError("species must name at least one Species")
    for one, other in itertools.combinations_with_replacement(species, 2):
        if (one, other) not in pair_potentials and (other, one) not in pair_potentials:
            raise ValueError(f"pair_potentials has no entry for the pair {one} and {other}")
    for one, other in itertools.combinations(species, 2):
        if (one, other) in pair_potentials and (other, one) in pair_potentials:
            raise ValueError(
                f"pair_potentials has the pair {one} and {other} in both orders; "
                "give each unordered pair once"
            )
    for pair, potential in pair_potentials.items():
        if not isinstance(potential, PairPotential):
            raise TypeError(
                f"pair_potentials[{pair}] must be a PairPotential instance, such as "
                f"LennardJones(epsilon=..., sigma=...), not {potential!r}"
            )


def _check_potentials_at_the_cut_off(
    species: Sequence[Species], pair_potentials: PairPotentials, cut_off: float, temperature: float
) -> None:
    """Check that every pair potential's energy has died away at the cut-off.

    Args:
        species: The species in the system.
        pair_potentials: The potential between each pair of species.
        cut_off: The cut-off, in metres.
        temperature: The temperature, in kelvin.

    Raises:
        ValueError: If any pair potential's energy at the cut-off is not
            finite or exceeds ``k_B T``, which is what a potential whose
            parameters are in the wrong units looks like.
    """
    for one, other in itertools.combinations_with_replacement(species, 2):
        potential = pairwise.pair_potential(pair_potentials, one, other)
        at_cut_off = np.asarray(potential.energies(np.array([cut_off])), dtype=float).reshape(-1)
        energy = float(at_cut_off[0])
        if not np.isfinite(energy) or energy > BOLTZMANN * temperature:
            pair = f"{one.name or 'particles'} and {other.name or 'particles'}"
            raise ValueError(
                f"{type(potential).__name__} between {pair} is still "
                f"{energy / (BOLTZMANN * temperature):.3g} k_B T at the cut-off of "
                f"{cut_off * 1e10:.1f} Angstrom, where the interaction should have died away. "
                "Check that its parameters are in metres and joules, or use a larger box or "
                "cut-off."
            )


#: Largest potential energy per particle, in units of k_B T, accepted for an
#: initial configuration by :func:`md.initialise` and :func:`mc.initialise`.
INITIAL_ENERGY_LIMIT = 10.0


def check_initial_energy(system: "System", energy: float) -> None:
    """Refuse a starting configuration that stores far more potential energy
    than thermal energy.

    Potential energy stored in an initial configuration is released as
    motion over the first steps. With the thermal energy already present, a
    configuration holding ``x`` k_B T per particle can by equipartition
    heat to as much as roughly ``x + 1`` times its temperature; how much of
    the stored energy is released depends on where the configuration
    relaxes to. Overlapping particles store enormous energy.

    Args:
        system: The system, after placement.
        energy: The total pair energy of the initial configuration, in joules.

    Raises:
        ValueError: If ``energy`` is not finite, or exceeds
            :data:`INITIAL_ENERGY_LIMIT` k_B T per particle.
    """
    if not np.isfinite(energy):
        raise ValueError(
            "The initial pair energy is not finite: particles sit inside a hard core. Use "
            "init_conf='metropolis', fewer particles or a larger box."
        )
    per_particle = energy / (system.number_of_particles * BOLTZMANN * system.temperature)
    if per_particle > INITIAL_ENERGY_LIMIT:
        raise ValueError(
            f"The initial configuration stores {per_particle:.3g} k_B T of potential energy per "
            f"particle, above the limit of {INITIAL_ENERGY_LIMIT:g}; released as motion it "
            f"could raise the temperature to as much as roughly {per_particle + 1:.3g} times "
            f"{system.temperature:g} K. Particles overlap: use fewer particles, a larger box, or "
            "init_conf='metropolis'."
        )


class System:
    """Simulation system.
    This class is designed to store all of the information about the job that
    is being run. This includes the particles
    object, as well as sampling objects such as the temperature, pressure, etc.
    arrays.

    Parameters
    ----------
    number_of_particles: int
        Number of particles to simulate.
    temperature: float
        Temperature of the simulation, in kelvin: the initial velocities for
        molecular dynamics, the temperature to pass to :func:`mc.accept` for
        Monte Carlo.
    box_length: float
        Length of a single dimension of the simulation square, in
        Angstrom.
    species: sequence of Species
        Required, keyword only. The species in the system. Particles are
        assigned to the species in turn, so particle i is of species
        ``i % len(species)``.
    pair_potentials: mapping of (Species, Species) to PairPotential
        Required, keyword only. The potential acting between each pair of
        species, keyed by the two species in either order. Every pair,
        including each species with itself, needs an entry.
    simulation: {'md', 'mc'}
        Required, keyword only. Which engine drives this system; set for you
        by :func:`md.initialise` and :func:`mc.initialise`.
    init_conf: string (optional)
        The way that the particles are initially positioned. Should be one of:
        - 'square', a square lattice
        - 'metropolis', sequential Metropolis insertion: each particle is
          given a uniform trial position, accepted by the Metropolis rule at
          ``placement_temperature`` on its interaction energy with the
          particles already placed
    timestep_length: float (optional)
        Length for each Velocity-Verlet integration step, in seconds.
    cut_off: float (optional)
        The distance apart that the particles must be to consider their
        interaction to be negligible.
    placement_temperature: float (optional)
        Temperature, in kelvin, of the Metropolis acceptance used by
        ``init_conf='metropolis'``; by default the run temperature. It is a
        parameter of the placement: the placed configuration is a starting
        point that the run equilibrates, and the acceptance temperature only
        sets how strictly close contacts are rejected. Raising it tolerates
        closer contacts; lowering it rejects them more strictly and can
        exhaust the trial budget.
        Ignored by ``'square'``.
    seed: int (optional)
        Seed for the random number generator used to place an initial
        configuration, draw the initial velocities and make Monte Carlo
        moves. The same seed reproduces the same run. Without one the run
        differs each time.

    Attributes
    ----------
    species: list of Species
        The species in the system; ``particles["types"]`` holds the index
        of each particle's species in this list.
    pair_potentials: dict of (Species, Species) to PairPotential
        The potential acting between each pair of species.
    masses: numpy.ndarray
        The mass of each particle, in atomic mass units, from its species.
    temperature: float
        The temperature given at construction, in kelvin.
    placement_temperature: float
        The Metropolis placement temperature, in kelvin.
    energy: float
        The total pair energy of the current configuration, in joules, for a
        Monte Carlo system: computed at construction, kept current by
        ``apply`` and set exactly on each sample. Zero for a molecular
        dynamics system.
    rng: numpy.random.Generator
        The random number generator for this system.

    Raises
    ------
    ValueError
        If the temperature or placement temperature is not positive and
        finite, ``species`` is empty, a pair of species has no potential or
        one given in both orders, a pair potential's energy at the cut-off
        is not finite or exceeds k_B T, or Metropolis placement exhausts its
        trial budget or receives NaN from a potential.
    TypeError
        If a pair potential is not a ``PairPotential`` instance.
    NotImplementedError
        If ``init_conf`` is not ``'square'`` or ``'metropolis'``.
    """

    def __init__(
        self,
        number_of_particles: int,
        temperature: float,
        box_length: float,
        *,
        species: Sequence[Species],
        pair_potentials: PairPotentials,
        simulation: Literal["md", "mc"],
        init_conf: str = "square",
        timestep_length: float = 1e-14,
        cut_off: float = 15,
        placement_temperature: float | None = None,
        seed: int | None = None,
    ):
        if simulation not in ("md", "mc"):
            raise ValueError(f"simulation must be 'md' or 'mc', not {simulation!r}")
        self.simulation = simulation
        self.number_of_particles = number_of_particles
        if not (np.isfinite(temperature) and temperature > 0):
            raise ValueError(f"temperature must be positive and finite, not {temperature}")
        self.temperature = temperature
        if placement_temperature is None:
            placement_temperature = temperature
        elif not (np.isfinite(placement_temperature) and placement_temperature > 0):
            raise ValueError(
                f"placement_temperature must be positive and finite, not {placement_temperature}"
            )
        self.placement_temperature = placement_temperature
        self.species = list(species)
        self.pair_potentials = dict(pair_potentials)
        _check_pair_potentials(self.species, self.pair_potentials)
        self.rng = np.random.default_rng(seed)
        if box_length <= 600:
            self.box_length = box_length * 1e-10
        else:
            raise AttributeError(
                f"With a box length of {box_length} the particles are "
                "probably too small to be seen in the "
                "viewer. Try something (much) less than "
                "600."
            )
        if box_length >= 4:
            self.box_length = box_length * 1e-10
        else:
            raise AttributeError(
                f"With a box length of {box_length} the cell is too "
                "small to really hold more than one "
                "particle."
            )
        self.timestep_length = timestep_length
        self.particles: np.ndarray = np.zeros(self.number_of_particles, dtype=particle_dt())
        self.particles["types"] = np.arange(self.number_of_particles) % len(self.species)
        self.masses = pairwise.particle_masses(self.particles, self.species)
        if box_length > 30:
            self.cut_off = cut_off * 1e-10
        else:
            self.cut_off = box_length / 2 * 1e-10
        _check_potentials_at_the_cut_off(
            self.species, self.pair_potentials, self.cut_off, self.temperature
        )
        if init_conf == "square":
            self.square()
        elif init_conf == "metropolis":
            self._place_by_metropolis()
        else:
            raise NotImplementedError(
                f"The initial configuration type {init_conf} is "
                "not recognised. Available options are: "
                "square or metropolis"
            )
        self.particles["xunwrapped"] = self.particles["xposition"]
        self.particles["yunwrapped"] = self.particles["yposition"]
        self.step = 0
        self.time = 0.0
        self.distances = np.zeros(self.number_of_pairs())
        self.forces = np.zeros(self.number_of_pairs())
        self.energies = np.zeros(self.number_of_pairs())
        self.temperature_sample = np.array([])
        self.pressure_sample = np.array([])
        self.force_sample = np.array([])
        self.msd_sample = np.array([])
        self.energy_sample = np.array([])
        self.step_sample = np.array([])
        self.initial_particles = np.array(self.particles)
        self.energy = 0.0
        if simulation == "mc":
            self.compute_energy()
            self.energy = float(self.energies.sum())

    def number_of_pairs(self):
        """Calculates the number of pairwise interactions in the simulation.
        Returns
        -------
        int:
            Number of pairwise interactions in the system.
        """
        return int((self.number_of_particles - 1) * self.number_of_particles / 2)

    def restart(self) -> Self:
        """Return a new system that continues from the current configuration.

        The new system keeps the box, species, pair potentials, timestep
        and cut-off, copies the state of the random number generator so the
        new system's draws do not depend on what the current system does
        next, and copies the particle positions, velocities
        and accelerations and the pair distances, forces and energies. A
        Monte Carlo system also keeps its energy. Step and time are
        zero, the sample arrays are empty, and initial_particles is replaced
        by the copied particles, so the mean squared displacement is measured
        from the restarted configuration. The current system is not changed.
        Use it to start a production run after equilibration::

            argon = Species(mass=39.948, name="argon")
            lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
            system = md.initialise(
                100, 300, 40, "metropolis",
                species=[argon], pair_potentials={(argon, argon): lj},
            )
            for _ in range(1000):
                system.integrate(md.velocity_verlet)
                system.heat_bath(300)
            production = system.restart()
            for _ in range(5000):
                production.integrate(md.velocity_verlet)
                production.step += 1
                production.time += production.timestep_length
                production.md_sample()

        Returns:
            The new system.
        """
        # A shallow copy shares the box, species and pair potentials, which never
        # change, and keeps the energy; the state that belongs to one run is
        # copied or reset below.
        new = copy.copy(self)
        new.rng = copy.deepcopy(self.rng)
        new.particles = self.particles.copy()
        new.particles["xunwrapped"] = new.particles["xposition"]
        new.particles["yunwrapped"] = new.particles["yposition"]
        new.initial_particles = new.particles.copy()
        new.distances = self.distances.copy()
        new.forces = self.forces.copy()
        new.energies = self.energies.copy()
        new.step = 0
        new.time = 0.0
        new.temperature_sample = np.array([])
        new.pressure_sample = np.array([])
        new.force_sample = np.array([])
        new.msd_sample = np.array([])
        new.energy_sample = np.array([])
        new.step_sample = np.array([])
        return new

    def square(self) -> None:
        """Place the particles on a square lattice.

        The lattice has ``ceil(sqrt(number_of_particles))`` sites along each
        side of the box and the particles fill it in order. No overlap check
        is made: a lattice too dense for the potential gives a large, or for
        a hard core infinite, initial energy on the first evaluation.
        """
        m = int(np.ceil(np.sqrt(self.number_of_particles)))
        d = self.box_length / m
        n = 0
        for i in range(0, m):
            for j in range(0, m):
                if n < self.number_of_particles:
                    self.particles[n]["xposition"] = (i + 0.5) * d
                    self.particles[n]["yposition"] = (j + 0.5) * d
                    n += 1

    def _place_by_metropolis(self) -> None:
        """Place the particles one at a time by Metropolis insertion.

        Each particle in turn is given a uniform trial position in the box,
        accepted by :func:`mc.accept` at ``placement_temperature`` on its
        total interaction energy with the particles already placed, and
        redrawn on rejection. Inserting from vacuum makes that energy the
        energy change of the insertion, so a trial that overlaps a core is
        rejected with probability close to one and one whose net interaction
        is attractive is always accepted.

        Sequential insertion is an initialiser, not an equilibrium sample:
        the placed configuration is free of hard overlaps, close contacts
        become likelier as the placement temperature rises, and the run
        equilibrates it.

        Raises:
            ValueError: If :data:`PLACEMENT_ATTEMPTS` trial positions are
                rejected for a single particle, or the pair potential returns
                NaN for a trial position.
        """
        x = self.particles["xposition"]
        y = self.particles["yposition"]
        types = self.particles["types"]
        for i in range(self.number_of_particles):
            placed = self.particles[:i]
            for _attempt in range(PLACEMENT_ATTEMPTS):
                trial = (self.rng.uniform(0, self.box_length), self.rng.uniform(0, self.box_length))
                energy = pairwise.particle_energy(
                    trial, int(types[i]), placed,
                    self.box_length, self.cut_off, self.pair_potentials, self.species,
                )
                if np.isnan(energy):
                    raise ValueError(
                        "The pair potential returned NaN for a trial position; check its "
                        "energies for a division by zero or an invalid parameter."
                    )
                if mc.accept(energy, self.placement_temperature, rng=self.rng):
                    x[i], y[i] = trial
                    break
            else:
                raise ValueError(
                    f"Could not place particle {i + 1} of {self.number_of_particles} in a "
                    f"{self.box_length * 1e10:.1f} Angstrom box at a placement temperature of "
                    f"{self.placement_temperature:g} K after {PLACEMENT_ATTEMPTS} attempts; "
                    "reduce the number of particles or use a larger box; for a soft "
                    "potential, raising placement_temperature tolerates closer contacts; "
                    "and check the units of the potential's parameters (metres and joules)."
                )

    def compute_force(self):
        """Compute the pair forces and the accelerations of the current
        configuration, storing the pair distances, forces and energies."""
        self.particles, self.distances, self.forces, self.energies = pairwise.compute_force(
            self.particles, self.box_length, self.cut_off, self.pair_potentials, self.species
        )

    def compute_energy(self):
        """Compute the pair energies of the current configuration, storing
        the pair distances and energies.

        Only the energies are evaluated, so the accelerations and stored
        forces are untouched and a potential with no finite force can be
        used.
        """
        self.distances, self.energies = pairwise.compute_energy(
            self.particles, self.box_length, self.cut_off, self.pair_potentials, self.species
        )

    def integrate(self, method):
        """Maps the chosen integration method.
        Parameters
        ----------
        method: method
            The integration method to be used, e.g. md.velocity_verlet.
        """
        self.particles, self.distances, self.forces, self.energies = method(
            self.particles,
            self.timestep_length,
            self.box_length,
            self.cut_off,
            self.pair_potentials,
            self.species,
        )

    def md_sample(self):
        """Maps to the md.sample function.
        """
        md.sample(self.particles, self.box_length, self.initial_particles, self)

    def heat_bath(self, bath_temperature: float) -> None:
        """Rescale the particle velocities to the bath temperature.

        Args:
            bath_temperature: The desired temperature, in kelvin.

        Raises:
            ValueError: If the bath temperature is not positive, or the
                particles are at rest or the simulation has diverged.
        """
        self.particles = md.heat_bath(self.particles, self.masses, bath_temperature)

    def mc_sample(self):
        """Record the current energy and step.

        The pair distances and energies are recomputed first, so the stored
        pair arrays match the configuration and ``energy`` is the exact
        total; between samples ``apply`` keeps a running total. A viewer
        updated between samples reads the pair arrays of the last sample.
        """
        self.compute_energy()
        self.energy = float(self.energies.sum())
        mc.sample(self.energy, self)

    def propose(self) -> mc.Proposal:
        """Propose a Monte Carlo move: one particle relocated at random.

        A particle is chosen at random and given a uniform trial position in
        the box. The energy change is that particle's interaction energy at
        the trial position minus that at its current position, with every
        other particle. The configuration is not changed.

        Returns:
            The proposed configuration and its energy change.
        """
        particle = int(self.rng.integers(self.number_of_particles))
        trial = (self.rng.uniform(0, self.box_length), self.rng.uniform(0, self.box_length))
        current = (self.particles["xposition"][particle], self.particles["yposition"][particle])
        species_index = int(self.particles["types"][particle])
        others = np.delete(self.particles, particle)
        trial_energy = pairwise.particle_energy(
            trial, species_index, others,
            self.box_length, self.cut_off, self.pair_potentials, self.species,
        )
        current_energy = pairwise.particle_energy(
            current, species_index, others,
            self.box_length, self.cut_off, self.pair_potentials, self.species,
        )
        energy_change = trial_energy - current_energy
        xposition = self.particles["xposition"].copy()
        yposition = self.particles["yposition"].copy()
        xposition[particle], yposition[particle] = trial
        return mc.Proposal(xposition, yposition, energy_change)

    def apply(self, proposal: mc.Proposal) -> None:
        """Make a proposed configuration the current one.

        The positions become the proposal's and ``energy`` gains its energy
        change, which is relative to the configuration the proposal was made
        from. The unwrapped positions are left as they are.

        Args:
            proposal: The proposal to apply, from :meth:`propose`.
        """
        self.particles["xposition"] = proposal.xposition
        self.particles["yposition"] = proposal.yposition
        self.energy += proposal.energy_change


def __cite__():  # pragma: no cover
    """This function will launch the website for the JOSE publication on
    pylj."""
    webbrowser.open("http://jose.theoj.org/papers/58daa1a1a564dc8e0f99ffcdae20eb1d")


def __version__():  # pragma: no cover
    """This will print the number of the pylj version currently in use."""
    major = 1
    minor = 4
    micro = 1
    print(f"pylj-{major:d}.{minor:d}.{micro:d}")


def particle_dt():
    """Builds the data type for the particles, this consists of:

    - xposition and yposition, wrapped into the simulation cell
    - xunwrapped and yunwrapped, the positions without periodic wrapping,
      advanced by md.velocity_verlet only and used for the mean squared
      displacement; Monte Carlo moves leave them unchanged
    - xvelocity and yvelocity
    - xacceleration and yacceleration
    - types, the index of each particle's species in ``System.species``
    """
    return np.dtype(
        [
            ("xposition", np.float64),
            ("yposition", np.float64),
            ("xunwrapped", np.float64),
            ("yunwrapped", np.float64),
            ("xvelocity", np.float64),
            ("yvelocity", np.float64),
            ("xacceleration", np.float64),
            ("yacceleration", np.float64),
            ("types", np.int64),
        ]
    )
