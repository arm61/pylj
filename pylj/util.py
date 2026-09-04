import copy
import itertools
import webbrowser
from collections.abc import Sequence
from typing import Literal, Self

import numpy as np

from pylj import mc, md, pairwise
from pylj.pairwise import PairPotentials
from pylj.potentials import PairPotential, Species

#: Number of rejection-sampling attempts made for a single particle in
#: :meth:`System.random` before it gives up and raises ``ValueError``.
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
        Initial temperature of the particles, in kelvin.
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
        - 'square'
        - 'random'
        Both raise ``ValueError`` if the particles cannot be placed without
        their repulsive cores overlapping.
    timestep_length: float (optional)
        Length for each Velocity-Verlet integration step, in seconds.
    cut_off: float (optional)
        The distance apart that the particles must be to consider their
        interaction to be negligible.
    seed: int (optional)
        Seed for the random number generator used to place a random initial
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
    cores: list of float
        Separation at which each species' pair energy falls to zero, in
        metres. Particles of one species are placed at least this far apart,
        and particles of two species at least the mean of their cores apart.
    rng: numpy.random.Generator
        The random number generator for this system.

    Raises
    ------
    ValueError
        If the temperature is not positive and finite, ``species`` is empty,
        or a pair of species has no potential or one given in both orders.
    TypeError
        If a pair potential is not a ``PairPotential`` instance.
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
        seed: int | None = None,
    ):
        if simulation not in ("md", "mc"):
            raise ValueError(f"simulation must be 'md' or 'mc', not {simulation!r}")
        self.simulation = simulation
        self.number_of_particles = number_of_particles
        if not (np.isfinite(temperature) and temperature > 0):
            raise ValueError(f"temperature must be positive and finite, not {temperature}")
        self.init_temp = temperature
        self.species = list(species)
        self.pair_potentials = dict(pair_potentials)
        _check_pair_potentials(self.species, self.pair_potentials)
        self.cores: list[float] = []
        self.setup_cores()
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
        if init_conf == "square":
            self.square()
        elif init_conf == "random":
            self.random()
        else:
            raise NotImplementedError(
                f"The initial configuration type {init_conf} is "
                "not recognised. Available options are: "
                "square or random"
            )
        self.particles["xunwrapped"] = self.particles["xposition"]
        self.particles["yunwrapped"] = self.particles["yposition"]
        if box_length > 30:
            self.cut_off = cut_off * 1e-10
        else:
            self.cut_off = box_length / 2 * 1e-10
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
        self.position_store = [0, 0]
        self.old_energy = 0
        self.new_energy = 0
        self.random_particle = 0

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
        Monte Carlo system also keeps its accepted energy. Step and time are
        zero, the sample arrays are empty, and initial_particles is replaced
        by the copied particles, so the mean squared displacement is measured
        from the restarted configuration. The current system is not changed.
        Use it to start a production run after equilibration::

            argon = Species(mass=39.948, name="argon")
            lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
            system = md.initialise(
                100, 300, 40, "random",
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
        # change, and keeps the accepted energy; the state that belongs to one
        # run is copied or reset below.
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
        new.new_energy = 0
        new.position_store = [0, 0]
        new.random_particle = 0
        return new

    def square(self) -> None:
        """Places the particles on a square lattice.

        Raises:
            ValueError: If the lattice spacing, ``box_length`` divided by
                ``ceil(sqrt(number_of_particles))``, is less than the
                largest repulsive core (``self.cores``). Reduce the number
                of particles or use a larger box.
        """
        m = int(np.ceil(np.sqrt(self.number_of_particles)))
        d = self.box_length / m
        core = max(self.cores)
        if d < core:
            n_max = int(np.floor(self.box_length / core)) ** 2
            l_min = np.ceil(np.sqrt(self.number_of_particles)) * core
            l_min_angstrom = np.ceil(l_min * 1e10 * 10) / 10
            fits = "1 particle fits" if n_max == 1 else f"{n_max} particles fit"
            raise ValueError(
                f"A square lattice of {self.number_of_particles} particles in a "
                f"{self.box_length * 1e10:.1f} Angstrom box spaces them {d * 1e10:.2f} "
                f"Angstrom apart, less than the largest repulsive core of "
                f"{core * 1e10:.2f} Angstrom; at most {fits} in this "
                f"box, or a box of at least {l_min_angstrom:.1f} Angstrom fits "
                f"{self.number_of_particles}."
            )
        n = 0
        for i in range(0, m):
            for j in range(0, m):
                if n < self.number_of_particles:
                    self.particles[n]["xposition"] = (i + 0.5) * d
                    self.particles[n]["yposition"] = (j + 0.5) * d
                    n += 1

    def random(self) -> None:
        """Places the particles at random positions, without overlap.

        Particles are placed one at a time by rejection sampling: a
        candidate position for a particle is accepted only if, for every
        already placed particle, the minimum-image distance between them is
        at least the two particles' repulsive core, ``self.cores``, one per
        species. Two particles of different species are kept at least
        the mean of their two cores apart.

        Rejection sampling reaches area fractions of roughly 0.4 to 0.5;
        near that limit the same call may succeed or raise depending on
        the random draw.

        Raises:
            ValueError: If :data:`PLACEMENT_ATTEMPTS` candidate positions are
                rejected for a single particle, which suggests the particles
                are too large, or too many, for the box. Reduce the number
                of particles or use a larger box.
        """
        x = self.particles["xposition"]
        y = self.particles["yposition"]
        for i in range(self.number_of_particles):
            x[i], y[i] = self._place_particle(i, x, y)

    def _place_particle(
        self, index: int, placed_x: np.ndarray, placed_y: np.ndarray
    ) -> tuple[float, float]:
        """Find a non-overlapping position for one particle by rejection sampling.

        Args:
            index: Index of the particle being placed into
                ``self.particles["types"]``.
            placed_x: x positions of the particles already placed, indexed
                0 to ``index - 1``.
            placed_y: y positions of the particles already placed, indexed
                0 to ``index - 1``.

        Returns:
            An (x, y) position, in metres, at least the mean of the two
            particles' repulsive cores from every already-placed particle.

        Raises:
            ValueError: If :data:`PLACEMENT_ATTEMPTS` candidate positions are
                rejected.
        """
        box_length = self.box_length
        types = self.particles["types"]
        type_i = int(types[index])
        cores = np.asarray(self.cores)
        thresholds = (cores[type_i] + cores[types[:index]]) / 2
        for _attempt in range(PLACEMENT_ATTEMPTS):
            x = self.rng.uniform(0, box_length)
            y = self.rng.uniform(0, box_length)
            dx = x - placed_x[:index]
            dy = y - placed_y[:index]
            # minimum-image convention: wrap the separation to the
            # nearest periodic copy of each already-placed particle
            dx -= box_length * np.round(dx / box_length)
            dy -= box_length * np.round(dy / box_length)
            separations = np.sqrt(dx**2 + dy**2)
            if np.all(separations >= thresholds):
                return x, y
        core_angstrom = self.cores[type_i] * 1e10
        box_angstrom = box_length * 1e10
        largest_core_angstrom = max(self.cores) * 1e10
        area_fraction = (
            sum(np.pi * (self.cores[int(t)] / 2) ** 2 for t in self.particles["types"])
            / box_length**2
        )
        raise ValueError(
            f"Could not place particle {index + 1} of {self.number_of_particles} "
            f"(repulsive core {core_angstrom:.2f} Angstrom, largest "
            f"{largest_core_angstrom:.2f} Angstrom) without overlap in a "
            f"{box_angstrom:.1f} Angstrom box after {PLACEMENT_ATTEMPTS} attempts "
            f"(requested area fraction {area_fraction:.2f}); reduce the number of particles or "
            "use a larger box; a square lattice (init_conf='square') packs more "
            "densely than random placement and may still fit."
        )

    def setup_cores(self) -> None:
        """Set the separation at which each species' pair energy falls to
        zero, in metres, one per species. Particles closer than this sit
        inside each other's repulsive core, so the initial configurations
        keep them at least this far apart.

        Raises:
            ValueError: If a potential's ``energies`` does not return one
                value per separation, if its pair energy is positive at
                every grid point between 0.1 and 50 Angstrom (suggesting its
                parameters are in the wrong units), or if its pair energy
                never falls from positive to non-positive on that range, so
                it has no repulsive core there.
        """
        r = np.logspace(-11, np.log10(5e-9), 4000)
        self.cores = []
        for one in self.species:
            potential = pairwise.pair_potential(self.pair_potentials, one, one)
            name = type(potential).__name__
            energy = np.asarray(potential.energies(r), dtype=float)
            if energy.shape != r.shape:
                raise ValueError(
                    f"{name}.energies must return one value per separation; got "
                    f"shape {energy.shape} for {r.shape[0]} separations"
                )
            positive = energy > 0
            crossings = np.flatnonzero(positive[:-1] & ~positive[1:])
            if crossings.size == 0:
                if positive.all():
                    raise ValueError(
                        f"{name}: the pair energy is still positive at 50 Angstrom; "
                        "check the units of the parameters"
                    )
                raise ValueError(
                    f"{name} has no repulsive core: its pair energy never falls from "
                    "positive to zero between 0.1 and 50 Angstrom"
                )
            i = int(crossings[-1])
            r0, r1 = r[i], r[i + 1]
            e0, e1 = energy[i], energy[i + 1]
            if np.isfinite(e0) and np.isfinite(e1):
                # linear interpolation to the point where the energy is zero
                core = r0 + (r1 - r0) * (-e0) / (e1 - e0)
            else:
                # a discontinuous step (e.g. an infinite repulsive wall): the
                # first non-positive grid point is a safe, slightly
                # conservative estimate
                core = r1
            self.cores.append(float(core))

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
        """Maps to the mc.sample function, recording the current accepted energy."""
        mc.sample(self.old_energy, self)

    def select_random_particle(self):
        """Maps to the mc.select_random_particle function.
        """
        self.random_particle, self.position_store = mc.select_random_particle(
            self.particles, self.rng
        )

    def new_random_position(self):
        """Maps to the mc.get_new_particle function.
        """
        self.particles = mc.get_new_particle(
            self.particles, self.random_particle, self.box_length, self.rng
        )

    def metropolis(self) -> bool:
        """Decide whether to accept the current trial move.

        Applies the Metropolis condition at the system's temperature to the
        stored energies before and after the move, drawing from the system's
        random number generator.

        Returns:
            True if the move should be accepted.
        """
        return mc.metropolis(self.init_temp, self.old_energy, self.new_energy, rng=self.rng)

    def accept(self):
        """Maps to the mc.accept function.
        """
        self.old_energy = mc.accept(self.new_energy)

    def reject(self):
        """Maps to the mc.reject function.
        """
        self.particles = mc.reject(
            self.position_store, self.particles, self.random_particle
        )


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
    - energy
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
            ("energy", np.float64),
            ("types", np.int64),
        ]
    )
