import copy
import webbrowser
from collections.abc import Iterable
from typing import Literal, Self

import numpy as np

from pylj import mc, md

#: Number of rejection-sampling attempts made for a single particle in
#: :meth:`System.random` before it gives up and raises ``ValueError``.
PLACEMENT_ATTEMPTS = 1000


class System:
    """Simulation system.
    This class is designed to store all of the information about the job that
    is being run. This includes the particles
    object, as will as sampling objects such as the temperature, pressure, etc.
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
    constants: float, array_like
        The values of the constants for the forcefield used, one
        set per particle type.
    forcefield: class
        The particular forcefield to be used to find the energy and
        forces.
    mass: float
        The mass of the particles being simulated.
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
        The distance apart that the particles must be to consider there
        interaction to be negliable.
    diameter: float or iterable of float (optional)
        Drawn diameter of the particles in Angstrom, one value or one per
        set of constants. Each value must be positive, and at least 0.01, as
        values below that are metres mistaken for Angstrom. Defaults to the
        separation at the pair-potential minimum of the forcefield, which the
        forcefield must then provide as its ``diameter`` property. Stored in
        metres as ``diameters``. This sets how the particles are drawn;
        placement uses ``cores``.
    seed: int (optional)
        Seed for the random number generator used to place a random initial
        configuration, draw the initial velocities and make Monte Carlo
        moves. The same seed reproduces the same run. Without one the run
        differs each time.

    Attributes
    ----------
    diameters: list of float
        Drawn diameter of each particle type, in metres.
    cores: list of float
        Separation at which each type's pair energy falls to zero, in
        metres. Particles of one type are placed at least this far apart,
        and particles of two types at least the mean of their cores apart.
    rng: numpy.random.Generator
        The random number generator for this system.
    """

    def __init__(
        self,
        number_of_particles: int,
        temperature: float,
        box_length: float,
        constants: list[list[float]],
        forcefield: type,
        mass: float,
        *,
        simulation: Literal["md", "mc"],
        init_conf: str = "square",
        timestep_length: float = 1e-14,
        cut_off: float = 15,
        diameter: float | Iterable[float] | None = None,
        seed: int | None = None,
    ):
        if simulation not in ("md", "mc"):
            raise ValueError(f"simulation must be 'md' or 'mc', not {simulation!r}")
        self.simulation = simulation
        self.number_of_particles = number_of_particles
        self.init_temp = temperature
        self.constants = constants
        self.mass = mass
        self.forcefield = forcefield
        self.particle_list = None
        self.long_const = None
        self.types = None
        self.diameters: list[float] = []
        self.setup_diameters(diameter)
        self.cores: list[float] = []
        self.setup_cores()
        self.setup_types()
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
        self.particles["types"] = self.types
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

        The new system keeps the box, forcefield, constants, mass, timestep
        and cut-off, copies the state of the random number generator so the
        new system's draws do not depend on what the current system does
        next, and copies the particle positions, velocities
        and accelerations and the pair distances, forces and energies. A
        Monte Carlo system also keeps its accepted energy. Step and time are
        zero, the sample arrays are empty, and initial_particles is replaced
        by the copied particles, so the mean squared displacement is measured
        from the restarted configuration. The current system is not changed.
        Use it to start a production run after equilibration::

            system = md.initialise(100, 300, 40, "random")
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
        # A shallow copy shares the box, forcefield and constants, which never
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
        set of constants. Two particles of different types are kept at least
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
        thresholds = (cores[type_i] + cores[[int(t) for t in types[:index]]]) / 2
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

    def setup_types(self):
        """Sets the long constants and types arrays of the particles
        """
        long_const = []
        types = []
        particle_list = []
        for i in range(self.number_of_particles):
            # Get set of constants and index
            constants_type = self.constants[i%len(self.constants)]
            particle_type = f'{i%len(self.constants)}'
            # Append to lists
            long_const.append(constants_type)
            types.append(particle_type)
            particle = Particle(constants_type, i, self.mass, particle_type)
            particle.add(particle_list)
        self.particle_list = particle_list
        self.long_const = long_const
        self.types = types

    def setup_diameters(self, diameter: float | Iterable[float] | None) -> None:
        """Set the drawn diameter of each particle type, in metres.

        Args:
            diameter: Diameter in Angstrom. ``None`` takes the separation at
                the pair-potential minimum from the forcefield for each set
                of constants. A single number applies to every type; an
                iterable gives one value per set of constants.

        Raises:
            ValueError: If an iterable is given whose length differs from the
                number of sets of constants, if any diameter is not finite
                or not positive, if any diameter looks like a value in
                metres rather than Angstrom, or if ``None`` is given and the
                forcefield has no ``diameter`` property.
        """
        if diameter is None:
            self.diameters = []
            for c in self.constants:
                forcefield = self.forcefield(c)
                try:
                    value = forcefield.diameter
                except AttributeError as error:
                    raise ValueError(
                        f"{type(forcefield).__name__} has no diameter property. A "
                        "forcefield must provide a diameter property giving the "
                        "separation at the pair-potential minimum in metres, or the "
                        "caller must pass diameter= to initialise. See the bring "
                        "your own forcefield documentation."
                    ) from error
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"{type(forcefield).__name__}.diameter must be a positive, "
                        f"finite value in metres, but got {value}"
                    )
                self.diameters.append(value)
            return
        if isinstance(diameter, Iterable):
            values = [float(d) for d in diameter]
        else:
            values = [float(diameter)] * len(self.constants)
        if len(values) != len(self.constants):
            raise ValueError(
                f"Expected {len(self.constants)} diameters, one per set of "
                f"constants, but got {len(values)}"
            )
        for value in values:
            if not np.isfinite(value):
                raise ValueError(f"Every diameter must be finite, but got {value}")
            if value <= 0:
                raise ValueError(f"Every diameter must be positive, but got {value}")
            if value < 0.01:
                raise ValueError(
                    f"The diameter is in Angstrom, and {value} looks like a value in "
                    "metres. An Angstrom is 1e-10 metres."
                )
        self.diameters = [value * 1e-10 for value in values]

    def setup_cores(self) -> None:
        """Set the separation at which each forcefield's pair energy falls
        to zero, in metres, one per set of constants. Particles closer than
        this sit inside each other's repulsive core, so the initial
        configurations keep them at least this far apart.

        Raises:
            ValueError: If a forcefield's ``energy`` does not return one
                value per separation, if its pair energy is positive at
                every grid point between 0.1 and 50 Angstrom (suggesting its
                constants are in the wrong units), or if its pair energy
                never falls from positive to non-positive on that range, so
                it has no repulsive core there.
        """
        r = np.logspace(-11, np.log10(5e-9), 4000)
        self.cores = []
        for c in self.constants:
            forcefield = self.forcefield(c)
            energy = np.asarray(forcefield.energy(r), dtype=float)
            if energy.shape != r.shape:
                raise ValueError(
                    f"{type(forcefield).__name__}.energy must return one value per "
                    f"separation; got shape {energy.shape} for {r.shape[0]} separations"
                )
            positive = energy > 0
            crossings = np.flatnonzero(positive[:-1] & ~positive[1:])
            if crossings.size == 0:
                if positive.all():
                    raise ValueError(
                        f"{type(forcefield).__name__}: the pair energy is still "
                        "positive at 50 Angstrom; check the units of the constants"
                    )
                raise ValueError(
                    f"{type(forcefield).__name__} has no repulsive core: its pair "
                    "energy never falls from positive to zero between 0.1 and 50 "
                    "Angstrom"
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
        """Maps to the md.compute_force function, storing what it returns."""
        part, dist, forces, energies = md.compute_force(
            self.particles,
            self.box_length,
            self.cut_off,
            self.constants,
            self.forcefield,
            self.mass,
        )
        self.particles = part
        self.distances = dist
        self.forces = forces
        self.energies = energies

    def compute_energy(self):
        """Compute the pair energies of the current configuration.

        The forces, distances and accelerations are updated at the same time.
        """
        self.compute_force()

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
            self.constants,
            self.forcefield,
            self.mass
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
        self.particles = md.heat_bath(self.particles, self.mass, bath_temperature)

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

class Particle:
    def __init__(self,
               constants,
               index,
               mass,
               particle_type
               ):
        self.constants = constants
        self.index = index
        self.mass = mass
        self.type = particle_type

    def add(self, particles):
        particles.append(self)
        return particles

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
    - types
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
            ("types", list),
        ]
    )
