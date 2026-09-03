import webbrowser
from collections.abc import Iterable
from typing import Literal

import numpy as np

from pylj import mc, md


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
        Initial temperature of the particles, in Kelvin.
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
        metres as ``diameters``.
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
        self.setup_types()
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

    def square(self):
        """Places the particles on a square lattice."""
        m = int(np.ceil(np.sqrt(self.number_of_particles)))
        d = self.box_length / m
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
        at least the mean of the two particles' diameters (``self.diameters``,
        indexed by ``self.particles["types"]``).

        Raises:
            ValueError: If 1000 candidate positions are rejected for a
                single particle, which suggests the particles are too large,
                or too many, for the box. Use ``init_conf='square'`` or a
                larger box instead.
        """
        num_part = self.number_of_particles
        box_length = self.box_length
        types = self.particles["types"]
        max_attempts = 1000
        for i in range(num_part):
            type_i = int(types[i])
            for _attempt in range(max_attempts):
                x = np.random.uniform(0, box_length)
                y = np.random.uniform(0, box_length)
                placed = True
                for j in range(i):
                    type_j = int(types[j])
                    min_separation = (self.diameters[type_i] + self.diameters[type_j]) / 2
                    dx = x - self.particles[j]["xposition"]
                    dy = y - self.particles[j]["yposition"]
                    dx -= box_length * np.round(dx / box_length)
                    dy -= box_length * np.round(dy / box_length)
                    if np.sqrt(dx**2 + dy**2) < min_separation:
                        placed = False
                        break
                if placed:
                    self.particles[i]["xposition"] = x
                    self.particles[i]["yposition"] = y
                    break
            else:
                diameter_angstrom = max(self.diameters) * 1e10
                box_angstrom = box_length * 1e10
                raise ValueError(
                    f"Could not place {num_part} particles of diameter "
                    f"{diameter_angstrom:.2f} Angstrom without overlap in a "
                    f"{box_angstrom:.1f} Angstrom box after {max_attempts} "
                    "attempts; use init_conf='square' or a larger box."
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
                number of sets of constants, if any diameter is not positive,
                if any diameter looks like a value in metres rather than
                Angstrom, or if ``None`` is given and the forcefield has no
                ``diameter`` property.
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
            if value <= 0:
                raise ValueError(f"Every diameter must be positive, but got {value}")
            if value < 0.01:
                raise ValueError(
                    f"The diameter is in Angstrom, and {value} looks like a value in "
                    "metres. An Angstrom is 1e-10 metres."
                )
        self.diameters = [value * 1e-10 for value in values]

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
        """Maps to the compute_force function, as this also calculates energy
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
            bath_temperature: The desired temperature, in Kelvin.

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
            self.particles
        )

    def new_random_position(self):
        """Maps to the mc.get_new_particle function.
        """
        self.particles = mc.get_new_particle(
            self.particles, self.random_particle, self.box_length
        )

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
    
    - xposition and yposition
    - xvelocity and yvelocity
    - xacceleration and yacceleration
    - xprevious_position and yprevious_position
    - xforce and yforce
    - energy
    - types
    """
    return np.dtype(
        [
            ("xposition", np.float64),
            ("yposition", np.float64),
            ("xvelocity", np.float64),
            ("yvelocity", np.float64),
            ("xacceleration", np.float64),
            ("yacceleration", np.float64),
            ("xprevious_position", np.float64),
            ("yprevious_position", np.float64),
            ("energy", np.float64),
            ("xpbccount", int),
            ("ypbccount", int),
            ("types", list)
        ]
    )
