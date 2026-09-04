import numpy as np

from pylj import forcefields as ff
from pylj import pairwise as heavy
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN


def initialise(
    number_of_particles,
    temperature,
    box_length,
    init_conf,
    timestep_length=1e-14,
    mass=39.948,
    constants=None,
    forcefield=ff.lennard_jones,
    diameter=None,
    seed=None,
):
    """Initialise the particle positions (this can be either as a square or
    random arrangement) and velocities (based on the temperature defined), and
    calculate the initial forces/accelerations.

    The velocities are drawn from the Maxwell-Boltzmann distribution at the
    requested temperature, the centre-of-mass velocity is removed, and the
    result is rescaled so that the instantaneous temperature is exactly the
    requested one. Molecular dynamics needs at least two particles.

    Parameters
    ----------
    number_of_particles: int
        Number of particles to simulate.
    temperature: float
        Initial temperature of the particles, in kelvin.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    init_conf: string
        The way that the particles are initially positioned. Should be one of:
        - 'square'
        - 'random'
        Both raise ``ValueError`` if the particles cannot be placed without
        their repulsive cores overlapping.
    timestep_length: float (optional)
        Length for each Velocity-Verlet integration step, in seconds.
    mass: float (optional)
        The mass of the particles being simulated.
    constants: float, array_like (optional)
        The values of the constants for the forcefield used. Defaults to the
        argon Lennard-Jones constants, ``[[1.363e-134, 9.273e-78]]``.
    forcefield: class (optional)
        The particular forcefield to be used to find the energy and forces.
    diameter: float or iterable of float (optional)
        Drawn diameter of the particles in Angstrom, one value or one per
        set of constants. Defaults to the separation at the pair-potential
        minimum of the forcefield.
    seed: int (optional)
        Seed for the random number generator used to place a random initial
        configuration and draw the initial velocities. The same seed
        reproduces the same run.

    Returns
    -------
    System
        System information.

    Raises
    ------
    ValueError
        If fewer than two particles are requested, or the temperature is not
        positive.
    """
    from pylj import util

    if number_of_particles < 2:
        raise ValueError(
            "Molecular dynamics needs at least two particles: with one particle "
            "there is no thermal motion once the centre-of-mass velocity is removed."
        )
    if not temperature > 0:
        raise ValueError(f"temperature must be positive, not {temperature}")
    if constants is None:
        constants = [[1.363e-134, 9.273e-78]]
    system = util.System(
        number_of_particles,
        temperature,
        box_length,
        constants,
        forcefield,
        mass,
        simulation="md",
        init_conf=init_conf,
        timestep_length=timestep_length,
        diameter=diameter,
        seed=seed,
    )
    mass_kg = mass * ATOMIC_MASS_UNIT
    thermal_speed = np.sqrt(BOLTZMANN * temperature / mass_kg)
    v = system.rng.normal(0.0, thermal_speed, size=(number_of_particles, 2))
    v -= v.mean(axis=0)
    system.particles["xvelocity"] = v[:, 0]
    system.particles["yvelocity"] = v[:, 1]
    system.particles = heat_bath(system.particles, mass, temperature)
    system.compute_force()
    return system


initialize = initialise  # US spelling


def velocity_verlet(
    particles, timestep_length, box_length, cut_off, constants, forcefield, mass
):
    """Move the particles forward one step with the Velocity-Verlet
    integrator: update the positions, recompute the forces, then update the
    velocities from the mean of the old and new accelerations.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    timestep_length: float
        Length for each Velocity-Verlet integration step, in seconds.
    box_length: float
        Length of a single dimension of the simulation square, in metres.
    cut_off: float
        The separation beyond which the pair energy and force are taken to be
        zero, in metres.
    constants: float, array_like
        The constants associated with the particular forcefield used.
    forcefield: class
        The particular forcefield to be used to find the energy and forces.
    mass: float
        The mass of the particle being simulated (units of atomic mass units).

    Returns
    -------
    util.particle_dt, array_like:
        Information about the particles, with new positions, velocities and
        accelerations.
    float, array_like
        Current distances between pairs of particles in the simulation.
    float, array_like
        Current forces between pairs of particles in the simulation.
    float, array_like
        Current energies between pairs of particles in the simulation.
    """
    positions, unwrapped = update_positions(
        [particles["xposition"], particles["yposition"]],
        [particles["xunwrapped"], particles["yunwrapped"]],
        [particles["xvelocity"], particles["yvelocity"]],
        [particles["xacceleration"], particles["yacceleration"]],
        timestep_length,
        box_length,
    )
    [particles["xposition"], particles["yposition"]] = positions
    [particles["xunwrapped"], particles["yunwrapped"]] = unwrapped
    xacceleration_store = list(particles["xacceleration"])
    yacceleration_store = list(particles["yacceleration"])
    particles, distances, forces, energies = heavy.compute_force(
        particles, box_length, cut_off, constants, forcefield, mass
    )
    [particles["xvelocity"], particles["yvelocity"]] = update_velocities(
        [particles["xvelocity"], particles["yvelocity"]],
        [xacceleration_store, yacceleration_store],
        [particles["xacceleration"], particles["yacceleration"]],
        timestep_length,
    )
    return particles, distances, forces, energies


def sample(particles, box_length, initial_particles, system):
    """Sample parameters of interest in the simulation.

    The pressure is calculated from the pair distances and forces stored on
    the system by the last force evaluation, not from the current particle
    positions.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in metres.
    initial_particles: util.particle_dt, array-like
        Information about the initial particle conformation.
    system: System
        Details about the whole system

    Returns
    -------
    System:
        Details about the whole system, with the new step, temperature,
        pressure, energy, msd, and force appended to the appropriate
        arrays.
    """
    temperature_new = calculate_temperature(particles, system.mass)
    system.temperature_sample = np.append(system.temperature_sample, temperature_new)
    pressure_new = heavy.calculate_pressure(
        system.distances,
        system.forces,
        box_length,
        particles.size,
        temperature_new,
    )
    msd_new = calculate_msd(particles, initial_particles)
    system.pressure_sample = np.append(system.pressure_sample, pressure_new)
    system.force_sample = np.append(system.force_sample, np.sum(system.forces))
    system.energy_sample = np.append(system.energy_sample, np.sum(system.energies))
    system.msd_sample = np.append(system.msd_sample, msd_new)
    system.step_sample = np.append(system.step_sample, system.step)
    return system


def calculate_msd(particles, initial_particles):
    """Determines the mean squared displacement of the particles from their
    positions in initial_particles, using the unwrapped positions so that
    crossings of the periodic boundary are included. The unwrapped positions
    are maintained by md.velocity_verlet only, so the displacement is
    meaningful for a molecular dynamics system and not after Monte Carlo
    moves.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    initial_particles: util.particle_dt, array_like
        Information about the particles at the origin of the displacement.

    Returns
    -------
    float:
        Mean squared displacement of the particles, in metres squared.
    """
    dx = particles["xunwrapped"] - initial_particles["xunwrapped"]
    dy = particles["yunwrapped"] - initial_particles["yunwrapped"]
    return np.mean(dx * dx + dy * dy)


def update_positions(
    positions, unwrapped, velocities, accelerations, timestep_length, box_length
):
    """Update the particle positions using the Velocity-Verlet integrator.

    Parameters
    ----------
    positions: (2, N) array_like
        Where N is the number of particles, and the first row are the x
        positions and the second row the y positions, wrapped into the
        simulation cell.
    unwrapped: (2, N) array_like
        The same positions without periodic wrapping.
    velocities: (2, N) array_like
        Where N is the number of particles, and the first row are the x
        velocities and the second row the y velocities.
    accelerations: (2, N) array_like
        Where N is the number of particles, and the first row are the x
        accelerations and the second row the y accelerations.
    timestep_length: float
        Length for each Velocity-Verlet integration step, in seconds.
    box_length: float
        Length of a single dimension of the simulation square, in metres.

    Returns
    -------
    (2, N) array_like:
        Updated positions, wrapped into the simulation cell.
    (2, N) array_like:
        Updated unwrapped positions.
    """
    for axis in (0, 1):
        displacement = velocities[axis] * timestep_length + (
            0.5 * accelerations[axis] * timestep_length * timestep_length
        )
        positions[axis] = (positions[axis] + displacement) % box_length
        unwrapped[axis] = unwrapped[axis] + displacement
    return [positions[0], positions[1]], [unwrapped[0], unwrapped[1]]


def update_velocities(
    velocities, accelerations_old, accelerations_new, timestep_length
):
    """Update the particle velocities using the Velocity-Verlet algoritm.

    Parameters
    ----------
    velocities: (2, N) array_like
        Where N is the number of particles, and the first row are the x
        velocities and the second row the y velocities.
    accelerations: (2, N) array_like
        Where N is the number of particles, and the first row are the x
        accelerations and the second row the y
        accelerations.
    timestep_length: float
        Length for each Velocity-Verlet integration step, in seconds.

    Returns
    -------
    (2, N) array_like:
        Updated velocities.
    """
    velocities[0] += (
        0.5 * (accelerations_old[0] + accelerations_new[0]) * timestep_length
    )
    velocities[1] += (
        0.5 * (accelerations_old[1] + accelerations_new[1]) * timestep_length
    )
    return [velocities[0], velocities[1]]


def calculate_temperature(particles, mass):
    """Determine the instantaneous temperature of the system.

    The centre-of-mass velocity is zero at initialisation and conserved by
    the pair forces, so 2N - 2 velocity components carry thermal energy and
    the temperature is the kinetic energy divided by (N - 1) k_B.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    mass: float
        The mass of the particles being simulated, in atomic mass units.

    Returns
    -------
    float:
        Calculated instantaneous simulation temperature, in kelvin.
    """
    mass_kg = mass * ATOMIC_MASS_UNIT
    kinetic = 0.5 * mass_kg * np.sum(
        particles["xvelocity"] * particles["xvelocity"]
        + particles["yvelocity"] * particles["yvelocity"]
    )
    return kinetic / ((particles.size - 1) * BOLTZMANN)


def compute_force(particles, box_length, cut_off, constants, forcefield, mass):
    r"""Calculates the forces and therefore the accelerations on each of the
    particles in the simulation.

    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in metres.
    cut_off: float
        The distance beyond which the force between two particles is taken
        to be zero.
    constants: float, array_like (optional)
        The constants associated with the particular forcefield used, e.g. for
        the function forcefields.lennard_jones, theses are [A, B]
    forcefield: class (optional)
        The particular forcefield to be used to find the energy and forces.
    mass: float (optional)
        The mass of the particle being simulated (units of atomic mass units).

    Returns
    -------
    util.particle_dt, array_like
        Information about particles, with updated accelerations and forces.
    float, array_like
        Current distances between pairs of particles in the simulation.
    float, array_like
        Current forces between pairs of particles in the simulation.
    float, array_like
        Current energies between pairs of particles in the simulation.
    """
    part, dist, forces, energies = heavy.compute_force(
        particles, box_length, cut_off, constants, forcefield, mass=mass
    )
    return part, dist, forces, energies


def heat_bath(particles: np.ndarray, mass: float, bath_temperature: float) -> np.ndarray:
    r"""Rescale the velocities so the instantaneous temperature equals the
    bath temperature.

    This is a velocity-rescaling thermostat: each call sets the
    instantaneous temperature to the bath temperature. The velocities are
    rescaled according to

    .. math::
        v_{\text{new}} = v_{\text{old}} \times
        \sqrt{\frac{T_{\text{bath}}}{T_{\text{now}}}}

    where :math:`T_{\text{now}}` is the temperature of the current
    velocities.

    Args:
        particles: Information about the particles.
        mass: The mass of the particles being simulated, in atomic mass
            units.
        bath_temperature: The desired temperature of the simulation, in
            kelvin.

    Returns:
        The particles with velocities rescaled in place; the same array is
        returned.

    Raises:
        ValueError: If bath_temperature is not positive.
        ValueError: If the current temperature is zero (the particles are at
            rest, as in a Monte Carlo system) or not finite (the simulation
            has diverged).
    """
    if not bath_temperature > 0:
        raise ValueError(
            f"bath_temperature must be positive, not {bath_temperature}"
        )
    current_temperature = calculate_temperature(particles, mass)
    if current_temperature == 0:
        raise ValueError(
            "Cannot rescale velocities: the particles are at rest. A Monte Carlo "
            "system has no velocities to thermostat; use md.initialise for MD."
        )
    if not (np.isfinite(current_temperature) and current_temperature > 0):
        raise ValueError(
            "Cannot rescale velocities: the current temperature is "
            f"{current_temperature}, so the simulation has diverged."
        )
    scale = np.sqrt(bath_temperature / current_temperature)
    particles["xvelocity"] = particles["xvelocity"] * scale
    particles["yvelocity"] = particles["yvelocity"] * scale
    return particles
