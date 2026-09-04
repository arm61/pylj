import numpy as np
from numpy.typing import NDArray

from pylj import pairwise
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN


def initialise(
    number_of_particles,
    temperature,
    box_length,
    init_conf,
    *,
    species,
    pair_potentials,
    timestep_length=1e-14,
    seed=None,
):
    """Initialise the particle positions (this can be either as a square or
    random arrangement) and velocities (based on the temperature defined), and
    calculate the initial forces/accelerations.

    Each velocity component is drawn from a normal (Gaussian) distribution
    with the thermal width for the particle's mass at the requested
    temperature, the centre-of-mass velocity is removed, and the result is
    rescaled so that the instantaneous temperature is exactly the requested
    one. Molecular dynamics needs at least two particles.

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
    species: sequence of Species
        Required, keyword only. The species in the system; particles are
        assigned to them in turn.
    pair_potentials: mapping of (Species, Species) to PairPotential
        Required, keyword only. The potential acting between each pair of
        species, including each species with itself.
    timestep_length: float (optional)
        Length for each Velocity-Verlet integration step, in seconds.
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
    if not (np.isfinite(temperature) and temperature > 0):
        raise ValueError(f"temperature must be positive and finite, not {temperature}")
    system = util.System(
        number_of_particles,
        temperature,
        box_length,
        species=species,
        pair_potentials=pair_potentials,
        simulation="md",
        init_conf=init_conf,
        timestep_length=timestep_length,
        seed=seed,
    )
    masses_kg = system.masses * ATOMIC_MASS_UNIT
    thermal_speed = np.sqrt(BOLTZMANN * temperature / masses_kg)
    v = system.rng.normal(0.0, thermal_speed[:, None], size=(number_of_particles, 2))
    v -= (masses_kg[:, None] * v).sum(axis=0) / masses_kg.sum()
    system.particles["xvelocity"] = v[:, 0]
    system.particles["yvelocity"] = v[:, 1]
    system.particles = heat_bath(system.particles, system.masses, temperature)
    system.compute_force()
    return system


initialize = initialise  # US spelling


def velocity_verlet(particles, timestep_length, box_length, cut_off, pair_potentials, species):
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
    pair_potentials: mapping of (Species, Species) to PairPotential
        The potential acting between each pair of species.
    species: sequence of Species
        The species, in the order the particles' ``types`` field indexes.

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
    particles, distances, forces, energies = pairwise.compute_force(
        particles, box_length, cut_off, pair_potentials, species
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
    temperature_new = calculate_temperature(particles, system.masses)
    system.temperature_sample = np.append(system.temperature_sample, temperature_new)
    pressure_new = pairwise.calculate_pressure(
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
    mass: float or float, array_like
        The mass of the particles, in atomic mass units: one value per
        particle, or a single value for all of them.

    Returns
    -------
    float:
        Calculated instantaneous simulation temperature, in kelvin.

    Raises
    ------
    ValueError
        If there are fewer than two particles.
    """
    if particles.size < 2:
        raise ValueError(
            "The temperature needs at least two particles: with one particle there "
            "is no thermal motion once the centre-of-mass velocity is removed."
        )
    mass_kg = np.asarray(mass, dtype=float) * ATOMIC_MASS_UNIT
    kinetic = 0.5 * np.sum(
        mass_kg
        * (
            particles["xvelocity"] * particles["xvelocity"]
            + particles["yvelocity"] * particles["yvelocity"]
        )
    )
    return kinetic / ((particles.size - 1) * BOLTZMANN)


def heat_bath(
    particles: np.ndarray, mass: float | NDArray[np.float64], bath_temperature: float
) -> np.ndarray:
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
        mass: The mass of the particles, in atomic mass units: one value per
            particle, or a single value for all of them.
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
