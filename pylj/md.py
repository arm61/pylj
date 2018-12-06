import numpy as np
from pylj import pairwise as heavy
from pylj import forcefields as ff


def initialise(number_of_particles, temperature, box_length, init_conf,
               timestep_length=1e-14, mass=39.948,
               constants=[1.363e-134, 9.273e-78], forcefield=ff.lennard_jones):
    """Initialise the particle positions (this can be either as a square or
    random arrangement), velocities (based on the temperature defined, and
    calculate the initial forces/accelerations.

    Parameters
    ----------
    number_of_particles: int
        Number of particles to simulate.
    temperature: float
        Initial temperature of the particles, in Kelvin.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    init_conf: string, optional
        The way that the particles are initially positioned. Should be one of:
        - 'square'
        - 'random'
    timestep_length: float (optional)
        Length for each Velocity-Verlet integration step, in seconds.
    mass: float (optional)
        The mass of the particles being simulated.
    constants: float, array_like (optional)
        The values of the constants for the forcefield used.
    forcefield: function (optional)
        The particular forcefield to be used to find the energy and forces.

    Returns
    -------
    System
        System information.
    """
    from pylj import util
    system = util.System(number_of_particles, temperature, box_length,
                         constants, forcefield, mass,
                         init_conf=init_conf,
                         timestep_length=timestep_length)
    v = np.random.rand(system.particles.size, 2, 12)
    v = np.sum(v, axis=2) - 6.
    mass_kg = mass * 1.6605e-27
    v = v * np.sqrt(1.3806e-23 * system.init_temp / mass_kg)
    v = v - np.average(v)
    system.particles['xvelocity'] = v[:, 0]
    system.particles['yvelocity'] = v[:, 1]
    return system


def initialize(number_particles, temperature, box_length, init_conf,
               timestep_length=1e-14):
    """Maps to the md.initialise function to account for US english spelling.
    """
    a = initialise(number_particles, temperature, box_length, init_conf,
                   timestep_length)
    return a


def velocity_verlet(particles, timestep_length, box_length,
                    cut_off, constants, forcefield, mass):
    """Uses the Velocity-Verlet integrator to move forward in time. The
    Updates the particles positions and velocities in terms of the Velocity
    Verlet algorithm. Also calculates the instanteous temperature, pressure,
    and force and appends these to the appropriate system array.
    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    timestep_length: float
        Length for each Velocity-Verlet integration step, in seconds.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    Returns
    -------
    util.particle_dt, array_like:
        Information about the particles, with new positions and velocities.
    """
    xposition_store = list(particles['xposition'])
    yposition_store = list(particles['yposition'])
    pos, prev_pos = update_positions([particles['xposition'],
                                      particles['yposition']],
                                     [particles['xprevious_position'],
                                      particles['yprevious_position']],
                                     [particles['xvelocity'],
                                      particles['yvelocity']],
                                     [particles['xacceleration'],
                                      particles['yacceleration']],
                                     timestep_length, box_length)
    [particles['xposition'], particles['yposition']] = pos
    [particles['xprevious_position'], particles['yprevious_position']] = pos
    xacceleration_store = list(particles['xacceleration'])
    yacceleration_store = list(particles['yacceleration'])
    particles, distances, forces, energies = heavy.compute_force(
        particles, box_length, cut_off, constants, forcefield, mass)
    [particles['xvelocity'], particles['yvelocity']] = update_velocities(
        [particles['xvelocity'], particles['yvelocity']],
        [xacceleration_store, yacceleration_store],
        [particles['xacceleration'], particles['yacceleration']],
        timestep_length)
    particles['xprevious_position'] = xposition_store
    particles['yprevious_position'] = yposition_store
    return particles, distances, forces, energies


def sample(particles, box_length, initial_particles, system):
    """Sample parameters of interest in the simulation.
    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    initial_particles: util.particle_dt, array-like
        Information about the initial particle conformation.
    system: System
        Details about the whole system
    Returns
    -------
    System:
        Details about the whole system, with the new temperature, pressure,
        msd, and force appended to the appropriate
        arrays.
    """
    temperature_new = calculate_temperature(particles, system.mass)
    system.temperature_sample = np.append(system.temperature_sample,
                                          temperature_new)
    pressure_new = heavy.calculate_pressure(
        particles, box_length, temperature_new, system.cut_off,
        system.constants, system.forcefield)
    msd_new = calculate_msd(particles, initial_particles, box_length)
    system.pressure_sample = np.append(system.pressure_sample, pressure_new)
    system.force_sample = np.append(system.force_sample, np.sum(system.forces))
    system.energy_sample = np.append(
        system.energy_sample, np.sum(system.energies))
    system.msd_sample = np.append(system.msd_sample, msd_new)
    return system


def calculate_msd(particles, initial_particles, box_length):
    """Determines the mean squared displacement of the particles in the system.
    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    initial_particles: util.particle_dt, array_like
        Information about the initial state of the particles.
    box_length: float
        Size of the cell vector.
    Returns
    -------
    float:
        Mean squared deviation for the particles at the given timestep.
    """
    xpos = np.array(particles['xposition'])
    ypos = np.array(particles['yposition'])
    dxinst = xpos - particles['xprevious_position']
    dyinst = ypos - particles['yprevious_position']
    for i in range(0, particles['xposition'].size):
        if np.abs(dxinst[i]) > 0.5 * box_length:
            if xpos[i] <= 0.5 * box_length:
                particles['xpbccount'][i] += 1
            if xpos[i] > 0.5 * box_length:
                particles['xpbccount'][i] -= 1
        xpos[i] += box_length * particles['xpbccount'][i]
        if np.abs(dyinst[i]) > 0.5 * box_length:
            if ypos[i] <= 0.5 * box_length:
                particles['ypbccount'][i] += 1
            if ypos[i] > 0.5 * box_length:
                particles['ypbccount'][i] -= 1
        ypos[i] += box_length * particles['ypbccount'][i]
    dx = xpos - initial_particles['xposition']
    dy = ypos - initial_particles['yposition']
    dr = np.sqrt(dx * dx + dy * dy)
    return np.average(dr ** 2)


def update_positions(positions, old_positions, velocities,
                     accelerations, timestep_length, box_length):
    """Update the particle positions using the Velocity-Verlet integrator.
    Parameters
    ----------
    positions: (2, N) array_like
        Where N is the number of particles, and the first row are the x
        positions and the second row the y positions.
    old_positions: (2, N) array_like
        Where N is the number of particles, and the first row are the
        previous x positions and the second row are the y positions.
    velocities: (2, N) array_like
        Where N is the number of particles, and the first row are the x
        velocities and the second row the y velocities.
    accelerations: (2, N) array_like
        Where N is the number of particles, and the first row are the x
        accelerations and the second row the y
        accelerations.
    timestep_length: float
        Length for each Velocity-Verlet integration step, in seconds.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    Returns
    -------
    (2, N) array_like:
        Updated positions.
    """
    old_positions[0] = np.array(positions[0])
    old_positions[1] = np.array(positions[1])
    positions[0] += (
        velocities[0] * timestep_length) + (
            0.5 * accelerations[0] * timestep_length * timestep_length)
    positions[1] += (
        velocities[1] * timestep_length) + (
            0.5 * accelerations[1] * timestep_length * timestep_length)
    positions[0] = positions[0] % box_length
    positions[1] = positions[1] % box_length
    return [positions[0], positions[1]], [old_positions[0], old_positions[1]]


def update_velocities(velocities, accelerations_old, accelerations_new,
                      timestep_length):
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
    velocities[0] += 0.5 * (accelerations_old[0] +
                            accelerations_new[0]) * timestep_length
    velocities[1] += 0.5 * (accelerations_old[1] +
                            accelerations_new[1]) * timestep_length
    return [velocities[0], velocities[1]]


def calculate_temperature(particles, mass):
    """Determine the instantaneous temperature of the system.
    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    Returns
    -------
    float:
        Calculated instantaneous simulation temperature.
    """
    boltzmann_constant = 1.3806e-23  # joules/kelvin
    atomic_mass_unit = 1.660539e-27  # kilograms
    mass_kg = mass * atomic_mass_unit  # kilograms
    v = np.sqrt((particles['xvelocity'] * particles['xvelocity']) +
                (particles['yvelocity'] * particles['yvelocity']))
    k = 0.5 * np.sum(mass_kg * v * v)
    t = k / (particles.size * boltzmann_constant)
    return t


def compute_force(particles, box_length, cut_off, constants, forcefield, mass):
    r"""Calculates the forces and therefore the accelerations on each of the
    particles in the simulation.
    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    cut_off: float
        The distance greater than which the forces between particles is taken
        as zero.
    constants: float, array_like (optional)
        The constants associated with the particular forcefield used, e.g. for
        the function forcefields.lennard_jones, theses are [A, B]
    forcefield: function (optional)
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
    part, dist, forces, energies = heavy.compute_force(particles,
                                                       box_length,
                                                       cut_off,
                                                       constants,
                                                       forcefield,
                                                       mass=mass)
    return part, dist, forces, energies


def compute_energy(particles, box_length, cut_off, constants, forcefield):
    r"""Calculates the total energy of the simulation.
    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    box_length: float
        Length of a single dimension of the simulation square, in Angstrom.
    cut_off: float
        The distance greater than which the energies between particles is
        taken as zero.
    constants: float, array_like (optional)
        The constants associated with the particular forcefield used, e.g. for
        the function forcefields.lennard_jones, theses are [A, B]
    forcefield: function (optional)
        The particular forcefield to be used to find the energy and forces.
    Returns
    -------
    util.particle_dt, array_like
        Information about particles, with updated accelerations and forces.
    float, array_like
        Current distances between pairs of particles in the simulation.
    float, array_like
        Current energies between pairs of particles in the simulation.
    """
    dist, energies = heavy.compute_energy(particles,
                                          box_length,
                                          cut_off,
                                          constants,
                                          forcefield)
    return dist, energies


def heat_bath(particles, temperature_sample, bath_temperature):
    r"""Rescales the velocities of the particles in the system to control the
    temperature of the simulation. Thereby allowing for an NVT ensemble. The
    velocities are rescaled according the following relationship,
    .. math::
        v_{\text{new}} = v_{\text{old}} \times
        \sqrt{\frac{T_{\text{desired}}}{\bar{T}}}
    Parameters
    ----------
    particles: util.particle_dt, array_like
        Information about the particles.
    temperature_sample: float, array_like
        The temperature at each timestep in the simulation.
    bath_temp: float
        The desired temperature of the simulation.
    Returns
    -------
    util.particle_dt, array_like
        Information about the particles with new, rescaled velocities.
    """
    particles = heavy.heat_bath(particles, temperature_sample,
                                bath_temperature)
    return particles
