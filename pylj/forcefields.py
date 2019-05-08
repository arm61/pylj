import numpy as np
from numba import jit


@jit
def lennard_jones(dr, constants, force=False):
    r"""Calculate the energy or force for a pair of particles using the
    Lennard-Jones (A/B variant) forcefield.

    .. math::
        E = \frac{A}{dr^{12}} - \frac{B}{dr^6}

    .. math::
        f = \frac{12A}{dr^{13}} - \frac{6B}{dr^7}

    Parameters
    ----------
    dr: float, array_like
        The distances between the all pairs of particles.
    constants: float, array_like
        An array of length two consisting of the A and B parameters for the
        12-6 Lennard-Jones function
    force: bool (optional)
        If true, the negative first derivative will be found.

    Returns
    -------
    float: array_like
        The potential energy or force between the particles.
    """

    if force:
        return 12 * constants[0] * np.power(dr, -13) - (
            6 * constants[1] * np.power(dr, -7))

    else:
        return constants[0] * np.power(dr, -12) - (
            constants[1] * np.power(dr, -6))


@jit
def lennard_jones_sigma_epsilon(dr, constants, force=False):
    r"""Calculate the energy or force for a pair of particles using the
    Lennard-Jones (sigma/epsilon variant) forcefield.

    .. math::
        E = \frac{4e*a^{12}}{dr^{12}} - \frac{4e*a^{6}}{dr^6}

    .. math::
        f = \frac{48e*a^{12}}{dr^{13}} - \frac{24e*a^{6}}{dr^7}

    Parameters
    ----------
    dr: float, array_like
        The distances between the all pairs of particles.
    constants: float, array_like
        An array of length two consisting of the sigma (a) and epsilon (e)
        parameters for the 12-6 Lennard-Jones function
    force: bool (optional)
        If true, the negative first derivative will be found.

    Returns
    -------
    float: array_like
        The potential energy or force between the particles.
    """

    if force:
        return 48 * constants[1] * np.power(constants[0], 12) * np.power(
            dr, -13) - (24 * constants[1] * np.power(
                constants[0], 6) * np.power(dr, -7))
    else:
        return 4 * constants[1] * np.power(dr, -12) - (
            4 * constants[1] * np.power(constants[0], 6) * np.power(dr, -6))


@jit
def buckingham(dr, constants, force=False):
    r""" Calculate the energy or force for a pair of particles using the
    Buckingham forcefield.

    .. math::
        E = Ae^{(-Bdr)} - \frac{C}{dr^6}

    .. math::
        f = ABe^{(-Bdr)} - \frac{6C}{dr^7}

    Parameters
    ----------
    dr: float, array_like
        The distances between all the pairs of particles.
    constants: float, array_like
        An array of length three consisting of the A, B and C parameters for
        the Buckingham function.
    force: bool (optional)
        If true, the negative first derivative will be found.

    Returns
    -------
    float: array_like
        The potential energy or force between the particles.
    """

    if force:
        return constants[0] * constants[1] * np.exp(
            -constants[1] * dr) - 6 * constants[2] / np.power(dr, 7)
    else:
        return constants[0] * np.exp(
            -constants[1] * dr) - constants[2] / np.power(dr, 6)


def square_well(dr, constants, max_val=np.inf, force=False):
    r'''Calculate the energy or force for a pair of particles using a
    square well model.

    .. math::
        E = {
        if dr < sigma:
            E = max_val
        elif sigma <= dr < lambda * sigma:
            E = -epsilon
        elif r >= lambda * sigma:
            E = 0
        }
    .. math::
        f = {
        if sigma <= dr < lambda * sigma:
            f = inf
        else:
            f = 0
        }
    Parameters
    ----------
    dr: float, array_like
        The distances between all the pairs of particles.
    constants: float, array_like
        An array of length three consisting of the epsilon, sigma, and lambda
        parameters for the square well model.
    max_val: int (optional)
        Upper bound for values in square well - replaces usual infinite values
    force: bool (optional)
        If true, the negative first derivative will be found.

    Returns
    -------
    float: array_like
        The potential energy between the particles.
    '''
    if not isinstance(dr, np.ndarray):
        if isinstance(dr, list):
            dr = np.array(dr, dtype='float')
        elif isinstance(dr, float):
            dr = np.array([dr], dtype='float')

    if force:
        raise ValueError("Force is infinite at sigma <= dr < lambda * sigma")

    else:
        E = np.zeros_like(dr)
        E[np.where(dr < constants[0])] = max_val
        E[np.where(dr >= constants[2] * constants[1])] = 0

        # apply mask for sigma <= dr < lambda * sigma
        a = constants[1] <= dr
        b = dr < constants[2] * constants[1]
        E[np.where(a & b)] = -constants[0]

        if len(E) == 1:
            return float(E[0])
        else:
            return np.array(E, dtype='float')
