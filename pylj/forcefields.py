import numpy as np


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
        12-6 Lennard-Jones function.
    force: bool (optional)
        If true, the negative first derivative will be found.

    Returns
    -------
    float:
        The potential energy or force between the particles.
    """
    if force:
        return 12 * constants[0] * np.power(dr, -13) - (6 * constants[1] *
                                                        np.power(dr, -7))
    else:
        return constants[0] * np.power(dr, -12) - (constants[1] *
                                                   np.power(dr, -6))
