import numpy as np


def lennard_jones(dr, constants, force=False):
    """Calculate the energy of a pair of particles at a given distance.

    Parameters
    ----------
    dr: float, array_like
        The distances between the all pairs of particles.
    constants: float, array_like
        An array of lenght two consisting of the A and B parameters for the
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
