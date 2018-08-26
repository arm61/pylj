import numpy as np


def lennard_jones(dr, A, B, force=False):
    """Calculate the energy of a pair of particles at a given distance.

    Parameters
    ----------
    dr: float
        The distances between the pairs of particles.
    A: float
        The value of the A parameter for the Lennard-Jones potential.
    B: float
        The value of the B parameter for the Lennard-Jones potential.
    force: bool (optional)
        If true, the negative first derivative will be found.

    Returns
    -------
    float:
        The potential energy or force between the particles.
    """
    if force:
        return 12 * A * np.power(dr, -13) - 6 * B * np.power(dr, -7)
    else:
        return A * np.power(dr, -12) - B * np.power(dr, -6)
