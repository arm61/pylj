import numpy as np
from numpy.typing import ArrayLike, NDArray


def _scalar_or_array(x: NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Return a Python float for a 0-d array, otherwise the array itself.

    Args:
        x: The array to convert.

    Returns:
        A float when `x` is 0-d, otherwise `x` unchanged.
    """
    if x.ndim == 0:
        return float(x)
    return x


class lennard_jones_sigma_epsilon:
    r"""Calculate the energy or force for a pair of particles using the
    Lennard-Jones (sigma/epsilon variant) forcefield.

    Parameters
    ----------
    constants: float, array_like
        An array of length two consisting of the sigma (a) and epsilon (e)
        parameters for the 12-6 Lennard-Jones function

    """
    def __init__(self, constants):
        if len(constants) != 2:
            raise IndexError(f'There should be two constants per set, not {len(constants)}')
        
        self.sigma = constants[0]
        self.epsilon = constants[1]

    def energy(self, dr: ArrayLike) -> float | NDArray[np.float64]:
        r"""Calculate the energy for a pair of particles using the
        Lennard-Jones (sigma/epsilon variant) forcefield.

        .. math::
            E = \frac{4 \epsilon \sigma^{12}}{r^{12}} - \frac{4 \epsilon \sigma^{6}}{r^{6}}

        Args:
            dr: Separation between the particles, in metres; a float or an
                array.

        Returns:
            float or ndarray: The pair energy, as a float for scalar input
            and an array otherwise.
        """
        dr = np.asarray(dr, dtype=float)
        energy = 4 * self.epsilon * np.power(self.sigma, 12) * np.power(dr, -12) - (
                        4 * self.epsilon * np.power(self.sigma, 6) * np.power(dr, -6))
        return _scalar_or_array(energy)

    def force(self, dr: ArrayLike) -> float | NDArray[np.float64]:
        r"""Calculate the force for a pair of particles using the
        Lennard-Jones (sigma/epsilon variant) forcefield.

        .. math::
            f = \frac{48 \epsilon \sigma^{12}}{r^{13}} - \frac{24 \epsilon \sigma^{6}}{r^{7}}

        Args:
            dr: Separation between the particles, in metres; a float or an
                array.

        Returns:
            float or ndarray: The pair force, negative where the interaction
            is attractive, as a float for scalar input and an array
            otherwise.
        """
        dr = np.asarray(dr, dtype=float)
        force = 48 * self.epsilon * np.power(self.sigma, 12) * np.power(
            dr, -13) - (24 * self.epsilon * np.power(self.sigma, 6) * np.power(dr, -7))
        return _scalar_or_array(force)

    def mixing(self, constants_2):
        r"""Calculate mixing for two sets of constants.

        .. math::
            \sigma_{12} = \frac{\sigma_1 + \sigma_2}{2}

            \epsilon_{12} = \sqrt{\epsilon_1 \epsilon_2}

        Args:
            constants_2: The second set of constants.
        """
        sigma2 = constants_2[0]
        epsilon2 = constants_2[1]
        self.sigma = (self.sigma+sigma2)/2
        self.epsilon = np.sqrt(self.epsilon * epsilon2)

    @property
    def diameter(self) -> float:
        """Separation at the minimum of the pair potential, in metres.

        Used as the particle diameter when drawing the simulation cell.
        """
        return 2 ** (1 / 6) * self.sigma


class lennard_jones(lennard_jones_sigma_epsilon):
    r"""Converts a/b variant values to sigma/epsilon variant
    then maps to lennard_jones_sigma_epsilon class

    .. math::
        \sigma = \left(\frac{A}{B}\right)^{1/6}

        \epsilon = \frac{B^{2}}{4A}

    Parameters
    ----------
    constants: float, array_like
        An array of length two consisting of the A and B
        parameters for the 12-6 Lennard-Jones function
    """   
    def __init__(self, constants):
        if len(constants) != 2:
            raise IndexError(f'There should be two constants per set, not {len(constants)}')
        self.a = constants[0]
        self.b = constants[1]
        sigma = (self.a / self.b)**(1/6)
        epsilon = (self.b**2)/(4*self.a)
        super().__init__([sigma, epsilon])

    def mixing(self, constants_2):
        r"""Converts second set of a/b constants into sigma/epsilon
        for use in mixing method. Then converts changed self sigma/epsilon
        values back to a/b

        .. math::
            A = 4 \epsilon \sigma^{12}

            B = 4 \epsilon \sigma^{6}

        Args:
            constants_2: The second set of constants.
        """
        a2 = constants_2[0]
        b2 = constants_2[1]
        sigma2 = (a2 / b2)**(1/6)
        epsilon2 = (b2**2)/(4*a2)
        super().mixing([sigma2,epsilon2])
        self.a = 4 * self.epsilon * (self.sigma**12)
        self.b = 4 * self.epsilon * (self.sigma**6)


class buckingham:
    r""" Calculate the energy or force for a pair of particles using the
    Buckingham forcefield.

    Parameters
    ----------
    constants: float, array_like
        An array of length three consisting of the A, B and C parameters for
        the Buckingham function.

    """
    def __init__(self, constants):
        if len(constants) != 3:
            raise IndexError(f'There should be three constants per set, not {len(constants)}')
        self.a = constants[0]
        self.b = constants[1]
        self.c = constants[2]

    def energy(self, dr: ArrayLike) -> float | NDArray[np.float64]:
        r"""Calculate the energy for a pair of particles using the
        Buckingham forcefield.

        .. math::
            E = A e^{-Br} - \frac{C}{r^{6}}

        Args:
            dr: Separation between the particles, in metres; a float or an
                array.

        Returns:
            float or ndarray: The pair energy, as a float for scalar input
            and an array otherwise.
        """
        dr = np.asarray(dr, dtype=float)
        energy = self.a * np.exp(- np.multiply(self.b, dr)) - self.c / np.power(dr, 6)
        return _scalar_or_array(energy)

    def force(self, dr: ArrayLike) -> float | NDArray[np.float64]:
        r"""Calculate the force for a pair of particles using the
        Buckingham forcefield.

        .. math::
            f = A B e^{-Br} - \frac{6C}{r^{7}}

        Args:
            dr: Separation between the particles, in metres; a float or an
                array.

        Returns:
            float or ndarray: The pair force, negative where the interaction
            is attractive, as a float for scalar input and an array
            otherwise.
        """
        dr = np.asarray(dr, dtype=float)
        force = self.a * self.b * np.exp(- np.multiply(self.b, dr)) - 6 * self.c / np.power(dr, 7)
        return _scalar_or_array(force)

    def mixing(self, constants2):
        r"""Calculate mixing for two sets of constants.

        .. math::
            A_{12} = \sqrt{A_1 A_2}

            B_{12} = \sqrt{B_1 B_2}

            C_{12} = \sqrt{C_1 C_2}

        Args:
            constants2: The second set of constants.
        """
        self.a = np.sqrt(self.a*constants2[0])
        self.b = np.sqrt(self.b*constants2[1])
        self.c = np.sqrt(self.c*constants2[2])

    @property
    def diameter(self) -> float:
        """Separation at the minimum of the pair potential, in metres.

        The Buckingham potential has no closed-form minimum, so it is located
        numerically on a logarithmic grid between 0.1 and 50 Angstrom. The
        global maximum on that grid is the repulsive barrier separating the
        unphysical collapse at small separation from the well; the diameter
        is the position of the minimum beyond that barrier.

        Raises:
            ValueError: If the potential has no minimum between 0.1 and 50
                Angstrom.
        """
        r = np.logspace(-11, np.log10(5e-9), 2000)
        energy = self.a * np.exp(-self.b * r) - self.c / np.power(r, 6)
        barrier = int(np.argmax(energy))
        well = barrier + int(np.argmin(energy[barrier:]))
        if barrier == r.size - 1 or well == r.size - 1:
            raise ValueError(
                "No potential minimum was found between 0.1 and 50 Angstrom for "
                f"a={self.a}, b={self.b}, c={self.c}. Check the units of the constants, "
                "or pass diameter= to initialise."
            )
        return float(r[well])


class square_well:
    r'''Calculate the energy or force for a pair of particles using a
    square well model.

    The return rule is the same as the other forcefields: a scalar
    separation gives a scalar energy and an array gives an array. There is
    no ``force(dr)`` for this potential, so it cannot drive the molecular
    dynamics engine (#80).

    Parameters
    ----------
    constants: float, array_like
        An array of length three consisting of the epsilon, sigma, and lambda
        parameters for the square well model.
    max_val: int (optional)
        Upper bound for values in square well - replaces usual infinite values

    '''
    def __init__(self, constants, max_val=np.inf):
        if len(constants) != 3:
            raise IndexError(f'There should be three constants per set, not {len(constants)}')
        self.epsilon = constants[0]
        self.sigma = constants[1]
        # Spelling as lamda not lambda to avoid calling the python lambda keyword.
        self.lamda = constants[2]
        self.max_val = max_val

    def energy(self, dr: ArrayLike) -> float | NDArray[np.float64]:
        r'''Calculate the energy for a pair of particles using a
        square well model.

        .. math::
            E = \begin{cases}
                E_{\mathrm{max}} & r < \sigma \\
                -\epsilon & \sigma \le r < \lambda \sigma \\
                0 & r \ge \lambda \sigma
            \end{cases}

        Args:
            dr: Separation between the particles, in metres; a float or an
                array.

        Returns:
            float or ndarray: The pair energy, as a float for scalar input
            and an array otherwise.
        '''

        dr = np.asarray(dr, dtype=float)
        dr_1d = np.atleast_1d(dr)

        E = np.zeros_like(dr_1d, dtype=float)
        E[np.where(dr_1d < self.sigma)] = self.max_val
        E[np.where(dr_1d >= self.lamda * self.sigma)] = 0

        # apply mask for sigma <= dr < lambda * sigma
        a = self.sigma <= dr_1d
        b = dr_1d < self.lamda * self.sigma
        E[np.where(a & b)] = -self.epsilon

        if dr.ndim == 0:
            return float(E[0])
        return E

    @property
    def diameter(self) -> float:
        """Hard-core diameter sigma, in metres."""
        return self.sigma

    def force(self):
        r'''The force of a pair of particles using a square well model is given by:

        .. math::
            f = \begin{cases}
                \infty & r = \sigma \text{ or } r = \lambda \sigma \\
                0 & \text{otherwise}
            \end{cases}

        The force is infinite at the steps and zero elsewhere, so the model
        cannot be integrated and is for Monte Carlo only.

        Raises:
            ValueError: Always; the force is infinite at the steps of the
                well and cannot be returned as a finite value.
        '''
        raise ValueError("Force is infinite at sigma <= dr < lambda * sigma")