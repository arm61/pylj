import numpy as np


class lennard_jones_sigma_epsilon(object):
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
        self.point_size = 1.3e10 * (self.sigma*(2**(1/6)))
    
    def energy(self, dr):
        r"""Calculate the energy for a pair of particles using the
        Lennard-Jones (sigma/epsilon variant) forcefield.

        .. math::
            E = \frac{4e*a^{12}}{dr^{12}} - \frac{4e*a^{6}}{dr^6}

        Attributes:
        ----------
        dr (float): The distance between particles.
 
        Returns
        -------
        float: array_like
        The potential energy between the particles.
        """
        self.energy = 4 * self.epsilon * np.power(self.sigma, 12) * np.power(dr, -12) - (
                        4 * self.epsilon * np.power(self.sigma, 6) * np.power(dr, -6)) 
        return self.energy 
    
    def force(self, dr):
        r"""Calculate the force for a pair of particles using the
        Lennard-Jones (sigma/epsilon variant) forcefield.

        .. math::
            f = \frac{48e*a^{12}}{dr^{13}} - \frac{24e*a^{6}}{dr^7}

        Attributes:
        ----------
        dr (float): The distance between particles.
 
        Returns
        -------
        float: array_like
        The force between the particles.
        """
        self.force = 48 * self.epsilon * np.power(self.sigma, 12) * np.power(
            dr, -13) - (24 * self.epsilon * np.power(self.sigma, 6) * np.power(dr, -7))
        return self.force
    
    def mixing(self, constants_2):
        r""" Calculates mixing for two sets of constants
        
        ..math::
            \sigma_{12} = \frac{\sigma_1 + \sigma_2}{2}
            \epsilon{12} = \sqrt{\epsilon_1 * \epsilon_2}
        
        Parameters:
        ----------
        constants_2: float, array_like
            The second set of constantss
        """
        sigma2 = constants_2[0]
        epsilon2 = constants_2[1]
        self.sigma = (self.sigma+sigma2)/2
        self.epsilon = np.sqrt(self.epsilon * epsilon2)


class lennard_jones(lennard_jones_sigma_epsilon):
    r"""Converts a/b variant values to sigma/epsilon variant
    then maps to lennard_jones_sigma_epsilon class

    ..math::
        \sigma = \frac{a}{b}^(\frac{1}{6})
        \sigma = \frace{b^2}{4*a}

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

        ..math::
            a = 4*\epsilon*(\sigma^12)
            b = 4*\epsilon*(\sigma^6)

        Parameters:
        ----------
        constants_2: float, array_like
            The second set of constantss
        """
        a2 = constants_2[0]
        b2 = constants_2[1]
        sigma2 = (a2 / b2)**(1/6)
        epsilon2 = (b2**2)/(4*a2)
        super().mixing([sigma2,epsilon2])
        self.a = 4 * self.epsilon * (self.sigma**12)
        self.b = 4 * self.epsilon * (self.sigma**6)


class buckingham(object):
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
        self.point_size = 8 # Needs better solution relevant to constants
    
    def energy(self, dr):
        r"""Calculate the energy for a pair of particles using the
        Buckingham forcefield.

        .. math::
            E = Ae^{(-Bdr)} - \frac{C}{dr^6}

        Attributes:
        ----------
        dr (float): The distance between particles.
 
        Returns
        -------
        float: array_like
        The potential energy between the particles.
        """
        energy = self.a * np.exp(- np.multiply(self.b, dr)) - self.c / np.power(dr, 6)
        # Cut out infinite values where r = 0
        if type(dr) != float:
            energy = np.array(energy)
            energy[np.where(energy > 10e300)] = 0
            energy[np.where(energy < -10e300)] = 0
        self.energy = energy
        return self.energy
    
    def force(self, dr):
        r"""Calculate the force for a pair of particles using the
        Buckingham forcefield.

        .. math::
            f = ABe^{(-Bdr)} - \frac{6C}{dr^7}

        Attributes:
        ----------
        dr (float): The distance between particles.
 
        Returns
        -------
        float: array_like
        The force between the particles.
        """
        force = self.a * self.b * np.exp(- np.multiply(self.b, dr)) - 6 * self.c / np.power(dr, 7)
        # Cut out infinite values where r = 0
        if type(dr) != float:
            force = np.array(force)
            force[np.where(force > 10e300)] = 0
            force[np.where(force < -10e300)] = 0
        self.force = force
        return self.force

    def mixing(self, constants2):
        r""" Calculates mixing for two sets of constants
        
        ..math::
            a_{12} = \sqrt{a_1 * a_2}
            b_{12} = \sqrt{b_1 * b_2}
            c_{12} = \sqrt{c_1 * c_2}
        
        Attributes:
        ----------
        constants_2: float, array_like
            The second set of constantss
        """
        self.a = np.sqrt(self.a*constants2[0])
        self.b = np.sqrt(self.b*constants2[1])
        self.c = np.sqrt(self.c*constants2[2])


class square_well(object):
    r'''Calculate the energy or force for a pair of particles using a
    square well model.

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
        self.lamda = constants[2] #Spelling as lamda not lambda to avoid calling python lambda function
        self.max_val = max_val
        self.point_size = 10

    def energy(self, dr):
        r'''Calculate the energy for a pair of particles using a
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

        Attributes:
        ----------
        dr (float): The distance between particles.
 
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

        E = np.zeros_like(dr)
        E = np.zeros_like(dr)
        E[np.where(dr < self.epsilon)] = self.max_val
        E[np.where(dr >= self.lamda * self.sigma)] = 0

        # apply mask for sigma <= dr < lambda * sigma
        a = self.sigma <= dr
        b = dr < self.lamda * self.sigma
        E[np.where(a & b)] = -self.epsilon

        if len(E) == 1:
            self.energy = float(E[0])
        else:
            self.energy = np.array(E, dtype='float')
            
        return self.energy
        
    def force(self):
        r'''The force of a pair of particles using a square well model is given by:

        .. math::
        f = {
        if sigma <= dr < lambda * sigma:
            f = inf
        else:
            f = 0
        }

        Therefore the force here will always be infinite, and therefore not possible to simulate
        '''
        raise ValueError("Force is infinite at sigma <= dr < lambda * sigma")