import numpy as np


class lennard_jones(object):
    r"""Calculate the energy or force for a pair of particles using the
    Lennard-Jones (A/B variant) forcefield.

    Parameters
    ----------
    constants: float, array_like
        An array of length two consisting of the A and B parameters for the
        12-6 Lennard-Jones function

    """
    def __init__(self, constants):
        self.a = constants[0]
        self.b = constants[1]

    def energy(self, dr ):
        r"""Calculate the energy for a pair of particles using the
        Lennard-Jones (A/B variant) forcefield.

        .. math::
            E = \frac{A}{dr^{12}} - \frac{B}{dr^6}

        Attributes:
        ----------
        dr (float): The distance between particles.

        Returns
        -------
        float: array_like
        The potential energy between the particles.
        """
        self.energy = self.a * np.power(dr, -12) - (self.b * np.power(dr, -6))
        return self.energy
    
    def force(self, dr):
        r"""Calculate the force for a pair of particles using the
        Lennard-Jones (A/B variant) forcefield.

        .. math::
            f = \frac{12A}{dr^{13}} - \frac{6B}{dr^7}

        Attributes:
        ----------
        dr (float): The distance between particles.
        
        Returns
        -------
        float: array_like
        The force between the particles.
        """
        self.force = 12 * self.a * np.power(dr, -13) - (6 * self.b * np.power(dr, -7))
        return self.force



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
        self.sigma = constants[0]
        self.epsilon = constants[1]
    
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
        self.a = constants[0]
        self.b = constants[1]
        self.c = constants[2]
    
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
        self.energy = self.a * np.exp(- np.multiply(self.b, dr)) - self.c / np.power(dr, 6)
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
        self.force = self.a * self.b * np.exp(- np.multiply(self.b, dr)) - 6 * self.c / np.power(dr, 7)
        return self.force



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

        self.epsilon = constants[0]
        self.sigma = constants[1]
        self.lamda = constants[2] #Spelling as lamda not lambda to avoid calling python lambda function
        self.max_val = max_val

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