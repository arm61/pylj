bring your own potential
========================

A pylj system is built from the species it contains and the pair potential acting between each pair of species:

.. code-block:: python

    from pylj import md
    from pylj.potentials import Species, lennard_jones

    argon = Species(mass=39.948, name="argon")
    lj = lennard_jones(epsilon=1.577e-21, sigma=3.372e-10)

    system = md.initialise(
        100, 300, 40, "square",
        species=[argon],
        pair_potentials={(argon, argon): lj},
    )

The Lennard-Jones, Buckingham and square-well potentials in the :doc:`potentials` module are subclasses of :code:`PairPotential`, and a custom potential follows the same form:

.. code-block:: python

    import numpy as np
    from pylj.potentials import PairPotential

    class soft_sphere(PairPotential):

        def __init__(self, *, epsilon, sigma):
            self.epsilon = epsilon
            self.sigma = sigma

        def energies(self, dr):
            dr = np.asarray(dr, dtype=float)
            return self.epsilon * (self.sigma / dr) ** 12

        def forces(self, dr):
            dr = np.asarray(dr, dtype=float)
            return 12 * self.epsilon * (self.sigma / dr) ** 12 / dr

The two methods take an array of pair separations :code:`dr`, in metres, and return an array of the same shape.

- :code:`energies` returns the pair energy at each separation, in joules.
- :code:`forces` returns the radial force at each separation, in newtons: minus the derivative of the energy with respect to the separation, so it is positive where the interaction is repulsive and negative where it is attractive. A potential with no finite force, such as the square well, raises :code:`ValueError` here; it can still drive Monte Carlo, which evaluates the energies only.

The constructor is yours to define. Keyword-only parameters named after the physical quantities, as above, mean a swapped pair of numbers is an error rather than a silently wrong model.

A mixture is more species and more entries in :code:`pair_potentials`: one for each species with itself and one for each pair of different species, in either order.
