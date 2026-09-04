bring your own forcefield
=========================

pylj takes the pair potential as a class passed to :code:`md.initialise` or :code:`mc.initialise` as :code:`forcefield`, with its parameters passed as :code:`constants`. The Lennard-Jones, Buckingham and square-well potentials in the :doc:`forcefields` module are written this way, and a custom potential follows the same form.

Writing your own forcefield and passing it to the pylj engine is very simple. Firstly, the forcefield should have the following form:

.. code-block:: python

    class forcefield(object):

        def __init__(self, constants):
            # Define constants
            # For instance:
            self.a = constants[0]
            self.b = constants[1]

        @property
        def diameter(self):
            # The separation at the potential minimum, in metres
            return some_function_of(self.a, self.b)

        def energy(self, dr):
            return func(dr, self.a, self.b)

        def force(self, dr):
            return other_func(dr, self.a, self.b)

        def mixing(self, constants2):
            a2 = constants2[0]
            b2 = constants2[1]

            self.a = mixing_func(self.a, a2)
            self.b = other_mixing_func(self.b, b2)

The four members do the following.

- :code:`diameter` is the separation at the minimum of the pair potential, in metres. The particles are drawn with this diameter, so the picture of the cell is to scale. It can be overridden with the :code:`diameter` argument to :code:`md.initialise` or :code:`mc.initialise`, which is given in Angstrom.
- :code:`energy` and :code:`force` return the pair energy and the magnitude of the pair force for an array of separations :code:`dr`, in metres. When particles are placed for an initial configuration, pylj locates the separation at which :code:`energy` falls from positive to zero or negative, between 0.1 and 50 Angstrom, and keeps particles at least this far apart; it does not use :code:`diameter` for this. A forcefield whose energy never falls from positive within that range is refused when the system is constructed.
- :code:`mixing` combines this forcefield's constants with those of a second particle type, and is called when a simulation has more than one set of constants. Geometric or arithmetic means of the two sets are the usual choices.

The Lennard-Jones forcefield in the :doc:`forcefields` module is a complete example.
