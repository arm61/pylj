bring your own forcefield
=========================

Although pylj was originally designed to use only the Lennard-Jones potential model, hence pyLennard-Jones. Following the release of pylj-1.1.0, it is now possible to pass custom forcefields to the simulation. As of pylj-1.5.0, these have become class-based.

The particles are drawn with a diameter taken from the forcefield's :code:`diameter` property, which should return the separation at the minimum of the pair potential in metres. It can be overridden with the :code:`diameter` argument to :code:`md.initialise` or :code:`mc.initialise`, which is given in Angstrom. For mixing rules, geometric or arithmetic means of the two sets of constants are usual.

Writing your own forcefield and passing it to the pylj engine is very simple, firstly the forcefield should have the following form,

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

An example of this for the Lennard-Jones forcefield (which is the default for pylj) and be found in the :doc:`forcefields` module.

It is necessary to pass this forcefield to the pylj engine. This can be achieved during the initialisation of the :code:`System` class object, by passing the defined class as the variable :code:`forcefield`. This can be seen for the Lennard-Jones forcefield in the :code:`System` class definition in the :doc:`util` module. 

.. _issue: https://github.com/arm61/pylj/issues/29
