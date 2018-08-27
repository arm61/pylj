bring your own forcefield
=========================

Although pylj was originally designed to use only the Lennard-Jones potential model, hence pyLennard-Jones. Following the release of pylj-1.1.0, it is now possible to pass custom forcefields to the simulation.

A quick caveat is that currently the particle size in the :code:`cellview` is defined only by the box size, therefore not by the atom being simulated. The result is that if you try and simulated something significantly different in size to argon, things might look a bit funny. If you have any ideas of how to fix this please check this issue_ in the GitHub repository.

Writing your own forcefield and passing it to the pylj engine is very simple, firstly the forcefield should have the following form,

.. code-block:: python

    def forcefield(dr, constants, force=False):
        if force:
            return force
        else:
            return energy

An example of this for the Lennard-Jones forcefield (which is the default for pylj) and be found in the :doc:`forcefields` module.

It is necessary to pass this forcefield to the pylj engine. This can be achieved during the initialisation of the :code:`System` class object, by passing the defined function as the variable :code:`forcefield`. This can be seen for the Lennard-Jones forcefield in the :code:`System` class definition in the :doc:`util` module. 

.. _issue: https://github.com/arm61/pylj/issues/29
