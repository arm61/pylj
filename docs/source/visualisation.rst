visualisation
=============

existing viewers
----------------

pylj comes with eight viewers, each a live figure that redraws when its :code:`update(system)` method is called:

- :code:`JustCell`: the particle positions
- :code:`Energy`: positions and the total energy
- :code:`MaxBolt`: positions and a histogram of particle speeds
- :code:`RDF`: positions and the radial distribution function
- :code:`CellPlus`: positions and one plot of data you supply
- :code:`Interactions`: positions, temperature, pressure and total force
- :code:`Phase`: positions, total energy, mean squared displacement and the radial distribution function
- :code:`Scattering`: positions, the radial distribution function, mean squared displacement and the scattering profile

Every viewer takes the :code:`System` and an optional :code:`size` of :code:`'small'`, :code:`'medium'` or :code:`'large'`. The viewers that show the radial distribution function or the scattering profile also have an :code:`average()` method that replaces the latest curve with the mean of every update so far. Full details are in the :doc:`sample` module documentation.

The viewers use the inline matplotlib backend. Start notebooks with :code:`%matplotlib inline`.

panes
-----

A viewer is a grid of panes. A pane draws one quantity into one matplotlib axes and has two methods: :code:`setup(ax, system)` creates the line and labels once, and :code:`update(ax, system)` pushes the current state of the system into that line. The panes that exist are :code:`CellPane`, :code:`EnergyPane`, :code:`TemperaturePane`, :code:`PressurePane`, :code:`ForcePane`, :code:`MSDPane`, :code:`RDFPane`, :code:`ScatteringPane`, :code:`MaxwellBoltzmannPane` and :code:`CustomPane`.

Panes that plot a quantity against time read it from the sample arrays on the :code:`System` object, which the :code:`md.sample` and :code:`mc.sample` functions fill. Each call records the current step in :code:`step_sample`, so a loop may sample as often or as rarely as it likes.

building your own viewer
------------------------

To combine existing panes in a new layout, pass a list of one, two or four panes to :code:`Viewer`:

.. code-block:: python

    from pylj.sample import Viewer, CellPane, TemperaturePane

    viewer = Viewer(system, [CellPane(), TemperaturePane()])

To plot a new quantity, write a pane. This one plots the x velocity of the first particle against time:

.. code-block:: python

    import numpy as np
    from pylj.sample import Pane, Viewer, CellPane

    class FirstParticlePane(Pane):
        def __init__(self):
            self.velocities = []

        def setup(self, ax, system):
            ax.plot([], [])
            ax.set_xlabel("Time/s")
            ax.set_ylabel("x velocity/m s$^{-1}$")

        def update(self, ax, system):
            self.velocities.append(system.particles["xvelocity"][0])
            time = np.arange(len(self.velocities)) * system.timestep_length
            ax.lines[0].set_data(time, self.velocities)
            ax.relim()
            ax.autoscale_view()

    viewer = Viewer(system, [CellPane(), FirstParticlePane()])

A pane that needs a quantity sampled by the simulation itself, rather than one it can compute from the particles, needs that quantity added to :code:`System` and recorded in the :code:`sample` function of the engine in use.
