visualisation
=============

existing viewers
----------------

pylj comes with eight viewers, each a live figure that redraws when its :code:`update(simulation)` method is called:

- :code:`JustCell`: the particle positions
- :code:`Energy`: positions and the total energy
- :code:`MaxBolt`: positions and a histogram of particle speeds
- :code:`RDF`: positions and the radial distribution function
- :code:`CellPlus`: positions and one plot of data you supply
- :code:`Interactions`: positions, temperature, pressure and total energy
- :code:`Phase`: positions, total energy, mean squared displacement and the radial distribution function
- :code:`Scattering`: positions, the radial distribution function, mean squared displacement and the scattering profile

The :code:`MaxBolt`, :code:`Interactions`, :code:`Phase` and :code:`Scattering` viewers plot quantities that only a molecular dynamics run records, and refuse a Monte Carlo simulation before they build their figure, naming themselves in the error. Every viewer takes the :code:`MDSimulation` or :code:`MCSimulation` and an optional :code:`size` of :code:`'small'`, :code:`'medium'` or :code:`'large'`. Every viewer has an :code:`average()` method that replaces the latest curve with the mean of every update so far; it raises :code:`ValueError` unless one of the viewer's panes keeps a history, which the radial distribution function and scattering panes do. Full details are in the :doc:`sample` module documentation.

The viewers use the inline matplotlib backend. Start notebooks with :code:`%matplotlib inline`.

panes
-----

A viewer is a grid of panes. A pane draws one quantity into one matplotlib axes and has two methods: :code:`setup(ax, simulation)` creates the line and labels once, and :code:`update(ax, simulation)` pushes the current state of the simulation into that line. The panes that exist are :code:`CellPane`, :code:`EnergyPane`, :code:`TemperaturePane`, :code:`PressurePane`, :code:`MSDPane`, :code:`RDFPane`, :code:`ScatteringPane`, :code:`MaxwellBoltzmannPane` and :code:`CustomPane`.

Panes that plot a quantity against time read it from the sample arrays on the simulation, which :code:`sample()` fills. Each call records the current step in :code:`samples.step`, so a loop may sample as often or as rarely as it likes. A molecular dynamics pane plots against :code:`samples.step` times the timestep; a Monte Carlo pane plots against the step, since a Monte Carlo simulation has no timestep. :code:`step()` advances the step count.

building your own viewer
------------------------

To combine existing panes in a new layout, pass a list of one, two or four panes to :code:`Viewer`:

.. code-block:: python

    from pylj.sample import Viewer, CellPane, TemperaturePane

    viewer = Viewer(simulation, [CellPane(), TemperaturePane()])

To plot a new quantity, write a pane. This one plots the x velocity of the first particle against time:

.. code-block:: python

    from pylj.sample import Pane, Viewer, CellPane

    class FirstParticlePane(Pane):
        def __init__(self):
            self.times = []
            self.velocities = []

        def setup(self, ax, simulation):
            ax.plot([], [])
            ax.set_xlabel("Time/s")
            ax.set_ylabel("x velocity/m s$^{-1}$")

        def update(self, ax, simulation):
            self.times.append(simulation.time)
            self.velocities.append(simulation.configuration.velocity[0, 0])
            ax.lines[0].set_data(self.times, self.velocities)
            ax.relim()
            ax.autoscale_view()

    viewer = Viewer(simulation, [CellPane(), FirstParticlePane()])

A pane that needs a quantity sampled by the simulation itself, rather than one it can compute from the configuration, needs that quantity added to the simulation class and recorded in its :code:`sample` method.
