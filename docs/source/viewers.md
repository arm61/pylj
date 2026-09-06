---
file_format: mystnb
kernelspec:
  name: python3
---

# Viewers and panes

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

## Existing viewers

pylj comes with eight viewers, each a live figure that redraws when its `update(simulation)` method is called:

- `JustCell`: the particle positions
- `Energy`: positions and the energy: the total energy, potential plus kinetic, for molecular dynamics, and the potential energy for Monte Carlo, which has no kinetic energy to add
- `MaxBolt`: positions and a histogram of particle speeds
- `RDF`: positions and the radial distribution function
- `CellPlus`: positions and one plot of data you supply
- `Interactions`: positions, temperature, pressure and total energy
- `Phase`: positions, total energy, mean squared displacement and the radial distribution function
- `Scattering`: positions, the radial distribution function, mean squared displacement and the scattering profile

The `MaxBolt`, `Interactions`, `Phase` and `Scattering` viewers plot quantities that only a molecular dynamics run records, and refuse a Monte Carlo simulation before they build their figure, naming themselves in the error. Every viewer takes the `MDSimulation` or `MCSimulation`, an optional `size` of `'small'`, `'medium'` or `'large'`, and an optional `diameter` to draw the particles at, in Angstrom; `CellPlus` also takes the axis labels of its custom plot. Every viewer has an `average()` method that replaces the latest curve with the mean of every update so far; it raises `ValueError` unless one of the viewer's panes keeps a history, which the radial distribution function and scattering panes do. Full details are in the {doc}`sample` module documentation.

The viewers use the inline matplotlib backend. Start notebooks with `%matplotlib inline`.

## Panes

A viewer is a grid of panes. A pane draws one quantity into one matplotlib axes and has two methods: `setup(ax, simulation)` creates the line and labels once, and `update(ax, simulation)` pushes the current state of the simulation into that line. The panes that exist are `CellPane`, `EnergyPane`, `TemperaturePane`, `PressurePane`, `MSDPane`, `RDFPane`, `ScatteringPane`, `MaxwellBoltzmannPane` and `CustomPane`.

Panes that plot a quantity against time read it from the sample arrays on the simulation, which `sample()` fills. Each call records the current step in `samples.step`, so a loop may sample as often or as rarely as it likes. A molecular dynamics pane plots against `samples.step` times the timestep; a Monte Carlo pane plots against the step, since a Monte Carlo simulation has no timestep. The radial distribution function pane shows its axes in metres. `step()` advances the step count.

## Building your own viewer

The examples below share one molecular dynamics simulation of sixteen argon atoms.

```{code-cell} python
from pylj.md import MDSimulation
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
simulation = MDSimulation.initialise(
    16, 300, 30, species=[argon], pair_potentials={(argon, argon): lj}, seed=0
)
```

To combine existing panes in a new layout, pass a list of one, two or four panes to `Viewer`:

```{code-cell} python
from pylj.sample import Viewer, CellPane, TemperaturePane

viewer = Viewer(simulation, [CellPane(), TemperaturePane()])
for _ in range(300):
    simulation.step()
    if simulation.steps % 10 == 0:
        simulation.sample()
    if simulation.steps % 100 == 0:
        viewer.update(simulation)
```

To plot a new quantity, write a pane. This one plots the x velocity of the first particle against time; it reads velocities and a time, which only a molecular dynamics simulation has, so it sets `needs_md` and the viewer refuses a Monte Carlo simulation with a message rather than an `AttributeError`:

```{code-cell} python
from pylj.sample import Pane, Viewer, CellPane

class FirstParticlePane(Pane):
    needs_md = True

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

simulation = MDSimulation.initialise(
    16, 300, 30, species=[argon], pair_potentials={(argon, argon): lj}, seed=0
)
viewer = Viewer(simulation, [CellPane(), FirstParticlePane()])
for _ in range(300):
    simulation.step()
    if simulation.steps % 10 == 0:
        viewer.update(simulation)
```

A pane that needs a quantity sampled by the simulation itself, rather than one it can compute from the configuration, needs that quantity added to the simulation class and recorded in its `sample` method.
