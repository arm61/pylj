---
file_format: mystnb
kernelspec:
  name: python3
---

# Viewers and panes

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
%config InlineBackend.figure_format = "retina"
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

## Viewers

A viewer is a figure that redraws when `update(simulation)` is called. Eight are provided in `pylj.sample`:

- `JustCell`: the atom positions.
- `Energy`: positions and the energy: total energy for molecular dynamics, potential energy for Monte Carlo.
- `MaxBolt`: positions and a histogram of atom speeds.
- `RDF`: positions and the radial distribution function.
- `CellPlus`: positions and one plot of data you supply through `update(simulation, xdata, ydata)`.
- `Interactions`: positions, temperature, pressure and total energy.
- `Phase`: positions, total energy, mean squared displacement and the radial distribution function.
- `Scattering`: positions, the radial distribution function, mean squared displacement and the scattering profile.

`MaxBolt`, `Interactions`, `Phase` and `Scattering` plot quantities only a molecular dynamics run records and raise `ValueError` for a Monte Carlo simulation. Every viewer takes the simulation, an optional `size` of `'small'`, `'medium'` or `'large'`, and an optional `diameter` to draw the atoms at, in Angstrom; `CellPlus` also takes the axis labels of its plot. `average(simulation)` replaces the latest curve with the mean over the trajectory, the frames `sample()` has recorded, on the radial distribution function and scattering panes; the other panes it leaves alone. Axes are in Angstrom, picoseconds and otherwise SI units.

A viewer is for watching a run. For a plot to keep, take the arrays from `simulation.samples` or `simulation.trajectory` and use matplotlib.

Panes that plot a quantity against time read it from `simulation.samples`, so the loop must call `sample()` for those panes to have data. A molecular dynamics pane plots against time; a Monte Carlo pane plots against the step.

## Composing a viewer

`Viewer` takes a list of one, two or four panes:

```{code-cell} python
from pylj.md import MDSimulation
from pylj.model import Model
from pylj.potentials import LennardJones, Species
from pylj.sample import CellPane, TemperaturePane, Viewer

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
model = Model.single(argon, lj)
simulation = MDSimulation.initialise(model, number_of_atoms=16, temperature=300, box=30, seed=0)
viewer = Viewer(simulation, [CellPane(), TemperaturePane()])
for _ in range(300):
    simulation.step()
    if simulation.steps % 10 == 0:
        simulation.sample()
    if simulation.steps % 100 == 0:
        viewer.update(simulation)
```

The panes are `CellPane(diameter=None)`, `EnergyPane`, `TemperaturePane`, `PressurePane`, `MSDPane`, `RDFPane`, `ScatteringPane`, `MaxwellBoltzmannPane` and `CustomPane(xlabel, ylabel)`. A composed viewer takes the drawn diameter on its `CellPane`.

## Writing a pane

A pane has `setup(ax, simulation)`, which creates the line and labels once, and `update(ax, simulation)`, which sets the line's data from the current state. `needs_md = True` makes a viewer refuse a Monte Carlo simulation. A pane whose curve has a mean over the trajectory overrides `average(ax, simulation)`; `RDFPane` is the model.

```{code-cell} python
from pylj.sample import Pane


class FirstAtomPane(Pane):
    needs_md = True

    def __init__(self):
        self.times = []
        self.velocities = []

    def setup(self, ax, simulation):
        ax.plot([], [])
        ax.set_xlabel("Time / ps")
        ax.set_ylabel("x velocity / m s$^{-1}$")

    def update(self, ax, simulation):
        self.times.append(simulation.time * 1e12)
        self.velocities.append(simulation.configuration.velocity[0, 0])
        ax.lines[0].set_data(self.times, self.velocities)
        ax.relim()
        ax.autoscale_view()


simulation = MDSimulation.initialise(model, number_of_atoms=16, temperature=300, box=30, seed=0)
viewer = Viewer(simulation, [CellPane(), FirstAtomPane()])
for _ in range(300):
    simulation.step()
    if simulation.steps % 10 == 0:
        viewer.update(simulation)
```

A pane that needs a quantity the simulation samples, rather than one it can compute from the configuration, needs that quantity added to the simulation's `sample()` and its `Samples` record.

## A plot from the trajectory

`sample()` records the configuration each time it is called, so g(r) and the scattering profile can be computed after the run, over whichever frames are wanted:

```{code-cell} python
simulation = MDSimulation.initialise(model, number_of_atoms=25, temperature=100, box=30, seed=1)
for _ in range(2000):
    simulation.step()
    simulation.heat_bath(100)
    if simulation.steps % 10 == 0:
        simulation.sample()

r, gr = simulation.trajectory[50:].rdf()
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(r * 1e10, gr)
ax.set_xlabel("r / Angstrom")
ax.set_ylabel("g(r)")
```

`simulation.trajectory[50:]` drops the first fifty frames, the equilibration. [Simulations](simulations.md) describes the trajectory and the analyses on it.
