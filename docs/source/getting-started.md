---
file_format: mystnb
kernelspec:
  name: python3
---

# Getting started

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
%config InlineBackend.figure_format = "retina"
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

## A model

```{code-cell} python
from pylj.model import Model
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=0.9497, sigma=3.372)
model = Model.single(argon, lj)
```

`Species` takes the mass in atomic mass units. `LennardJones` takes the well depth in kJ/mol and the zero-crossing separation in Angstrom. A `Model` is the species and the potential between each pair of them; `Model.single` builds it for one species.

## Molecular dynamics

```{code-cell} python
import matplotlib.pyplot as plt

from pylj.md import MDSimulation

simulation = MDSimulation.initialise(model, number_of_atoms=16, temperature=300, box=30, seed=1)
for _ in range(2000):
    simulation.step()
    if simulation.steps % 50 == 0:
        simulation.rescale_velocities(300)
    simulation.sample()
simulation.view()
plt.show()
```

`initialise` takes the model, the number of atoms, the temperature in kelvin and the box side in Angstrom. `step()` advances one timestep. `sample()` records the configuration in `simulation.trajectory` and the temperature, pressure and energies in `simulation.samples`. `view()` draws the atoms as the run left them. The samples are NumPy arrays, so they plot directly:

```{code-cell} python
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(simulation.samples.step * simulation.timestep, simulation.samples.temperature)
ax.set_xlabel("Time / ps")
ax.set_ylabel("Temperature / K")
plt.show()
```

## Monte Carlo

```{code-cell} python
from pylj.mc import MCSimulation

simulation = MCSimulation.initialise(model, number_of_atoms=16, temperature=300, box=20, seed=1)
for _ in range(5000):
    simulation.step()
    if simulation.steps % 10 == 0:
        simulation.sample()
simulation.view()
plt.show()
```

`step()` makes one Metropolis move, and `sample()` records the configuration and the potential energy. The pages that follow describe each stage in full, from [building a simulation](building.md) to [pictures and animations](pictures.md), and [Custom potentials](custom-potentials.md) describes how to add a potential.
