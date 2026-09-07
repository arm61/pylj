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
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
model = dict(species=[argon], pair_potentials={(argon, argon): lj})
```

`Species` takes the mass in atomic mass units. `LennardJones` takes the well depth in joules and the zero-crossing separation in metres. `pair_potentials` maps each pair of species to the potential acting between them.

## Molecular dynamics

```{code-cell} python
from pylj import sample
from pylj.md import MDSimulation

simulation = MDSimulation.initialise(number_of_atoms=16, temperature=300, box=30, seed=1, **model)
viewer = sample.Interactions(simulation)
for _ in range(2000):
    simulation.step()
    simulation.heat_bath(300)
    simulation.sample()
    if simulation.steps % 100 == 0:
        viewer.update(simulation)
```

`initialise` takes the number of atoms, the temperature in kelvin and the box side in Angstrom. `step()` advances one timestep, `heat_bath()` holds the temperature, `sample()` records the temperature, pressure and energies in `simulation.samples`, and the viewer redraws when asked.

## Monte Carlo

```{code-cell} python
from pylj.mc import MCSimulation

simulation = MCSimulation.initialise(number_of_atoms=16, temperature=300, box=20, seed=1, **model)
viewer = sample.Energy(simulation)
for _ in range(5000):
    simulation.step()
    if simulation.steps % 10 == 0:
        simulation.sample()
    if simulation.steps % 500 == 0:
        viewer.update(simulation)
```

`step()` makes one Metropolis move, and `sample()` records the potential energy. [Simulations](simulations.md) describes both classes in full, [Custom potentials](custom-potentials.md) how to add a potential, and [Viewers and panes](viewers.md) the viewers.
