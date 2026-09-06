---
file_format: mystnb
kernelspec:
  name: python3
---

# pylj

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

pylj is a two-dimensional simulation of particles that you watch happen. It runs molecular dynamics and Monte Carlo on a few tens of particles in a periodic box, draws the box as the run goes on, and shows the temperature, pressure, energy and structure as they are measured. It exists to teach how these methods work, so its code is short, in plain Python and NumPy, and written to be read. This book explains the methods with pylj as the vehicle and shows the code that does each step.

pylj is for undergraduate chemistry and physics students meeting simulation for the first time, and for the lecturers who set them work. Everything on these pages was produced by the code shown next to it when the pages were built.

## Installing pylj

pylj runs inside a Jupyter notebook, because the figures redraw in place as a simulation runs. Install it with pip into an environment that has Jupyter:

```bash
pip install pylj
```

Start each notebook with `%matplotlib inline`, which is the figure backend the viewers use.

## A first simulation

Sixteen argon atoms in a box 30 Angstrom on a side, at 300 K, interacting through a Lennard-Jones potential. Five lines build the simulation and one loop runs it, redrawing the box every fifty steps.

```{code-cell} python
from pylj import sample
from pylj.md import MDSimulation
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
simulation = MDSimulation.initialise(
    16, 300, 30, species=[argon], pair_potentials={(argon, argon): lj}, seed=1
)
viewer = sample.JustCell(simulation)
for _ in range(500):
    simulation.step()
    if simulation.steps % 50 == 0:
        viewer.update(simulation)
```

In a notebook the figure redraws ten times; on this page it shows the last frame. The particles started on a square lattice and have spread out. Two things happened on the way: `step()` moved every particle forward by one timestep of ten femtoseconds, and particles that left one side of the box came back in on the other, because the box is periodic.

`Species` gives the particles a mass, in atomic mass units, and a name. `LennardJones` is the pair potential, with the well depth in joules and the zero-crossing separation in metres. `initialise` takes the number of particles, the temperature in kelvin and the box side in Angstrom, then the species and the potential acting between each pair of species. The box and the cut-off are the only lengths given in Angstrom; everything the simulation stores is in SI units.

## What is in this book

Chapter 2 sets up the pieces every simulation shares: particles, the pair potential and the periodic box. Chapters 3 and 4 build the two methods, molecular dynamics and Monte Carlo, from those pieces, writing each step by hand before handing it to pylj. Chapters 5 and 6 use molecular dynamics to test the ideal gas law and derive it from the partition function. Chapters 7 and 8 show how to extend pylj with your own potential and your own plots. The reference at the end documents every module.

pylj is developed on [GitHub](https://github.com/arm61/pylj). If you use it in teaching, please cite the [paper in the Journal of Open Source Education](http://jose.theoj.org/papers/58daa1a1a564dc8e0f99ffcdae20eb1d); `pylj.__cite__()` opens it.

```{toctree}
:hidden:
:caption: Getting started

particles-and-potentials
```

```{toctree}
:hidden:
:caption: The two methods

molecular-dynamics
monte-carlo
```

```{toctree}
:hidden:
:caption: Doing science with it

ideal-gas-law
first-principles
```

```{toctree}
:hidden:
:caption: Extending pylj

byof
visualisation
```

```{toctree}
:hidden:
:caption: Reference

modules
changelog
```
