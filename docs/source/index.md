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

pylj runs two-dimensional simulations of a few tens of atoms, and draws them as they move. It runs molecular dynamics and Monte Carlo in a periodic box and shows the temperature, pressure, energy and structure as they are measured. It exists to teach how these methods work, so its code is short, in plain Python and NumPy, and written to be read. These pages explain the two methods and show, in pylj's own code, how each step is done.

pylj is for undergraduate chemistry and physics students meeting simulation for the first time, and for the lecturers who set them work. Every figure and printed number on these pages was produced by the code shown above it when the pages were built.

## Installing pylj

pylj runs inside a Jupyter notebook, because the figures redraw in place as a simulation runs. If you do not have Jupyter, `pip install jupyterlab` provides it. `pip install pylj` installs the latest release. This book describes the version on GitHub, which names `MDSimulation` on its first page; if the release you get does not have it, install from GitHub instead:

```bash
pip install git+https://github.com/arm61/pylj
```

Start each notebook with `%matplotlib inline`, which selects the figure backend the viewers draw through.

## A first simulation

The first simulation is sixteen argon atoms in a box 30 Angstrom on a side, where an Angstrom is $10^{-10}$ m, about the size of an atom, at 300 K, interacting through a Lennard-Jones potential. A few lines build the simulation and one loop runs it, redrawing the box every fifty steps.

```{code-cell} python
from pylj import sample
from pylj.md import MDSimulation
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
simulation = MDSimulation.initialise(
    number_of_atoms=16,
    temperature=300,
    box=30,
    species=[argon],
    pair_potentials={(argon, argon): lj},
    seed=1,
)
viewer = sample.JustCell(simulation)
for _ in range(500):
    simulation.step()
    if simulation.steps % 50 == 0:
        viewer.update(simulation)
```

In a notebook the figure redraws ten times; on this page it shows the last frame. The atoms started on a square lattice and have spread out. Two things happened on the way: `step()` moved every atom forward by one timestep of ten femtoseconds, and atoms that left one side of the box came back in on the other, because the box is periodic.

`Species` gives the atoms a mass, in atomic mass units, and a name. `LennardJones` is the pair potential, with the well depth in joules and the zero-crossing separation in metres; these are the standard values for argon. {class}`~pylj.md.MDSimulation`'s `initialise` takes keyword arguments for the number of atoms, the temperature in kelvin and the box side in Angstrom, then the species and the potential acting between each pair of species. Two units conventions meet here. `initialise` and the viewers take lengths in Angstrom: the box, the cut-off and a drawn diameter. Potentials take SI units, joules and metres, and `Species` takes the mass in atomic mass units. Everything a simulation reports back is in SI units.

## What is in these pages

*Atoms and potentials* introduces what every simulation contains: atoms, the potential energy between them, and the box. *Molecular dynamics* and *Monte Carlo* explain the two methods, writing each step by hand before handing it to pylj. *The ideal gas law* uses molecular dynamics to test the law, and *The ideal gas law from first principles* derives it from the partition function. *Bring your own potential* and *Viewers and panes* show how to extend pylj with your own potential and your own plots. The reference at the end lists every class, function and argument, with units.

The first six chapters are for students, and each is one notebook to run from top to bottom: later cells use names defined in earlier ones. The two chapters on extending pylj are for readers who want to write their own potential or plot, and each needs a short Python class. *Teaching with pylj* is for lecturers: what pylj needs, what it can and cannot do, and which parameters are safe to vary.

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
viewers
```

```{toctree}
:hidden:
:caption: For lecturers

teaching
```

```{toctree}
:hidden:
:caption: Reference

modules
changelog
```
