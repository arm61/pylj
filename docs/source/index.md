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

`pylj` is code for running molecular dynamics and Metropolis Monte Carlo simulations of simple systems in two dimensions. It draws the atoms as they move, and reports the temperature, the pressure, the energy and how the atoms are arranged, measuring each as the simulation runs. It exists to teach how these two methods work, so its code is short, written in plain Python and NumPy, and meant to be read.

These pages are for undergraduate chemistry and physics students meeting simulation for the first time, and for the lecturers who set them work. They explain the two methods and show, in `pylj`'s own code, how each step is done. Every figure and printed number on these pages was produced by the code shown above it when the pages were built.

## Installing pylj

`pylj` runs inside a Jupyter notebook, because its figures redraw in place as a simulation runs. If you do not have Jupyter, `pip install jupyterlab` provides it. `pip install pylj` installs the latest release. These pages describe the version on GitHub; if the first example below fails with an import error, the release is older than these pages, and you should install from GitHub instead:

```bash
pip install git+https://github.com/arm61/pylj
```

Start each notebook with the line `%matplotlib inline`, which makes the figures appear in the notebook.

## A first look

Sixteen argon atoms, in a box thirty Angstrom on a side, at 300 K:

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

The atoms start on a grid, and the loop moves them forward in time five hundred times, redrawing the picture every fifty steps. In a notebook the picture changes as the loop runs; on this page it shows the end. Every name in the cell is explained in the next three chapters: what `Species` and `LennardJones` describe, what `initialise` builds, what one `step()` does, and what the viewer draws.

## Finding your way

*Atoms and potentials* introduces what every simulation contains: the atoms, the potential energy of a pair of atoms as a function of their separation, and the box. *Molecular dynamics* and *Monte Carlo* explain the two methods, writing each step out first and then showing the same step in `pylj`'s code. *The ideal gas law* uses molecular dynamics to test the law, and *The ideal gas law from first principles* derives it. *Bring your own potential* and *Viewers and panes* show how to extend `pylj` with your own potential and your own plots. The reference at the end lists every class, function and argument, with units.

The first six chapters are for students, and each is one notebook to run from top to bottom: later cells use names defined in earlier ones. The two chapters on extending `pylj` are for readers who want to write their own potential or plot, and each needs a short Python class. *Teaching with pylj* is for lecturers: what `pylj` needs, what it can and cannot do, and which parameters are safe to vary.

`pylj` is developed on [GitHub](https://github.com/arm61/pylj). If you use it in teaching, please cite the [paper in the Journal of Open Source Education](http://jose.theoj.org/papers/58daa1a1a564dc8e0f99ffcdae20eb1d); `pylj.__cite__()` opens it.

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
