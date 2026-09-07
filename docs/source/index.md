# pylj

`pylj` runs molecular dynamics and Metropolis Monte Carlo simulations of atoms interacting through pair potentials, in two dimensions, and draws them as they run. It is written for teaching, in Python and NumPy.

## Installation

```bash
pip install pylj
```

`pylj` needs Python 3.11 or later and draws its figures inside a Jupyter notebook. Start each notebook with `%matplotlib inline`.

## Contents

- [Getting started](getting-started.md): a model, a molecular dynamics run and a Monte Carlo run.
- [Simulations](simulations.md): building a simulation, the configuration, the two `step()` methods, sampling and restarting.
- [Custom potentials](custom-potentials.md): the `PairPotential` interface and mixtures.
- [Viewers and panes](viewers.md): the viewers, and writing a pane.
- [Modules](modules.rst): every class, function and argument, with units.
- [Changelog](changelog.md).

## Citing pylj

McCluskey, A. R., Morgan, B. J., Edler, K. J., and Parker, S. C. (2018). pylj: A teaching tool for classical atomistic simulation. *Journal of Open Source Education*, 1(2), 19. https://doi.org/10.21105/jose.00019

`pylj.__cite__()` opens the paper in a browser. Development is on [GitHub](https://github.com/arm61/pylj).

```{toctree}
:hidden:

getting-started
simulations
custom-potentials
viewers
```

```{toctree}
:hidden:
:caption: Reference

modules
changelog
```
