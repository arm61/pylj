# pylj

`pylj` runs molecular dynamics and Metropolis Monte Carlo simulations of atoms interacting through pair potentials, in two dimensions, and draws and animates the atoms. It is a teaching code, written in Python and NumPy.

## Installation

```bash
pip install --pre pylj
```

pylj 2.0 is a beta release, so pip needs `--pre` to install it. Without `--pre`, pip installs the previous release, 1.5.2.

`pylj` needs Python 3.12 or later.

## Contents

- [Getting started](getting-started.md): a model, a molecular dynamics run and a Monte Carlo run.
- [Building a simulation](building.md): `initialise`, the starting lattices, and the configuration.
- [Running a simulation](running.md): molecular dynamics, Monte Carlo, restarting, and how long a run takes.
- [Simulations](simulations.md): building a simulation, the configuration, the two `step()` methods, sampling, restarting, and pictures and animations.
- [Custom potentials](custom-potentials.md): the `PairPotential` interface and mixtures.
- [Modules](modules.rst): every class, function and argument, with units.
- [Changelog](changelog.md).

## Citing pylj

McCluskey, A. R., Morgan, B. J., Edler, K. J., and Parker, S. C. (2018). pylj: A teaching tool for classical atomistic simulation. *Journal of Open Source Education*, 1(2), 19. https://doi.org/10.21105/jose.00019

`pylj.__cite__()` opens the paper in a browser. Development is on [GitHub](https://github.com/arm61/pylj).

```{toctree}
:hidden:

getting-started
building
running
simulations
custom-potentials
```

```{toctree}
:hidden:
:caption: Reference

modules
changelog
```
