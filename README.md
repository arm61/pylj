<a href="https://pylj.readthedocs.io/"><img src="https://github.com/arm61/pylj/blob/master/logo/logo.png?raw=true" width="60%"/></a>

[![JOSE](http://jose.theoj.org/papers/58daa1a1a564dc8e0f99ffcdae20eb1d/status.svg)](http://jose.theoj.org/papers/58daa1a1a564dc8e0f99ffcdae20eb1d)
[![PyPI](https://badge.fury.io/py/pylj.svg)](https://badge.fury.io/py/pylj)
[![DOI](https://zenodo.org/badge/119863480.svg)](https://zenodo.org/badge/latestdoi/119863480)
[![Documentation](https://readthedocs.org/projects/pylj/badge/?version=latest)](https://pylj.readthedocs.io/en/latest/)
[![Build](https://github.com/arm61/pylj/actions/workflows/ci.yml/badge.svg)](https://github.com/arm61/pylj/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`pylj` runs molecular dynamics and Metropolis Monte Carlo simulations of atoms interacting through pair potentials, in two dimensions, and draws them as they run. It is a teaching code, written in Python and NumPy. Its viewers redraw in a Jupyter notebook; in a script, a viewer's `fig` is a matplotlib figure that can be saved.

## Installation

```bash
pip install pylj
```

`pylj` needs Python 3.11 or later. To watch the figures redraw you need a Jupyter notebook; `pip install jupyterlab` provides one. Start each notebook with `%matplotlib inline`.

## Example

Twenty-five argon atoms at 300 K, run with molecular dynamics and drawn as they go:

```python
from pylj import sample
from pylj.md import MDSimulation
from pylj.model import Model
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
model = Model.single(argon, lj)
simulation = MDSimulation.initialise(model, number_of_atoms=25, temperature=300, box=40, seed=1)
viewer = sample.Interactions(simulation)
for _ in range(2000):
    simulation.step()
    simulation.heat_bath(300)
    simulation.sample()
    if simulation.steps % 100 == 0:
        viewer.update(simulation)
```

<img src="https://github.com/arm61/pylj/blob/master/docs/readme/interactions.png?raw=true" width="70%"/>

`step()` advances one timestep, `heat_bath()` holds the temperature, and `sample()` records the temperature, pressure and energies in `simulation.samples` as NumPy arrays:

```python
simulation.samples.temperature.mean()  # K
simulation.samples.pressure.mean()     # N/m, the two-dimensional pressure
```

The same model runs under Monte Carlo:

```python
from pylj.mc import MCSimulation

simulation = MCSimulation.initialise(model, number_of_atoms=25, temperature=300, box=40, seed=1)
viewer = sample.Energy(simulation)
for _ in range(5000):
    simulation.step()
    if simulation.steps % 10 == 0:
        simulation.sample()
    if simulation.steps % 500 == 0:
        viewer.update(simulation)
```

Custom pair potentials are subclasses of `PairPotential` with an `energies` method and a `forces` method; custom plots are panes composed into a `Viewer`. The [documentation](https://pylj.readthedocs.io/) describes the simulation classes, custom potentials, the viewers, and every module.

## Contributing

Bug reports and pull requests are welcome on [GitHub](https://github.com/arm61/pylj/issues). [CONTRIBUTING.md](CONTRIBUTING.md) describes the development setup and the checks a pull request needs to pass.

## Citing pylj

McCluskey, A. R., Morgan, B. J., Edler, K. J., and Parker, S. C. (2018). pylj: A teaching tool for classical atomistic simulation. *Journal of Open Source Education*, 1(2), 19. https://doi.org/10.21105/jose.00019

`pylj.__cite__()` opens the paper in a browser.
