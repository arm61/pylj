<a href="https://pylj.readthedocs.io/"><img src="https://github.com/arm61/pylj/blob/master/logo/logo.png?raw=true" width="60%"/></a>

[![JOSE](http://jose.theoj.org/papers/58daa1a1a564dc8e0f99ffcdae20eb1d/status.svg)](http://jose.theoj.org/papers/58daa1a1a564dc8e0f99ffcdae20eb1d)
[![PyPI](https://badge.fury.io/py/pylj.svg)](https://badge.fury.io/py/pylj)
[![DOI](https://zenodo.org/badge/119863480.svg)](https://zenodo.org/badge/latestdoi/119863480)
[![Documentation](https://readthedocs.org/projects/pylj/badge/?version=latest)](https://pylj.readthedocs.io/en/latest/)
[![Build](https://github.com/arm61/pylj/actions/workflows/ci.yml/badge.svg)](https://github.com/arm61/pylj/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`pylj` runs molecular dynamics and Metropolis Monte Carlo simulations of atoms interacting through pair potentials, in two dimensions, and draws and animates the atoms. It is a teaching code, written in Python and NumPy.

## Installation

```bash
pip install --pre pylj
```

pylj 2.0 is a beta release, so pip needs `--pre` to install it. Without `--pre`, pip installs the previous release, 1.5.2.

`pylj` needs Python 3.12 or later.

## Example

Twenty-five argon atoms at 300 K, run with molecular dynamics and then drawn:

```python
import matplotlib.pyplot as plt

from pylj.md import MDSimulation
from pylj.model import Model
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=0.9497, sigma=3.372)
model = Model.single(argon, lj)
simulation = MDSimulation.initialise(model, number_of_atoms=25, temperature=300, box=40, seed=1)
for _ in range(2000):
    simulation.step()
    if simulation.steps % 50 == 0:
        simulation.rescale_velocities(300)
    simulation.sample()
simulation.view()
plt.show()
```

<img src="https://github.com/arm61/pylj/blob/master/docs/readme/atoms.png?raw=true" width="40%"/>

`step()` advances one timestep. `sample()` records the configuration in `simulation.trajectory` and the temperature, pressure and energies in `simulation.samples` as NumPy arrays:

```python
simulation.samples.temperature.mean()  # K
simulation.samples.pressure.mean()     # kJ/mol/Angstrom^2, the two-dimensional pressure
```

In a Jupyter notebook, `simulation.animate()` plays the recorded configurations as an animation. In a script, save `simulation.animate().data` to an `.html` file and open it in a web browser.

The same model runs under Monte Carlo:

```python
from pylj.mc import MCSimulation

simulation = MCSimulation.initialise(model, number_of_atoms=25, temperature=300, box=40, seed=1)
for _ in range(5000):
    simulation.step()
    if simulation.steps % 10 == 0:
        simulation.sample()
simulation.view()
plt.show()
```

Custom pair potentials are subclasses of `PairPotential` with an `energies` method and a `forces` method. The [documentation](https://pylj.readthedocs.io/) describes building, running and sampling a simulation, pictures and animations, custom potentials, units, and every module.

## Contributing

Bug reports and pull requests are welcome on [GitHub](https://github.com/arm61/pylj/issues). [CONTRIBUTING.md](CONTRIBUTING.md) describes the development setup and the checks a pull request needs to pass.

## Citing pylj

McCluskey, A. R., Morgan, B. J., Edler, K. J., and Parker, S. C. (2018). pylj: A teaching tool for classical atomistic simulation. *Journal of Open Source Education*, 1(2), 19. https://doi.org/10.21105/jose.00019

`pylj.__cite__()` opens the paper in a browser.
