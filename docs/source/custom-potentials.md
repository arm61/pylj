---
file_format: mystnb
kernelspec:
  name: python3
---

# Custom potentials

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
%config InlineBackend.figure_format = "retina"
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

Custom pair potentials can be defined in a few lines of Python. A potential is a subclass of `PairPotential` with two methods:

```{code-cell} python
import numpy as np
from pylj.potentials import PairPotential


class SoftSphere(PairPotential):
    def __init__(self, *, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma

    def energies(self, dr):
        dr = np.asarray(dr, dtype=float)
        return self.epsilon * (self.sigma / dr) ** 12

    def forces(self, dr):
        dr = np.asarray(dr, dtype=float)
        return 12 * self.epsilon * (self.sigma / dr) ** 12 / dr
```

`energies(dr)` takes an array of pair separations in metres and returns the pair energy of each in joules. `forces(dr)` returns the radial force in newtons, minus the derivative of the energy with respect to the separation, so positive where the pair repels and negative where it attracts. A potential with no finite force, such as `SquareWell`, raises `ValueError` from `forces` and can drive only Monte Carlo. The constructor takes whatever parameters the potential needs; the built-in potentials use keyword-only parameters named after the physical quantities.

`min_separation` is the separation below which the formula gives unphysical energies. It is a class attribute and defaults to `0.0`, meaning the formula is physical at every separation. A configuration gives a pair closer than it infinite energy, so placement and Monte Carlo never accept such a pair, and raises `ValueError` if asked for its forces. `Buckingham` sets it to the top of its short-range barrier in its constructor.

A purely repulsive potential has no energy minimum for a viewer to size the atoms by, so the viewer is given a `diameter` in Angstrom:

```{code-cell} python
from pylj import sample
from pylj.md import MDSimulation
from pylj.model import Model
from pylj.potentials import Species

argon = Species(mass=39.948, name="argon")
soft = SoftSphere(epsilon=1.577e-21, sigma=3.372e-10)
model = Model.single(argon, soft)
simulation = MDSimulation.initialise(model, number_of_atoms=25, temperature=300, box=30, seed=0)
viewer = sample.JustCell(simulation, diameter=3.4)
for _ in range(300):
    simulation.step()
    if simulation.steps % 100 == 0:
        viewer.update(simulation)
```

## Mixtures

A mixture is a `Model` with more species and one entry for each species with itself and for each pair of different species, in either order:

```python
from pylj.potentials import LennardJones

xenon = Species(mass=131.293, name="xenon")
lj_argon = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
lj_xenon = LennardJones(epsilon=3.05e-21, sigma=3.98e-10)
lj_cross = LennardJones(epsilon=2.19e-21, sigma=3.68e-10)
mixture = Model(
    (argon, xenon),
    {(argon, argon): lj_argon, (xenon, xenon): lj_xenon, (argon, xenon): lj_cross},
)
simulation = MDSimulation.initialise(mixture, number_of_atoms=24, temperature=200, box=40, seed=0)
```

Atoms are assigned to the species in turn. A missing pair raises `ValueError` and a potential class in place of an instance raises `TypeError`, both when the `Model` is built. The viewers draw each species at the minimum of its own pair potential unless `diameter` is given, as one value or one per species.
