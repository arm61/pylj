---
file_format: mystnb
kernelspec:
  name: python3
---

# Bring your own potential

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

A pylj simulation is built from the species it contains and the pair potential acting between each pair of species:

```{code-cell} python
from pylj import sample
from pylj.md import MDSimulation
from pylj.potentials import Species, LennardJones

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)

simulation = MDSimulation.initialise(
    100, 300, 40,
    species=[argon],
    pair_potentials={(argon, argon): lj},
    seed=0,
)
viewer = sample.JustCell(simulation)
```

The Lennard-Jones, Buckingham and square-well potentials in the {doc}`potentials` module are subclasses of `PairPotential`, and a custom potential follows the same form:

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

The two methods take an array of pair separations `dr`, in metres, and return an array of the same shape.

- `energies` returns the pair energy at each separation, in joules.
- `forces` returns the radial force at each separation, in newtons: minus the derivative of the energy with respect to the separation, so it is positive where the interaction is repulsive and negative where it is attractive. A potential with no finite force, such as the square well, raises `ValueError` here; it can still drive Monte Carlo, which evaluates the energies only.

A purely repulsive potential such as this one has no energy minimum for the viewers to size the particles by, so a viewer of such a simulation takes an explicit `diameter=`, in Angstrom.

```{code-cell} python
soft = SoftSphere(epsilon=1.577e-21, sigma=3.372e-10)
simulation = MDSimulation.initialise(
    25, 300, 30, species=[argon], pair_potentials={(argon, argon): soft}, seed=0
)
viewer = sample.JustCell(simulation, diameter=3.4)
for _ in range(300):
    simulation.step()
    if simulation.steps % 100 == 0:
        viewer.update(simulation)
```

If the formula stops being physical below some separation, as the Buckingham potential's does inside its short-range barrier, set `min_separation` to that separation: a simulation then treats any pair closer than it as forbidden, and the formula itself is left as it is.

The constructor is yours to define. Keyword-only parameters named after the physical quantities, as above, mean a swapped pair of numbers is an error rather than a silently wrong model.

A mixture is more species and more entries in `pair_potentials`: one for each species with itself and one for each pair of different species, in either order.

Argon and xenon, with a cross pair whose parameters are the means of the two, and the particles drawn at their own sizes:

```{code-cell} python
heavy = Species(mass=131.293, name="xenon")
xenon = LennardJones(epsilon=3.05e-21, sigma=3.98e-10)
cross = LennardJones(epsilon=2.19e-21, sigma=3.68e-10)
mixture = MDSimulation.initialise(
    24, 200, 40,
    species=[argon, heavy],
    pair_potentials={(argon, argon): lj, (heavy, heavy): xenon, (argon, heavy): cross},
    seed=0,
)
viewer = sample.JustCell(mixture)
for _ in range(300):
    mixture.step()
    if mixture.steps % 100 == 0:
        viewer.update(mixture)
```
