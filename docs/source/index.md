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
for _ in range(200):
    simulation.step()
    if simulation.steps % 50 == 0:
        viewer.update(simulation)
print(simulation.steps)
```

```{toctree}
:hidden:

using-pylj
byof
visualisation
modules
```
