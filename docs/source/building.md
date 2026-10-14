---
file_format: mystnb
kernelspec:
  name: python3
---

# Building a simulation

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
%config InlineBackend.figure_format = "retina"
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

A simulation is built from a model and a starting configuration. The model names the species and the pair potential between each pair of them, and the configuration places the atoms in a periodic box. `initialise` builds the configuration from the model and returns a simulation ready to run. The examples on this page use the argon model from [Getting started](getting-started.md):

```{code-cell} python
from pylj.md import MDSimulation
from pylj.model import Model
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
model = Model.single(argon, LennardJones(epsilon=0.9497, sigma=3.372))
```

`MDSimulation` and `MCSimulation` share a base class, `Simulation`, and are built the same way. Both hold `configuration`, `model`, `cut_off`, `rng`, `steps`, `samples` and `trajectory`, and both have `step()`, `sample()` and `restart()`.

## `initialise`

```{code-cell} python
simulation = MDSimulation.initialise(
    model,
    number_of_atoms=16,
    temperature=300,
    box=30,
    init_conf="square",
    placement_temperature=None,
    max_strain=0.05,
    timestep=0.01,
    cut_off=None,
    seed=None,
)
```

`model`, a `Model` of the species and the potential between each pair of them, is the one positional argument; the rest are given by keyword. Three are required: `number_of_atoms`; `temperature`, in kelvin; and `box`, the side of the square periodic box in Angstrom. `init_conf` selects the starting positions. `"square"` places the atoms on a grid of columns and rows that fills the box, which is square when the number of atoms is a square number. `"triangular"` places them on a triangular lattice, and `"metropolis"` inserts them one at a time by Metropolis acceptance at `placement_temperature`, which defaults to `temperature`. `max_strain` applies only to the triangular lattice, described below. `timestep` is in picoseconds and applies to molecular dynamics only. `cut_off` is in Angstrom and defaults to 15 or half the box, whichever is smaller; it may not exceed half the box. `seed` seeds `simulation.rng`, which draws the initial velocities and the Monte Carlo moves.

The constructors take a ready configuration instead: `MDSimulation(configuration, model, *, cut_off=None, timestep=0.01, seed=None)` with an `MDConfiguration`, and `MCSimulation(configuration, model, temperature, *, cut_off=None, max_displacement=0.5, seed=None)` with a `Configuration`. Each simulation runs on its own copy of the configuration it is given; `MDSimulation` brings the copy's centre of mass to rest.

### A triangular lattice

A triangular lattice fills the box, so the number of atoms has to be a number of columns times a number of rows. Each row is offset along x by half a column spacing from the row below, so the number of rows has to be even: with an odd number, the last row and the first would meet unoffset across the periodic boundary.

The rows of a triangular lattice sit `sqrt(3) / 2` of a column spacing apart, so the columns divided by the rows should be `sqrt(3) / 2`, which a square box rarely allows exactly. The strain is how far that ratio sits from `sqrt(3) / 2`, as a fraction of it, and it splits the six neighbours of each atom into two distances that differ by about three quarters of the strain. `max_strain` is the largest strain accepted: 0.05 by default, and at most `placement.MOST_STRAIN`, a third.

At the default the counts up to 300 that fit are 30, 56, 90, 120, 168, 224, 270 and 288. Any other count raises `ValueError`, naming the nearest counts that fit.

The three starts look like this for 30 atoms at 100 K:

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, init_conf in zip(axes, ["square", "triangular", "metropolis"]):
    start = MDSimulation.initialise(
        model, number_of_atoms=30, temperature=100, box=25, init_conf=init_conf, seed=1
    )
    start.view(ax=ax)
    ax.set_title(init_conf)
plt.show()
```

## The configuration

`simulation.configuration` is the current state. `positions` is an `(N, 2)` array in Angstrom, `box` the side in Angstrom, `species` and `species_index` name each atom's species, and `masses` is in atomic mass units. `pairs(model, cut_off, forces=False)` evaluates every pair and returns their distances, separations, energies and, if asked, radial forces; `potential_energy`, `forces` and `virial` take the same arguments. An `MDConfiguration` adds `velocities` and `images`, the number of times each atom has crossed the box along x and along y. Its `unwrapped` positions, `positions + images * box`, are the positions without periodic wrapping. It also has `kinetic_energy()` and `temperature()`, which divides the kinetic energy by $(N - 1) k_B$ because the centre of mass is held at rest.

A configuration's arrays can be changed, and another configuration can be assigned to `simulation.configuration`. Before its next step, the simulation recomputes its `forces` or `energy` if the positions, species, box, model or cut-off have changed. If the number of atoms or the box has changed, it raises `ValueError` instead; to run such a configuration, build a new simulation from it, such as from `configuration.without(index)`. `copy()` returns an independent copy of a configuration.
