---
file_format: mystnb
kernelspec:
  name: python3
---

# Particles and potentials

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

A pylj simulation is a set of particles in a periodic box, with a rule for the energy between every pair. This chapter builds those three pieces and reads the Lennard-Jones potential for argon, which every later chapter uses.

## Species

Each particle belongs to a species: a mass, in atomic mass units, and a name. Argon is one species. A mixture is two or more.

```{code-cell} python
from pylj.potentials import Species

argon = Species(mass=39.948, name="argon")
argon
```

Masses go in as atomic mass units because that is the number in a data book. Once the particles are in a simulation, their masses are stored in kilograms.

## The pair potential

Two argon atoms attract each other weakly at a few Angstrom, through the London dispersion interaction, and repel each other strongly when their electron clouds overlap. The Lennard-Jones potential is the simplest formula with both features:

$$
E(r) = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right].
$$

The two parameters have a direct meaning. The well depth $\epsilon$ is how strongly a pair binds at its best separation, and $\sigma$ is the separation at which the energy crosses zero, which is a little less than the size of the atom. For argon they are

```{code-cell} python
from pylj.potentials import LennardJones

lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
```

in joules and metres. Keyword-only arguments mean a swapped pair of numbers is an error rather than a silently wrong model.

A potential has two methods, `energies` and `forces`. Both take a separation in metres, or an array of them, and return a value of the same shape. `forces` returns minus the derivative of the energy with respect to the separation, so it is positive where the pair repels and negative where it attracts.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(3e-10, 8e-10, 200)
fig, (energy_ax, force_ax) = plt.subplots(1, 2, figsize=(8, 3.2))
energy_ax.plot(r * 1e10, lj.energies(r) * 1e21)
energy_ax.axhline(0, color="grey", linewidth=0.5)
energy_ax.set_xlabel("r / Angstrom")
energy_ax.set_ylabel("E / zJ")
force_ax.plot(r * 1e10, lj.forces(r) * 1e12)
force_ax.set_ylim(-15, 60)
force_ax.axhline(0, color="grey", linewidth=0.5)
force_ax.set_xlabel("r / Angstrom")
force_ax.set_ylabel("f / pN")
fig.tight_layout()
```

Reading the two curves from left to right: below 3.37 Angstrom the energy is positive and rises steeply, and the force is positive, so the pair is pushed apart. The repulsive force grows so fast that the plot cuts it off at 60 pN; at 3 Angstrom it is nearly 800 pN. The energy crosses zero at $\sigma$ and reaches its minimum of $-\epsilon$ where the force is zero:

```{code-cell} python
r_min = 2 ** (1 / 6) * lj.sigma
print(f"minimum at {r_min * 1e10:.2f} Angstrom, energy {lj.energies(r_min) * 1e21:.3f} zJ")
print(f"force there {lj.forces(r_min) * 1e12:.3f} pN")
```

Beyond the minimum the force is negative, pulling the pair together, and it fades to nothing by about 8 Angstrom. The well depth of 1.58 zeptojoules is about 0.4 $k_B T$ at room temperature and 1.3 $k_B T$ at 87 K, which is why argon is a gas at room temperature and boils at 87 K.

The Lennard-Jones potential is the model for a rare gas. Two other forms are available for other physics: `Buckingham`, an exponential repulsion with the same $r^{-6}$ attraction, and `SquareWell`, a hard core surrounded by a well of constant depth. The square well has no finite force, so only Monte Carlo can use it; the Monte Carlo chapter does.

## The box

The particles live in a square box with periodic boundaries. A particle that leaves through the right-hand edge comes back in through the left, and a pair near opposite edges interacts across the boundary as if they were neighbours. The box is therefore a small piece of an endless, repeating system, with no walls and no surface. The rule for the distance between two particles is to take the nearest of the periodic copies, the minimum image.

Beyond a cut-off separation the pair energy is set to zero, because the Lennard-Jones interaction has faded to nothing by then. pylj's default cut-off is 15 Angstrom, or half the box if the box is smaller than 30 Angstrom. Half the box is the largest cut-off the minimum image rule allows. Beyond it, two periodic copies of the same neighbour could both lie inside the cut-off, and the rule counts only the nearer one.

A box between 4 and 600 Angstrom is accepted: a smaller box holds at most one particle, and in a larger one the particles are too small to see.

```{code-cell} python
from pylj.md import MDSimulation

model = dict(species=[argon], pair_potentials={(argon, argon): lj})
simulation = MDSimulation.initialise(9, 300, 20, seed=0, **model)
configuration = simulation.configuration
print(f"box {configuration.box * 1e10:.0f} Angstrom, cut-off {simulation.cut_off * 1e10:.0f} Angstrom")
print(configuration.position[:3] * 1e10)
```

`pair_potentials` names the potential acting between each pair of species. For one species that is a single entry; a mixture of argon and a heavier species needs three, one for each species with itself and one for the cross pair.

## Every pair at once

The configuration can evaluate every pair under the potential. `pairs` returns the minimum-image distance of each pair, its separation vector, its energy and, on request, its radial force. The pairs are ordered by the lower particle index and then the higher, so the first entries are particle 0 with each of the others. Nine particles make thirty-six pairs.

```{code-cell} python
pairs = configuration.pairs(simulation.pair_potentials, simulation.cut_off, forces=True)
print(pairs.distance.size, "pairs")
print(f"nearest pair {pairs.distance.min() * 1e10:.2f} Angstrom")
print(f"total energy {pairs.energy.sum() * 1e21:.2f} zJ")
```

Summing the pair energies gives the potential energy of the configuration; summing the pair forces onto the particles gives the net force on each. The molecular dynamics chapter turns those forces into motion, and the Monte Carlo chapter uses the energies alone.
