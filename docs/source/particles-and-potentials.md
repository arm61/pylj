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

pylj simulates a small number of atoms moving in two dimensions. The atoms in these chapters are argon. Argon atoms interact only weakly, so their behaviour is simple enough to follow, and yet a simulation of them shows the things a simulation measures: how fast the atoms move, how they arrange themselves, and what pressure they exert. Before the two simulation methods are introduced, this chapter sets out what a simulation needs to know about its atoms, how heavy they are, how they interact and where they are confined, then builds a simulation from those three things and adds up the energy of the atoms in it.

## The atoms

pylj describes an atom by two things, its mass and a name. The mass is given in atomic mass units, the unit used in data tables, so for argon it is 39.948:

```{code-cell} python
from pylj.potentials import Species

argon = Species(mass=39.948, name="argon")
argon
```

pylj calls this description a species, because one description serves every atom of the same kind. All the atoms in a simulation of argon share this one species. A simulation of a mixture of argon and xenon would have two. pylj's code calls the atoms particles, since it does not care whether they are atoms or something else.

## How the atoms interact

Two argon atoms attract each other weakly when they are a few Angstrom apart, through the dispersion interaction, and repel each other strongly when they come close enough for their electron clouds to overlap. The potential energy of the pair therefore falls as the atoms approach from far away, reaches a minimum, the well, and then rises steeply. The Lennard-Jones potential is the simplest formula with this shape:

$$
E(r) = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right].
$$

Here $r$ is the distance between the two atoms. The well depth $\epsilon$ is the energy at the bottom of the well, which is the energy needed to pull a bound pair apart. The length $\sigma$ is the distance at which the energy passes through zero, a little less than the diameter of the atom. For argon, $\epsilon$ is 1.577 zJ, where a zeptojoule is $10^{-21}$ J, and $\sigma$ is 3.372 Angstrom.

In pylj the formula is represented by an object. `LennardJones` takes $\epsilon$ in joules and $\sigma$ in metres:

```{code-cell} python
from pylj.potentials import LennardJones

lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
```

The object evaluates the formula at any distance, or at a whole array of distances at once, through its `energies` method. Its `forces` method gives the force between the two atoms, which is minus the slope of the energy: positive where the atoms repel and negative where they attract. Plotting both from 3 to 8 Angstrom:

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

The left-hand curve is the potential energy and the right-hand curve the force, in piconewtons, where a piconewton is $10^{-12}$ N. Below 3.37 Angstrom the energy rises steeply as the atoms are pushed together, and the force is large and repulsive; at 3 Angstrom it is nearly 800 pN, far above the top of the plot. The energy passes through zero at $\sigma$ and reaches its minimum of $-\epsilon$ at 3.78 Angstrom, where the force is zero:

```{code-cell} python
r_min = 2 ** (1 / 6) * lj.sigma
print(f"minimum at {r_min * 1e10:.2f} Angstrom, energy {lj.energies(r_min) * 1e21:.3f} zJ")
print(f"force there {lj.forces(r_min) * 1e12:.3f} pN")
```

Beyond the minimum the atoms attract each other, and the attraction fades to nothing by about 8 Angstrom.

Whether two atoms stay bound depends on how the well depth compares with the thermal energy $k_B T$, where $k_B$ is the Boltzmann constant and $T$ the temperature. At room temperature the argon well is only 0.4 $k_B T$ deep, so collisions break a pair apart almost as soon as it forms, and argon is a gas. At 87 K the well is 1.3 $k_B T$ deep, deep enough for atoms to stick together, and argon condenses; 87 K is its boiling point.

## The box

The atoms are confined to a square box, but its edges are not walls. An atom that moves out through the right-hand edge comes back in through the left, and an atom near the right-hand edge interacts with one near the left-hand edge as if they were neighbours. The box behaves as one cell of a pattern that repeats endlessly in both directions, so the simulation has no walls and no surface, and a few tens of atoms stand in for a much larger sample. Boundaries of this kind are called periodic.

Because the pattern repeats, every atom has infinitely many copies, one in each cell. When pylj needs the distance between two atoms it takes the shortest distance to any copy of the second atom. That shortest distance is called the minimum image.

The attraction between two argon atoms has faded to nothing by about 8 Angstrom, so beyond some distance a pair contributes nothing worth computing. pylj therefore ignores pairs of atoms further apart than a cut-off distance, and sets their energy and force to zero. The default cut-off is 15 Angstrom, or half the box side if the box is smaller than 30 Angstrom.

If the cut-off were larger than half the box side, an atom could be within the cut-off distance of two copies of the same neighbour, one on each side of it, while the minimum image counts only the nearer copy. The cut-off is therefore never allowed to exceed half the box side.

Two details of the cut-off matter to anyone comparing with other simulation codes. The energy is set to zero at the cut-off with no adjustment for the small interaction that remains, and pylj refuses a potential whose energy at the cut-off is still larger than $k_B T$.

The box side must lie between 4 and 600 Angstrom. A smaller box has room for one atom at most, and in a larger one the atoms are too small to see.

## Building a simulation

A simulation is built from the number of atoms, the temperature in kelvin, the box side in Angstrom, and the model: the species present, and the potential that acts between each pair of species. The cell below builds nine argon atoms in a 20 Angstrom box at 300 K:

```{code-cell} python
from pylj.md import MDSimulation

model = dict(species=[argon], pair_potentials={(argon, argon): lj})
simulation = MDSimulation.initialise(9, 300, 20, seed=0, **model)
configuration = simulation.configuration
print(f"box {configuration.box * 1e10:.0f} Angstrom, cut-off {simulation.cut_off * 1e10:.0f} Angstrom")
print(configuration.position[:3] * 1e10)
```

The species and the pair potentials are collected in a dictionary called `model`. Writing `**model` in the call passes its two entries as the keyword arguments `species=` and `pair_potentials=`. Every later chapter builds its simulations this way, so that the model is written once.

With one species, `pair_potentials` has a single entry. A mixture of argon and xenon would need three: argon with argon, xenon with xenon, and argon with xenon.

The simulation starts by placing the atoms on a square lattice, and `configuration` records where they are. pylj reports every quantity in SI units, so the positions are in metres; the cell above converts them to Angstrom for printing.

## The energy of the whole box

The potential energy of the box is the sum of the pair energies over every pair of atoms. Nine atoms make thirty-six pairs, and `pairs` evaluates them all at once, returning the distance and the energy of each pair and, if asked, the force:

```{code-cell} python
pairs = configuration.pairs(simulation.pair_potentials, simulation.cut_off, forces=True)
print(pairs.distance.size, "pairs")
print(f"nearest pair {pairs.distance.min() * 1e10:.2f} Angstrom")
print(f"total energy {pairs.energy.sum() * 1e21:.2f} zJ")
```

Adding up the forces from every pair an atom belongs to gives the net force on that atom. The next chapter uses those forces to move the atoms, and the Monte Carlo chapter uses the energies alone.
