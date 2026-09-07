---
file_format: mystnb
kernelspec:
  name: python3
---

# Atoms and potentials

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

A simulation of argon starts from three pieces of information: the mass of an argon atom, the potential energy of two argon atoms as a function of the distance between them, and the size of the box they are in. This chapter covers each in turn, then builds a simulation from them.

## The atoms

`pylj` describes an atom by its mass and a name. The mass is in atomic mass units, the unit used in data tables, so for argon it is 39.948:

```{code-cell} python
from pylj.potentials import Species

argon = Species(mass=39.948, name="argon")
argon
```

`pylj` calls this description a species. One species serves every atom of the same kind, so a simulation of argon has one species and a simulation of argon mixed with xenon has two.

## How the atoms interact

An Angstrom is $10^{-10}$ m, about the size of an atom. Two argon atoms a few Angstrom apart attract each other weakly. The attraction is the dispersion interaction, which arises from momentary fluctuations in the electron clouds of the two atoms and acts between any two atoms whatever they are. When the atoms come close enough for their electron clouds to overlap, they repel each other strongly. The potential energy of the pair therefore falls as the atoms approach from far away, reaches a minimum, and then rises steeply. The minimum is called the well. The Lennard-Jones potential is the simplest formula with this shape:

$$
E(r) = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right].
$$

$r$ is the distance between the two atoms. $\epsilon$ is the depth of the well, which is the energy needed to pull a bound pair apart. $\sigma$ is the distance at which the energy passes through zero, a little less than the diameter of the atom. For argon, $\epsilon$ is 1.577 zJ, where a zeptojoule is $10^{-21}$ J, and $\sigma$ is 3.372 Angstrom. Argon atoms have no bonds, no charge and no shape of their own, so this one formula describes their interaction well.

`LennardJones` is `pylj`'s version of the formula. It takes $\epsilon$ in joules and $\sigma$ in metres:

```{code-cell} python
from pylj.potentials import LennardJones

lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
```

`lj.energies(r)` gives the energy at a distance `r`, or at every distance in an array. `lj.forces(r)` gives the force, which is minus the slope of the energy: positive where the atoms repel and negative where they attract. The cell below plots both between 3 and 8 Angstrom.

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

The force is in piconewtons, where a piconewton is $10^{-12}$ N. Below 3.37 Angstrom the energy rises steeply and the force is large and repulsive: at 3 Angstrom the force is nearly 800 pN, far above the top of the plot. The energy passes through zero at $\sigma$ and reaches its minimum of $-\epsilon$ at 3.78 Angstrom, where the force is zero:

```{code-cell} python
r_min = 2 ** (1 / 6) * lj.sigma
print(f"minimum at {r_min * 1e10:.2f} Angstrom, energy {lj.energies(r_min) * 1e21:.3f} zJ")
print(f"force there {lj.forces(r_min) * 1e12:.3f} pN")
```

Beyond the minimum the atoms attract each other, and the attraction has fallen to nothing by about 8 Angstrom.

Whether two atoms stay bound depends on how the depth of the well compares with the thermal energy $k_B T$, where $k_B$ is the Boltzmann constant and $T$ the temperature. At room temperature the argon well is 0.4 $k_B T$ deep, so a collision breaks a pair apart almost as soon as it forms, and argon is a gas. At 87 K, the boiling point of argon, the well is 1.3 $k_B T$ deep and atoms stick together.

## The box

We want to simulate a bulk material, which contains around $10^{26}$ atoms, but we can only simulate somewhere between $10^{3}$ and $10^{6}$, and in `pylj` a few tens. Even in a large simulation, most of the atoms would then be near the edges of the sample rather than in a bulk environment. Periodic boundary conditions remove the edges. The square box is treated as one cell of a pattern that repeats without end in every direction: an atom that leaves through the right-hand side of the box re-enters through the left, and an atom near the right-hand side interacts with atoms near the left-hand side as though they were next to it. There are no walls and no surface.

In a repeating pattern every atom has a copy in every cell. When `pylj` calculates the distance between two atoms it uses the distance to the nearest copy of the second atom. That distance is called the minimum image.

The attraction between two argon atoms has fallen to nothing by about 8 Angstrom, so a pair of atoms further apart than that adds nothing to the energy. `pylj` therefore counts only pairs closer than a cut-off distance, and sets the energy and force of every other pair to zero. The default cut-off is 15 Angstrom, or half the box side if the box is smaller than 30 Angstrom.

The cut-off can never be more than half the box side. If it were, an atom could be within the cut-off distance of two copies of the same neighbour, one on each side of it, while the minimum image counts only the nearer copy.

The energy is cut to zero at the cut-off, and nothing is added back for the small interaction that remains beyond it. `pylj` refuses a potential whose energy at the cut-off is still larger than $k_B T$.

## Building a simulation

A simulation is built from the number of atoms, the temperature in kelvin, the box side in Angstrom, and the model: the species present and the potential that acts between each pair of species. The cell below builds nine argon atoms in a 20 Angstrom box at 300 K, and prints the box, the cut-off and the positions of the first three atoms:

```{code-cell} python
from pylj.md import MDSimulation

model = dict(species=[argon], pair_potentials={(argon, argon): lj})
simulation = MDSimulation.initialise(number_of_atoms=9, temperature=300, box=20, seed=0, **model)
configuration = simulation.configuration
print(
    f"box {configuration.box * 1e10:.0f} Angstrom, cut-off {simulation.cut_off * 1e10:.0f} Angstrom"
)
print(configuration.position[:3] * 1e10)
```

The atoms start on a square lattice, and `configuration` records where they are. `pylj` reports every quantity in SI units, so the positions are in metres, and the cell converts them to Angstrom for printing. The inputs are in mixed units: `initialise` takes the box in Angstrom, `LennardJones` takes joules and metres, and `Species` takes atomic mass units.

The species and the pair potentials are collected in a dictionary called `model`, and `**model` in the call passes its two entries as the keyword arguments `species=` and `pair_potentials=`. Every later chapter builds its simulations this way, so that the model is written once. With one species, `pair_potentials` has a single entry. A mixture of argon and xenon would need three: argon with argon, xenon with xenon, and argon with xenon.

## The energy of the whole box

The potential energy of the box is the sum of the pair energies over every pair of atoms. Nine atoms make thirty-six pairs, and `pairs` evaluates them all at once, returning the distance and the energy of each pair and, if asked, the force:

```{code-cell} python
pairs = configuration.pairs(simulation.pair_potentials, simulation.cut_off, forces=True)
print(pairs.distance.size, "pairs")
print(f"nearest pair {pairs.distance.min() * 1e10:.2f} Angstrom")
print(f"total energy {pairs.energy.sum() * 1e21:.2f} zJ")
```

Adding up the forces from every pair an atom belongs to gives the net force on that atom. The next chapter uses those forces to move the atoms, and the Monte Carlo chapter uses the energies alone.
