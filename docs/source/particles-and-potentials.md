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

Throughout these pages our example system is argon. Argon is the traditional first system to simulate. Its atoms have no bonds, no charge and no shape, so the only thing we have to describe is how two argon atoms attract and repel each other, and yet a box of argon atoms shows everything we want to study: how atoms move, how they arrange themselves, and what pressure they exert.

To simulate argon we need to answer three questions. How do we describe an argon atom? How do two argon atoms interact? And where do the atoms live, given that a simulation cannot contain a whole beaker of gas? This chapter takes each question in turn, then builds a simulation and computes its energy.

## Describing an atom

In a classical simulation an atom is a point with a mass. Its electrons never appear explicitly; their effect is folded into the potential we choose in the next section. So the only thing `pylj` needs to know about an argon atom is its mass, which is 39.948 atomic mass units, together with a name to label it by:

```{code-cell} python
from pylj.potentials import Species

argon = Species(mass=39.948, name="argon")
argon
```

`pylj` calls this a species. Every argon atom in a simulation shares the same species, so a simulation of pure argon has one species, and a simulation of argon mixed with xenon would have two.

## How two argon atoms interact

Two argon atoms a few Angstrom apart attract each other weakly. This is the dispersion interaction: a momentary fluctuation in the electron cloud of one atom induces a matching fluctuation in the other, and the two attract. Bring the atoms closer, so that their electron clouds overlap, and they repel each other strongly. The potential energy of the pair therefore falls as the atoms approach from a distance, passes through a minimum, which we call the well, and then rises steeply. (An Angstrom is $10^{-10}$ m, roughly the size of an atom.)

The Lennard-Jones potential is the simplest formula with this shape:

$$
E(r) = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right],
$$

where $r$ is the distance between the two atoms. The formula has two parameters. $\epsilon$ sets the depth of the well: it is the energy needed to pull a bound pair apart. $\sigma$ sets the distance at which the energy passes through zero, which is slightly less than the diameter of the atom. For argon, $\epsilon = 1.577$ zJ (a zeptojoule is $10^{-21}$ J) and $\sigma = 3.372$ Angstrom.

In `pylj` this formula is a `LennardJones` object, built from $\epsilon$ in joules and $\sigma$ in metres:

```{code-cell} python
from pylj.potentials import LennardJones

lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
```

We can ask it for the energy at any distance, or at a whole array of distances, with `lj.energies()`, and for the force with `lj.forces()`. The force is minus the slope of the energy: positive where the atoms repel and negative where they attract. Let us plot both between 3 and 8 Angstrom:

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

Reading the plot from left to right: below $\sigma = 3.37$ Angstrom the energy rises steeply and the force is strongly repulsive. At 3 Angstrom the force is nearly 800 pN, far above the top of the plot (a piconewton is $10^{-12}$ N). The energy passes through zero at $\sigma$ and reaches its minimum of $-\epsilon$ at 3.78 Angstrom, where the force is zero:

```{code-cell} python
r_min = 2 ** (1 / 6) * lj.sigma
print(f"minimum at {r_min * 1e10:.2f} Angstrom, energy {lj.energies(r_min) * 1e21:.3f} zJ")
print(f"force there {lj.forces(r_min) * 1e12:.3f} pN")
```

Beyond the minimum the atoms attract, and by about 8 Angstrom the attraction has fallen to nothing.

How strongly do two argon atoms bind? The natural comparison is with the thermal energy $k_B T$, where $k_B$ is the Boltzmann constant and $T$ the temperature. At room temperature the argon well is only 0.4 $k_B T$ deep, so a collision breaks a pair apart almost as soon as it forms, and argon is a gas. At 87 K, the boiling point of argon, the well is 1.3 $k_B T$ deep and atoms begin to stick together.

## The box

We want to simulate a bulk material, which contains around $10^{26}$ atoms, but we can only simulate somewhere between $10^{3}$ and $10^{6}$, and in `pylj` a few tens. Even in a large simulation, most of the atoms would then be near the edges of the sample rather than in a bulk environment. Periodic boundary conditions remove the edges. We treat the square box as one cell of a pattern that repeats without end in every direction: an atom that leaves through the right-hand side of the box re-enters through the left, and an atom near the right-hand side interacts with atoms near the left-hand side as though they were next to it. There are no walls and no surface.

Because the pattern repeats, every atom has a copy in every cell, and we have to decide which copy to use when we calculate the distance between two atoms. `pylj` uses the nearest copy. The distance to it is called the minimum image.

We also have to decide which pairs of atoms to include when we add up the energy. In principle every pair interacts, but the attraction between two argon atoms has fallen to nothing by about 8 Angstrom, so a pair further apart than that contributes nothing worth computing. `pylj` therefore counts only pairs closer than a cut-off distance and sets the energy and force of every other pair to zero. The default cut-off is 15 Angstrom, or half the box side if the box is smaller than 30 Angstrom. The cut-off can never be more than half the box side. If it were, an atom could be within the cut-off distance of two copies of the same neighbour, one on each side, while the minimum image counts only the nearer copy. Two details of the cut-off are worth knowing. The energy is cut to zero abruptly, with nothing added back for the small interaction that remains beyond the cut-off, and `pylj` refuses a potential whose energy at the cut-off is still larger than $k_B T$.

## Building a simulation

We now have everything we need. A simulation is built from the number of atoms, the temperature, the box, and the model: the species present and the potential that acts between each pair of species. Here are nine argon atoms in a 20 Angstrom box at 300 K:

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

The atoms start on a square lattice, and `configuration` records where they are. Everything `pylj` reports is in SI units, so the positions are in metres, and the cell converts them to Angstrom for printing. The inputs are in mixed units: `initialise` takes the box in Angstrom, `LennardJones` takes joules and metres, and `Species` takes atomic mass units.

The species and the pair potentials are collected in a dictionary called `model`, and `**model` in the call passes its two entries as the keyword arguments `species=` and `pair_potentials=`. Every later chapter builds its simulations this way, so that the model is written once. With one species, `pair_potentials` has a single entry. A mixture of argon and xenon would need three: argon with argon, xenon with xenon, and argon with xenon.

## The energy of the whole box

The potential energy of the box is the sum of the pair energies over every pair of atoms. Nine atoms make thirty-six pairs, and `pairs` evaluates them all at once, returning the distance and the energy of each pair and, if we ask, the force:

```{code-cell} python
pairs = configuration.pairs(simulation.pair_potentials, simulation.cut_off, forces=True)
print(pairs.distance.size, "pairs")
print(f"nearest pair {pairs.distance.min() * 1e10:.2f} Angstrom")
print(f"total energy {pairs.energy.sum() * 1e21:.2f} zJ")
```

Adding up the forces from every pair an atom belongs to gives the net force on that atom. The next chapter uses those forces to move the atoms; the Monte Carlo chapter uses only the energies.
