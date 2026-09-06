---
file_format: mystnb
kernelspec:
  name: python3
---

# The ideal gas law

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

The ideal gas law, $pV = N k_B T$, holds for particles that do not interact and take up no space. Argon at room temperature and pressure is close to that, and a simulation can show both how close and where the law breaks down. This chapter runs argon at a series of densities, measures the pressure, and compares it with the law. The two-dimensional form of the law is $pA = N k_B T$, with $A$ the area of the box; the next chapter derives it.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylj import sample
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN
from pylj.md import MDSimulation
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
model = dict(species=[argon], pair_potentials={(argon, argon): lj})


def run(number_of_particles, temperature, box, steps, viewer_class=None, seed=0):
    """Settle for a fifth of the steps, then sample the rest at the temperature."""
    simulation = MDSimulation.initialise(number_of_particles, temperature, box, seed=seed, **model)
    for _ in range(steps // 5):
        simulation.step()
        simulation.heat_bath(temperature)
    production = simulation.restart()
    viewer = viewer_class(production) if viewer_class else None
    for _ in range(steps):
        production.step()
        production.heat_bath(temperature)
        production.sample()
        if viewer and production.steps % 100 == 0:
            viewer.update(production)
    if viewer and hasattr(viewer.panes[-1], "history"):
        viewer.average()
    return production
```

The first fifth of each run brings the lattice to the temperature and is thrown away; `restart` begins a fresh record from that point, and the viewer is built after it so its history covers only the settled run. Every step is thermostatted, so the temperature is held at the value the ideal gas law will be evaluated at.

## Temperature and the speed distribution

Forty particles in a 40 Angstrom box, at 100 K and at 1000 K. The speed histogram pools every frame drawn.

```{code-cell} python
cold = run(40, 100, 40, 2000, sample.MaxBolt)
```

```{code-cell} python
hot = run(40, 1000, 40, 2000, sample.MaxBolt)
```

At 100 K the particles cluster: the well depth of 1.58 zJ is about $1.1\,k_B T$, so pairs stay bound. At 1000 K it is $0.1\,k_B T$ and the particles move through each other's wells without sticking. The speed distribution is the two-dimensional Maxwell-Boltzmann distribution, whose peak moves out as $\sqrt{T}$.

## Density and structure

The radial distribution function $g(r)$ is the probability of finding a particle at distance $r$ from another, relative to the same probability in an ideal gas at the same density. Averaged over a run, it shows structure. At low density there is one peak at the well minimum and $g(r)$ is one beyond it; at high density the first peak sharpens and further peaks appear, the shells of neighbours of a liquid.

```{code-cell} python
dilute = run(20, 100, 40, 2000, sample.RDF)
```

```{code-cell} python
dense = run(100, 100, 40, 2000, sample.RDF)
```

## Argon at standard temperature and pressure

The density of argon at STP is 1.784 g/L. In three dimensions that is a number density $n_3$; the two-dimensional density with the same spacing between particles is $n_3^{2/3}$.

```{code-cell} python
avogadro = 6.02214076e23
n3 = 1.784e3 / 39.948 * avogadro  # particles per cubic metre
n2 = n3 ** (2 / 3)  # particles per square metre
box = 150  # Angstrom
number = round(n2 * (box * 1e-10) ** 2)
print(f"{n2 * 1e-20:.2e} particles per square Angstrom: {number} particles in a {box} Angstrom box")
```

At that density a 40 Angstrom box would hold one particle, so the box is 150 Angstrom.

```{code-cell} python
stp = run(number, 273.15, box, 2000, sample.Interactions)
ideal = number * BOLTZMANN * 273.15 / (box * 1e-10) ** 2
print(f"measured pressure {stp.samples.pressure.mean():.3e} N/m, ideal {ideal:.3e} N/m")
print(f"mean potential energy per particle {stp.samples.potential_energy.mean() / number / (BOLTZMANN * 273.15):.3f} k_B T")
```

The particles rarely come within range of each other, the potential energy is a small fraction of $k_B T$ per particle, and the pressure is close to the ideal value. Argon at STP behaves as an ideal gas because it is dilute.

## Pressure against density

Keeping the box at 40 Angstrom and the temperature at 273 K, and raising the number of particles from 4 to 100, takes the gas from dilute to dense. The pressure pylj measures is the virial pressure,

$$
p = \frac{1}{2A}\left(2K + \sum_{\text{pairs}} f_{ij}\, r_{ij}\right),
$$

where $K$ is the kinetic energy and the sum runs over the radial force times the distance for every pair. With no forces the second term vanishes and $2K / 2A = K / A$ is the kinetic pressure.

```{code-cell} python
numbers = np.array([4, 9, 16, 25, 36, 49, 64, 81, 100])
area = (40e-10) ** 2
pressures = np.array([run(n, 273, 40, 2000, seed=n).samples.pressure.mean() for n in numbers])
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.plot(numbers, pressures, "o", label="measured")
ax.plot(numbers, numbers * BOLTZMANN * 273 / area, "-", label="ideal, $N$")
ax.plot(numbers, (numbers - 1) * BOLTZMANN * 273 / area, "--", label="ideal, $N - 1$")
ax.set_xlabel("N")
ax.set_ylabel("p / N m$^{-1}$")
ax.legend()
fig.tight_layout()
```

Two things separate the points from the ideal line.

The simulation holds the centre of mass at rest, so the kinetic energy of $N$ particles is $(N - 1) k_B T$ rather than $N k_B T$, and the kinetic term of the pressure is one particle short. The dashed line is the ideal gas law for $N - 1$ particles, and the dilute points sit on it. In a real gas of $10^{23}$ particles the difference is nothing; in a box of four it is a quarter.

The remaining gap grows with $N$, and it never goes negative: at $N = 4$ the measured pressure is already a few percent above the $N - 1$ line, and by $N = 100$ it is more than three times higher. The virial term in the pressure sums the force between a pair times their separation, and a collision spends most of that sum on the repulsive core. The potential falls steeply at short range, so a close pass pushes hard, and that push outweighs the slow pull between the many pairs that are merely nearby. Past about 50 particles the box is crowded enough that these repulsive collisions happen often, and the pressure rises steeply above the kinetic term alone.

## The van der Waals equation

The two corrections have a classical form. The van der Waals equation in two dimensions is

$$
p = \frac{k_B T}{a_{\text{p}} - b} - \frac{a}{a_{\text{p}}^2},
$$

with $a_{\text{p}} = A / N$ the area per particle, $b$ the area a particle excludes and $a$ the strength of the attraction. Fitting both to the measured pressures:

```{code-cell} python
def van_der_waals(area_per_particle, a, b):
    return BOLTZMANN * 273 / (area_per_particle - b) - a / area_per_particle**2

area_per_particle = area / numbers
(a, b), _ = curve_fit(van_der_waals, area_per_particle, pressures, p0=(1e-40, 1e-19))
print(f"a = {a:.2e} J m^2, b = {b:.2e} m^2, so an excluded diameter of {2 * np.sqrt(b / np.pi) * 1e10:.1f} Angstrom")
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.plot(numbers, pressures, "o", label="measured")
fine = np.linspace(numbers[0], numbers[-1], 200)
ax.plot(fine, van_der_waals(area / fine, a, b), "-", label="van der Waals fit")
ax.set_xlabel("N")
ax.set_ylabel("p / N m$^{-1}$")
ax.legend()
fig.tight_layout()
```

The excluded diameter the fit returns is close to $\sigma$, the separation at which the Lennard-Jones energy crosses zero: the fit has recovered the size of the particle from the pressure alone, which is what van der Waals did from experiment in 1873.
