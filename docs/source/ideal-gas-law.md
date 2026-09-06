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
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit
from pylj import sample
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN
from pylj.md import MDSimulation
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
model = dict(species=[argon], pair_potentials={(argon, argon): lj})


def run(number_of_particles, temperature, box, steps, viewer_class=None, draw_every=100, seed=0):
    """Settle for steps // 5, then run steps at the temperature, sampling every fifth."""
    simulation = MDSimulation.initialise(number_of_particles, temperature, box, seed=seed, **model)
    for _ in range(steps // 5):
        simulation.step()
        simulation.heat_bath(temperature)
    production = simulation.restart()
    viewer = viewer_class(production) if viewer_class else None
    for _ in range(steps):
        production.step()
        production.heat_bath(temperature)
        if production.steps % 5 == 0:
            production.sample()
        if viewer and production.steps % draw_every == 0:
            viewer.update(production)
    if viewer and any(pane.keeps_history for pane in viewer.panes):
        viewer.average()
    return production
```

The thermostat holds the temperature from the first step, but the square lattice the particles start on is not a fluid. The settling steps let it relax into one while the thermostat absorbs the potential energy released; `restart` then begins a fresh record, and the viewer is built after it so its history covers only the settled run. Sampling every fifth step is enough, because consecutive steps are almost the same configuration.

## Temperature and the speed distribution

In two dimensions the speeds of particles of mass $m$ at temperature $T$ follow the Maxwell-Boltzmann distribution

$$
p(v) = \frac{m v}{k_B T} \exp\!\left(-\frac{m v^2}{2 k_B T}\right),
$$

whose peak is at $\sqrt{k_B T / m}$. Collecting the speeds of forty particles over a run at 100 K and again at 1000 K, and drawing the distribution over each histogram:

```{code-cell} python
def speeds(temperature, steps=1000):
    simulation = MDSimulation.initialise(40, temperature, 40, seed=0, **model)
    for _ in range(200):
        simulation.step()
        simulation.heat_bath(temperature)
    collected = []
    for _ in range(steps):
        simulation.step()
        simulation.heat_bath(temperature)
        if simulation.steps % 10 == 0:
            collected.append(np.linalg.norm(simulation.configuration.velocity, axis=1))
    return np.concatenate(collected)


mass = 39.948 * ATOMIC_MASS_UNIT
fig, ax = plt.subplots(figsize=(5, 3.2))
for temperature in (100, 1000):
    v = speeds(temperature)
    ax.hist(v, bins=25, density=True, alpha=0.5, label=f"{temperature} K")
    grid = np.linspace(0, v.max(), 300)
    kt = BOLTZMANN * temperature
    ax.plot(grid, mass * grid / kt * np.exp(-mass * grid**2 / (2 * kt)), color="black")
    print(f"{temperature} K: peak predicted at {np.sqrt(kt / mass):.0f} m/s")
ax.set_xlabel("speed / m s$^{-1}$")
ax.set_ylabel("probability density")
ax.legend()
fig.tight_layout()
```

The histograms follow the curves, and the predicted peak moves out by $\sqrt{10}$ between the two temperatures. The thermostat fixes the total kinetic energy at every step, and the collisions between one rescaling and the next share it among the forty particles, so each particle's speed still varies as the distribution says.

## Density and structure

The radial distribution function $g(r)$ is the probability of finding a particle at distance $r$ from another, relative to the same probability in an ideal gas at the same density, so it tends to one at large $r$. Averaged over a run, it shows structure. At 100 K the well depth of 1.58 zJ is $1.1\,k_B T$, deep enough for pairs to linger near the minimum of the potential.

```{code-cell} python
dilute = run(20, 100, 40, 20000, sample.RDF, draw_every=200)
```

```{code-cell} python
dense = run(100, 100, 40, 2000, sample.RDF, draw_every=20)
```

With twenty particles there is one peak, at the minimum of the potential, and beyond it $g(r)$ settles to one: a particle has a neighbour at the well minimum more often than chance, and no order beyond that. In the dilute limit the height of the peak is the Boltzmann factor of the well depth, $\exp(\epsilon / k_B T)$, which is 3.1 at 100 K, and the dilute run is long because a curve from twenty particles takes many frames to converge. With a hundred particles in the same box the first peak sharpens and a second and third appear at twice and three times the distance. These are the shells of neighbours of a liquid. The axis is in metres, with a factor of 1e-9 printed in its corner.

## Argon at standard temperature and pressure

The density of argon at STP is 1.784 g/L. In three dimensions that is a number density $n_3$; the two-dimensional density with the same spacing between particles is $n_3^{2/3}$.

```{code-cell} python
n3 = 1.784 / (39.948 * ATOMIC_MASS_UNIT)  # particles per cubic metre
n2 = n3 ** (2 / 3)  # particles per square metre
box = 150  # Angstrom
number = round(n2 * (box * 1e-10) ** 2)
print(f"{n2 * 1e-20:.2e} particles per square Angstrom: {number} particles in a {box} Angstrom box")
```

At that density a 40 Angstrom box would hold one particle, so the box is 150 Angstrom.

```{code-cell} python
stp = run(number, 273.15, box, 2000, sample.JustCell)
area = (box * 1e-10) ** 2
measured = stp.samples.pressure.mean()
print(f"measured pressure {measured:.3e} N/m")
print(f"ideal gas law with N particles {number * BOLTZMANN * 273.15 / area:.3e} N/m")
print(f"ideal gas law with N - 1 particles {(number - 1) * BOLTZMANN * 273.15 / area:.3e} N/m")
print(f"mean potential energy per particle {stp.samples.potential_energy.mean() / number / (BOLTZMANN * 273.15):.3f} k_B T")
```

The particles rarely come within range of each other, and the potential energy is a small fraction of $k_B T$ per particle. The measured pressure sits on the second of the two ideal lines, not the first. The simulation holds the centre of mass at rest, so the kinetic energy of $N$ particles is $(N - 1) k_B T$ rather than $N k_B T$, and the kinetic term of the pressure is one particle short; the next section returns to this. Allowing for it, argon at STP is an ideal gas to a fraction of a per cent, because it is dilute.

## Pressure against density

Keeping the box at 40 Angstrom and the temperature at 273 K, and raising the number of particles from 9 to 100, takes the gas from dilute to dense. The pressure pylj measures is the virial pressure,

$$
p = \frac{1}{2A}\left(2K + \sum_{\text{pairs}} f_{ij}\, r_{ij}\right),
$$

where $K$ is the kinetic energy and the sum runs over the radial force times the distance for every pair. The thermostat sets $K$ to $(N - 1) k_B T$ at every step, so the first term is fixed and only the second, the virial, is measured. With no forces the virial vanishes and the pressure is $(N - 1) k_B T / A$.

```{code-cell} python
numbers = np.array([9, 16, 25, 36, 49, 64, 81, 100])
area = (40e-10) ** 2
kt = BOLTZMANN * 273
pressures = np.array([run(n, 273, 40, 1500, seed=n).samples.pressure.mean() for n in numbers])
ideal = (numbers - 1) * kt / area
ratio = pressures / ideal
fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.2))
left.plot(numbers, pressures, "o", label="measured")
left.plot(numbers, ideal, "--", label="ideal, $N - 1$")
left.set_xlabel("N")
left.set_ylabel("p / N m$^{-1}$")
left.legend()
right.plot(numbers, ratio, "o")
right.axhline(1, color="grey", linewidth=0.5)
right.set_xlabel("N")
right.set_ylabel("p / p$_{\\mathrm{ideal}}$")
fig.tight_layout()
for n, r in zip(numbers, ratio):
    print(f"N = {n:3d}: {r:.2f} times ideal")
```

The right-hand panel divides the measured pressure by the ideal one. At the lowest densities the ratio is a few per cent above one. The excess grows steeply with $N$, and at 100 particles the pressure is several times ideal: the particles take up a large fraction of the box, their repulsive cores push on each other, and the virial is large and positive.

The pressure never falls below the ideal line. The Lennard-Jones well is attractive, and at low enough temperature the attraction wins at low density and pulls the pressure under the line. The temperature at which the two effects balance is where the second virial coefficient changes sign:

```{code-cell} python
def second_virial(temperature):
    kt = BOLTZMANN * temperature
    integrand = lambda r: (np.exp(-lj.energies(r) / kt) - 1) * 2 * np.pi * r
    return -0.5 * quad(integrand, 0, 15e-10, points=[2.5e-10, 3.4e-10, 5e-10], limit=200)[0]


boyle = brentq(second_virial, 50, 500)
print(f"second virial coefficient at 273 K: {second_virial(273) * 1e20:+.2f} Angstrom^2")
print(f"it changes sign at {boyle:.0f} K")
```

At 273 K the well is only $0.42\,k_B T$ deep and the coefficient is positive: repulsion wins at every density. Below the temperature printed above the coefficient is negative, and a run at 100 K, where the earlier sections found pairs lingering in the well, would show the pressure dip below the line at low density.

At low density the excess is set by the second virial coefficient alone: the pressure is $(N - 1) k_B T / A$ plus $B_2 N^2 k_B T / A^2$, so the ratio is $1 + B_2 N^2 / (A (N - 1))$. A single run scatters by a few per cent about this, but the prediction and the measurements agree in size and sign:

```{code-cell} python
b2 = second_virial(273)
for n, r in zip(numbers[:3], ratio[:3]):
    print(f"N = {n:3d}: predicted {1 + b2 * n**2 / (area * (n - 1)):.2f}, measured {r:.2f}")
```

## The van der Waals equation

The two corrections have a classical form. Writing $v = A / N$ for the area per particle, the van der Waals equation in two dimensions is

$$
p = \frac{k_B T}{v - b} - \frac{a}{v^2},
$$

with $b$ the area a particle excludes and $a$ the strength of the attraction. Expanded at low density it gives a second virial coefficient of $b - a / k_B T$. The fit is to the pressure $N$ particles would give with the measured virial, which adds the one missing particle's $k_B T / A$ back to the kinetic term:

```{code-cell} python
def van_der_waals(v, a, b):
    return kt / (v - b) - a / v**2


v = area / numbers
full = pressures + kt / area
(a, b), _ = curve_fit(van_der_waals, v, full, p0=(1e-40, 1e-19))
diameter = np.sqrt(2 * b / np.pi)
print(f"a = {a:.2e} J m^2, b = {b:.2e} m^2")
print(f"b - a / k_B T = {(b - a / kt) * 1e20:+.2f} Angstrom^2, against {second_virial(273) * 1e20:+.2f} from the potential")
print(f"hard discs with b = pi d^2 / 2 have diameter d = {diameter * 1e10:.2f} Angstrom")
print(f"the attractive term a / v^2 at 100 particles is {a / v[-1]**2 / full[-1]:.0%} of the pressure")
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.plot(numbers, full, "o", label="measured")
fine = np.linspace(numbers[0], numbers[-1], 200)
ax.plot(fine, van_der_waals(area / fine, a, b), "-", label="van der Waals fit")
ax.set_xlabel("N")
ax.set_ylabel("p / N m$^{-1}$")
ax.legend()
fig.tight_layout()
```

The fitted $b - a / k_B T$ has the sign and size of the second virial coefficient of the potential itself, and comes out about a fifth above it: a two-parameter form fitted up to 100 particles, where the box is nearly full, is only a rough guide to the dilute limit. For hard discs of diameter $d$ the excluded area is $b = \pi d^2 / 2$, and the diameter that comes out is about $0.8\,\sigma$, where $\sigma$ is the separation at which the Lennard-Jones energy crosses zero. Part of the shortfall is that the repulsive wall is soft, so collisions at 273 K push inside $\sigma$; the rest is the crudeness of the fit. The attractive term is a small correction at every density here, as the positive second virial coefficient requires. The fit has recovered the size of the particle from the pressure alone, which is what van der Waals did from experiment in 1873.
