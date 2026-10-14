---
file_format: mystnb
kernelspec:
  name: python3
---

# Samples and the trajectory

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
%config InlineBackend.figure_format = "retina"
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

A run records nothing until `sample()` is called. Each call records the quantities the method measures in `simulation.samples`, and a copy of the configuration in `simulation.trajectory`. Everything on this page is computed from those two records. The examples sample a short molecular dynamics run of liquid argon every ten steps:

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

from pylj.md import MDSimulation
from pylj.model import Model
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
model = Model.single(argon, LennardJones(epsilon=0.9497, sigma=3.372))
simulation = MDSimulation.initialise(
    model, number_of_atoms=30, temperature=300, box=20, init_conf="triangular", seed=1
)
for _ in range(3000):
    simulation.step()
    if simulation.steps % 50 == 0:
        simulation.rescale_velocities(300)
    if simulation.steps % 10 == 0:
        simulation.sample()
```

## Samples

`sample()` records a copy of the configuration in `trajectory` and appends one entry to each array of `samples`, an `MDSamples`: `step`, `temperature`, `pressure`, `potential_energy` and `kinetic_energy`, with `total_energy` derived from them. The pressure is the virial pressure, `(2 K + sum(f r)) / (2 L^2)`, in kJ/mol per Angstrom squared.

`sample()` records a copy of the configuration in `trajectory`, recomputes the energy exactly and appends `step` and `potential_energy` to `samples`, an `MCSamples`.

The samples are NumPy arrays, so they plot directly:

```{code-cell} python
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(simulation.samples.step * simulation.timestep, simulation.samples.temperature)
ax.set_xlabel("Time / ps")
ax.set_ylabel("Temperature / K")
plt.show()
```

## Trajectory

`sample()` also records the current configuration in `simulation.trajectory`, one frame per sample. A frame is a `Configuration`, so `simulation.trajectory[-1].positions` is the last sampled positions, and `simulation.trajectory.positions` is every frame's, an array of shape `(frames, N, 2)`. Slicing gives a trajectory, so `simulation.trajectory[100:]` is the run after the first hundred frames. A molecular dynamics frame carries the time it was sampled at, and `simulation.trajectory.times` is the array of them in picoseconds; a Monte Carlo trajectory has no times. `restart()` starts an empty trajectory.

A configuration is kept only if it was sampled, so memory grows with the number of samples and not with the number of steps. A molecular dynamics frame takes about sixty-five bytes per atom: a hundred atoms sampled a thousand times is six and a half megabytes, and sampled a hundred thousand times is about two thirds of a gigabyte.

Three analyses are computed on demand, from one frame or averaged over a trajectory. `rdf(bins=100, r_max=None)` returns the bin centres in Angstrom and g(r), which is one where the atoms are spread as evenly as an ideal gas; `r_max` defaults to half the box. `structure_factor(q_max=None)` returns the wavevector magnitudes in inverse Angstrom and S(q) at each, which is one where the atoms are spread as evenly as an ideal gas. S(q) is evaluated at the wavevectors `2 pi (h, k) / L` commensurate with the box, for integer `h` and `k` not both zero. The amplitude of a wavevector is the sum of `exp(i q . r)` over the atoms, and S(q) is the square of its modulus divided by the number of atoms. Wavevectors of equal magnitude are averaged together, so the result holds one value per magnitude. `q_max` defaults to six times `2 pi sqrt(N) / L`, the wavevector that matches the mean spacing between the atoms, which grows as the square root of the density. How far a nearest neighbour sits is set by the potential instead, so a dilute configuration is drawn up to a smaller multiple of its first peak, where its S(q) is close to one throughout. Two runs at different densities are drawn over different ranges; give both the same `q_max` to compare them directly.

```{code-cell} python
r, gr = simulation.trajectory[100:].rdf()
q, s = simulation.trajectory[100:].structure_factor()
fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3))
left.plot(r, gr)
left.set_xlabel("r / Angstrom")
left.set_ylabel("g(r)")
right.plot(q, s)
right.set_xlabel("q / Angstrom$^{-1}$")
right.set_ylabel("S(q)")
plt.show()
```

`msd(max_lag=None, average_origins=True)` is for a molecular dynamics trajectory. It returns the lag times in picoseconds, from one frame interval up to `max_lag`, and the mean squared displacement at each in Angstrom squared, averaged over every pair of frames that lag apart. Short lags are averaged over many origins and long lags over few, so the curve is smooth at the start and jagged at the end. With `average_origins=False`, every displacement is measured from the first frame instead. The two curves agree when the system stays the same throughout the run. Where they differ, something changed during it, such as melting, and the averaged curve describes neither state. Fit the diffusion coefficient over a window of lags that is short compared with the run; in two dimensions the mean squared displacement is `4 D t`.

```{code-cell} python
lag, msd = simulation.trajectory[100:].msd()
window = (lag >= 1) & (lag <= 5)
slope, intercept = np.polyfit(lag[window], msd[window], 1)
D = slope / 4  # Angstrom^2/ps
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(lag, msd)
ax.plot(lag[window], slope * lag[window] + intercept, "--")
ax.set_xlabel("Lag / ps")
ax.set_ylabel("MSD / Angstrom$^2$")
plt.show()
print(f"D = {D:.2f} Angstrom^2/ps")
```
