---
file_format: mystnb
kernelspec:
  name: python3
---

# Molecular dynamics

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

Molecular dynamics follows the atoms through time by solving Newton's equations of motion. The loop is short: find the force on each atom from the potential, move every atom forward by one small timestep, and repeat. This chapter builds the loop one piece at a time, writes the integration step by hand, and then shows the same step inside `pylj`.

The algorithm:

1. Place the atoms and give them velocities at the chosen temperature.
2. Calculate the force on each atom.
3. Move the atoms forward by one timestep.
4. Sample whatever is being measured.
5. Go to step 2.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from pylj import md, sample
from pylj.constants import BOLTZMANN
from pylj.md import MDSimulation
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
model = dict(species=[argon], pair_potentials={(argon, argon): lj})
```

## Initialisation

### Positions

The atoms start on a square lattice by default. Sixteen atoms in a 50 Angstrom box sit on a four-by-four grid.

```{code-cell} python
simulation = MDSimulation.initialise(number_of_atoms=16, temperature=300, box=50, seed=0, **model)
viewer = sample.JustCell(simulation)
```

A lattice is a safe start because no two atoms are close enough to repel strongly. The alternative, `init_conf="metropolis"`, places the atoms one at a time at random positions and accepts each position according to its interaction energy with the atoms already placed, using the Monte Carlo rule of the next chapter. Placing the atoms purely at random, with no regard to their energy, would sometimes put two almost on top of each other. The force between them would then be enormous, the first step would fling them apart at a speed the timestep cannot follow, and the run would be meaningless from its first step. `pylj` refuses a starting configuration that stores more than ten $k_B T$ of potential energy per atom for this reason.

```{code-cell} python
placed = MDSimulation.initialise(
    number_of_atoms=16, temperature=300, box=50, init_conf="metropolis", seed=0, **model
)
viewer = sample.JustCell(placed)
```

### Velocities

Each component of each velocity is drawn from a normal distribution of width $\sqrt{k_B T / m}$, the thermal speed of an atom of mass $m$ at temperature $T$. Two corrections follow. The velocity of the centre of mass is subtracted, so the box as a whole does not drift. Then every velocity is scaled by one factor so that the temperature of the sample is exactly the one asked for.

```{code-cell} python
configuration = simulation.configuration
thermal_speed = np.sqrt(BOLTZMANN * 300 / configuration.masses[0])
print(f"thermal speed {thermal_speed:.0f} m/s")
print(
    f"root mean square speed {np.sqrt(np.mean(np.sum(configuration.velocity**2, axis=1))):.0f} m/s"
)
print(f"centre of mass velocity {np.abs(configuration.velocity.mean(axis=0)).max():.1e} m/s")
print(f"temperature {configuration.temperature():.1f} K")
```

The temperature is the kinetic energy divided by $(N - 1) k_B$, where $N$ is the number of atoms, not by $N k_B$. With the centre of mass at rest, two of the $2N$ velocity components are fixed, and the remaining $2N - 2$ each carry $k_B T / 2$ on average. Each atom has two velocity components, so the root mean square speed is $\sqrt{2 (N - 1) / N}$ times the thermal speed, which tends to $\sqrt{2}$ for many atoms; for sixteen it is 1.37.

## Forces

The force on a pair is minus the slope of the pair energy, which the previous chapter plotted. The net force on an atom is the sum of the pair forces from every other atom, each pointing along the line between them. The configuration does that sum. The code below is `pylj`'s own, shown as it is in the source. The annotations after the colons, such as `NDArray[np.float64]`, and the return type after the arrow, document the types; they are not something you type to use the function.

```{literalinclude} ../../pylj/configuration.py
:pyobject: Configuration.forces
```

`pairs` evaluates every pair once, with the atom of lower index first. The radial force times the unit separation vector is the force that pair exerts on its first atom; the second atom feels the opposite. `np.add.at` accumulates those onto the atoms. The simulation holds the result as `forces`, one two-component vector per atom, and Newton's second law turns it into accelerations. On the square lattice every atom's four nearest neighbours pull equally in four directions and the net force is zero, so the accelerations below are those of the Metropolis-placed configuration, whose atoms have uneven surroundings:

```{code-cell} python
configuration = placed.configuration
accelerations = placed.forces / configuration.masses[:, None]
print(f"largest acceleration {np.abs(accelerations).max():.2e} m/s^2")
```

## Integration

Knowing the positions, velocities and accelerations, the atoms can be moved forward in time. The integrator `pylj` uses is Velocity-Verlet. The positions advance with the current velocity and acceleration,

$$
\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \mathbf{v}(t)\,\Delta t + \tfrac{1}{2}\mathbf{a}(t)\,\Delta t^2,
$$

the forces are evaluated at the new positions, and the velocities advance with the average of the old and new accelerations,

$$
\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \tfrac{1}{2}\left[\mathbf{a}(t) + \mathbf{a}(t + \Delta t)\right]\Delta t.
$$

Written by hand from `pylj`'s two update functions, one step is:

```{code-cell} python
def verlet_step(configuration, forces, timestep, pair_potentials, cut_off):
    masses = configuration.masses[:, None]
    accelerations = forces / masses
    position, unwrapped = md.update_positions(configuration, accelerations, timestep)
    moved = configuration.replace(position=position, unwrapped=unwrapped)
    next_forces = moved.forces(pair_potentials, cut_off)
    velocity = md.update_velocities(
        configuration.velocity, accelerations, next_forces / masses, timestep
    )
    return moved.replace(velocity=velocity), next_forces
```

`update_positions` returns two arrays because the configuration keeps two copies of the positions: `position`, wrapped back into the box when an atom crosses an edge, and `unwrapped`, which is not, so that the distance an atom has travelled can be measured later. `replace` makes a new configuration with some arrays changed; a configuration is never edited in place, so the state before the step is still there to compare against.

`pylj`'s own step is the same code, with one addition:

```{literalinclude} ../../pylj/md.py
:pyobject: velocity_verlet
```

If an atom moves further than half the cut-off in one step, the timestep is too long or the run has already diverged, and the integrator stops with a message that says so. The forces evaluated at the new positions are returned and kept for the next step, which is why the simulation stores `forces` beside the configuration; step 2 of the algorithm is the last thing step 3 does. Running both on the same configuration gives the same result:

```{code-cell} python
ours, our_forces = verlet_step(
    configuration, placed.forces, placed.timestep, placed.pair_potentials, placed.cut_off
)
theirs, their_forces = md.velocity_verlet(
    configuration, placed.forces, placed.timestep, placed.pair_potentials, placed.cut_off
)
assert np.array_equal(ours.position, theirs.position)
assert np.array_equal(ours.velocity, theirs.velocity)
print(
    f"the fastest atom moved {np.linalg.norm(ours.position - configuration.position, axis=1).max() * 1e10:.4f} Angstrom"
)
```

The timestep is ten femtoseconds by default. An atom at the thermal speed moves about 0.025 Angstrom in that time, and the fastest atom about twice that, a small fraction of the distance over which the force changes, which is what the integrator needs.

## The loop

`step()` integrates one timestep and advances the clock; `sample()` records the temperature, pressure, potential and kinetic energies and the mean squared displacement at the current step. The viewer redraws on request, and drawing is the slowest part, so the loop draws every fiftieth step.

```{code-cell} python
simulation = MDSimulation.initialise(number_of_atoms=16, temperature=300, box=50, seed=0, **model)
viewer = sample.Interactions(simulation)
for _ in range(2000):
    simulation.step()
    simulation.sample()
    if simulation.steps % 50 == 0:
        viewer.update(simulation)
```

The total energy is potential plus kinetic. Nothing adds or removes energy in this loop, so the total should be constant. The energy pane above rescales its axis to the data, so the whole height of that panel is a few parts in a thousand of the energy, the small error of the integrator. Plotting the two parts separately shows them trading against each other as pairs approach and separate:

```{code-cell} python
s = simulation.samples
time = s.step * simulation.timestep * 1e12
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(time, s.potential_energy * 1e21, label="potential")
ax.plot(time, s.kinetic_energy * 1e21, label="kinetic")
ax.plot(time, s.total_energy * 1e21, label="total")
ax.set_xlabel("t / ps")
ax.set_ylabel("E / zJ")
ax.legend()
fig.tight_layout()
```

```{code-cell} python
drift = (s.total_energy.max() - s.total_energy.min()) / abs(s.total_energy.mean())
print(f"total energy varies by {drift:.1e} of its value over the run")
print(f"mean temperature {s.temperature.mean():.0f} K")
```

## Sampling

`samples` holds one array per measured quantity, and `step` says when each was taken, so a loop may sample as often or as rarely as it likes. The mean squared displacement measures how far atoms have travelled from where they started, using the unwrapped positions. While an atom flies freely its displacement grows with time, so the mean squared displacement grows with time squared and the curve bends upward. Once the atoms have collided many times the curve straightens into a line whose slope gives the diffusion coefficient. In two dimensions the diffusion coefficient $D$ is the slope divided by four, because the mean squared displacement is $\mathrm{MSD} = 4 D t$. Sixteen atoms in a 50 Angstrom box are dilute, and in these 20 picoseconds most have not yet collided, so the curve here is still bending upward.

```{code-cell} python
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(s.step * simulation.timestep * 1e12, s.msd * 1e20)
ax.set_xlabel("t / ps")
ax.set_ylabel("MSD / Angstrom$^2$")
fig.tight_layout()
```

## The thermostat

The run above conserves energy, so its temperature drifts from 300 K as the atoms leave the lattice and fall into each other's attractive wells, turning potential energy into kinetic. To hold a temperature, `pylj` rescales the velocities. `heat_bath(T)` multiplies every velocity by $\sqrt{T / T_{\text{now}}}$, which sets the instantaneous temperature to $T$ exactly:

```{literalinclude} ../../pylj/md.py
:pyobject: heat_bath
```

Called every step it is a crude thermostat: it removes the natural fluctuations of the temperature, and averages taken under it are close to, but not exactly, those of a system at that temperature. It is simple and it holds the target, which is what the ideal gas law chapter needs.

```{code-cell} python
simulation = MDSimulation.initialise(number_of_atoms=16, temperature=300, box=50, seed=0, **model)
for _ in range(2000):
    simulation.step()
    simulation.heat_bath(300)
    simulation.sample()
print(f"mean temperature {simulation.samples.temperature.mean():.1f} K")
```

## Your own integrator

This section writes a Python class. Skip it on a first reading; nothing later depends on it.

The integrator is one method on the simulation, `integrate`, which replaces the configuration and the forces. A subclass that overrides it runs any integrator under the same loop, viewers and samples. The hand-written step from above, installed this way, runs in place of `pylj`'s:

```{code-cell} python
class HandWritten(MDSimulation):
    def integrate(self):
        self.configuration, self.forces = verlet_step(
            self.configuration, self.forces, self.timestep, self.pair_potentials, self.cut_off
        )


ours = HandWritten.initialise(number_of_atoms=16, temperature=300, box=50, seed=0, **model)
theirs = MDSimulation.initialise(number_of_atoms=16, temperature=300, box=50, seed=0, **model)
for _ in range(100):
    ours.step()
    theirs.step()
assert np.array_equal(ours.configuration.position, theirs.configuration.position)
print("100 steps, identical trajectories")
```

`verlet_step` has no half-cut-off check, so a run that goes wrong under `HandWritten` does not stop: the positions become meaningless and nothing raises an error unless the potential sets `min_separation`, as `Buckingham` does.

The ideal gas law chapter uses this loop to measure the pressure of argon and test the ideal gas law. The next chapter reaches equilibrium properties by a different route, with no velocities and no clock.
