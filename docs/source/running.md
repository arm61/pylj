---
file_format: mystnb
kernelspec:
  name: python3
---

# Running a simulation

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
%config InlineBackend.figure_format = "retina"
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

A built simulation advances one step at a time. A molecular dynamics step integrates the equations of motion over one timestep, and a Monte Carlo step makes one Metropolis move. This page describes both, then restarting, which separates equilibration from a production run, and how long a run takes. The examples use the argon model from [Building a simulation](building.md):

```{code-cell} python
from pylj.md import MDSimulation
from pylj.model import Model
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
model = Model.single(argon, LennardJones(epsilon=0.9497, sigma=3.372))
```

## Molecular dynamics

`step()` advances the configuration by one timestep with Velocity-Verlet and updates `forces`, the net force on each atom, and `steps`. `time` is `steps * timestep`. `integrate()` is the method `step()` calls; a subclass overrides it to use a different integrator.

`rescale_velocities(temperature)` rescales the velocities so that the instantaneous temperature is `temperature`. Called every step, it holds the temperature fixed; the examples here call it every 50 steps, so that the temperature fluctuates between calls.

`step()` raises `ValueError` if an atom moves further than half the cut-off in one step, which means the timestep is too long or the run has diverged.

## Monte Carlo

`temperature` is the temperature the acceptance rule uses, and `max_displacement` the largest distance a move shifts an atom along each axis, in Angstrom, 0.5 by default. `energy` is the running potential energy and `accepted` the number of accepted moves.

`step()` is `propose()`, `mc.accept()` and `apply()`:

```{code-cell} python
from pylj import mc
from pylj.mc import MCSimulation

simulation = MCSimulation.initialise(model, number_of_atoms=16, temperature=300, box=20, seed=1)
proposal = simulation.propose()
if mc.accept(proposal.energy_change, simulation.temperature, rng=simulation.rng):
    simulation.apply(proposal)
```

`propose()` moves one atom, chosen at random, by a random distance of up to `max_displacement` along each axis, and returns a `Proposal` holding the trial positions, the energy change the move would cause and a copy of the positions it was made from. `mc.accept(energy_change, temperature, rng=...)` returns `True` for a move that does not raise the energy, and otherwise with probability `exp(-energy_change / (k_B T))`. `apply()` replaces the positions with the proposal's and adds the energy change to `energy`; it raises `ValueError` if the positions have changed since the proposal was made.

`accepted` counts the moves that `step()` has accepted, so after a run it gives the fraction of moves the acceptance rule took:

```{code-cell} python
simulation = MCSimulation.initialise(model, number_of_atoms=16, temperature=300, box=20, seed=1)
for _ in range(5000):
    simulation.step()
print(f"{simulation.accepted} of {simulation.steps} moves accepted")
```

## Restarting

`restart()` returns a new simulation that continues from the current configuration with `steps` at zero, an empty `samples` and `trajectory`, and a copy of the random number generator, so that a production run can be recorded separately from equilibration:

```{code-cell} python
for _ in range(1000):
    simulation.step()
production = simulation.restart()
for _ in range(5000):
    production.step()
    production.sample()
```

## Watching a run

A loop that only calls `step()` shows nothing until it finishes. Print the step count every so often to see it working:

```{code-cell} python
simulation = MDSimulation.initialise(model, number_of_atoms=16, temperature=300, box=30, seed=1)
for _ in range(4000):
    simulation.step()
    if simulation.steps % 1000 == 0:
        print(simulation.steps, simulation.configuration.temperature())
    if simulation.steps % 50 == 0:
        simulation.rescale_velocities(300)
```

How long a run takes depends on the computer, but it grows with the number of atoms in a way that can be predicted. A molecular dynamics step evaluates every pair of atoms, so its cost grows roughly with the square of the number of atoms: four times as many atoms make each step about sixteen times slower. A Monte Carlo step moves one atom and evaluates only that atom's pairs, so its cost grows only in proportion to the number of atoms, and below a few hundred atoms it barely changes. Giving every atom one move, though, takes as many Monte Carlo steps as there are atoms, and those steps together cost more than one molecular dynamics step. To estimate a long run, time a short one:

```{code-cell} python
import time

start = time.perf_counter()
for _ in range(1000):
    simulation.step()
print(f"{1000 / (time.perf_counter() - start):.0f} steps a second")
```
