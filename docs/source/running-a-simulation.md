---
file_format: mystnb
kernelspec:
  name: python3
---

# Running a simulation

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

## The model

```{code-cell} python
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
model = dict(species=[argon], pair_potentials={(argon, argon): lj})
```

`Species` takes the mass in atomic mass units. `LennardJones` takes the well depth in joules and the zero-crossing separation in metres. `pair_potentials` maps each pair of species to the potential acting between them; one species needs one entry. `**model` passes both entries to `initialise` as keyword arguments.

## Molecular dynamics

```{code-cell} python
from pylj import sample
from pylj.md import MDSimulation

simulation = MDSimulation.initialise(number_of_atoms=16, temperature=300, box=30, seed=1, **model)
viewer = sample.Interactions(simulation)
for _ in range(2000):
    simulation.step()
    simulation.heat_bath(300)
    simulation.sample()
    if simulation.steps % 100 == 0:
        viewer.update(simulation)
```

`initialise` places the atoms on a square lattice (`init_conf="metropolis"` places them by Metropolis insertion instead), draws velocities at `temperature` in kelvin with the centre of mass at rest, and evaluates the forces. `box` is the side of the square periodic box in Angstrom. `cut_off`, in Angstrom, defaults to 15 or half the box, whichever is smaller. `timestep` defaults to `1e-14` s. `seed` makes the run reproducible.

`step()` advances one timestep with Velocity-Verlet. `heat_bath(T)` rescales the velocities so that the instantaneous temperature is `T`. `sample()` appends one entry to each array of `simulation.samples`: `step`, `temperature`, `pressure`, `potential_energy`, `kinetic_energy` and `msd`, with `total_energy` derived from them. `simulation.time` is `steps * timestep`.

```python
samples = simulation.samples
samples.temperature.mean()  # K
samples.pressure.mean()  # N/m, the two-dimensional pressure
samples.msd[-1]  # m^2, from the unwrapped positions
```

`simulation.configuration` holds the current state: `position` and `velocity` as `(N, 2)` arrays in SI units, `box` in metres, `masses` in kilograms, and the methods `kinetic_energy()`, `temperature()` and `potential_energy(pair_potentials, cut_off)`. `simulation.forces` is the net force on each atom at that configuration.

`simulation.restart()` returns a new simulation that continues from the current configuration with `steps` at zero and an empty `samples`, so that a production run can be recorded separately from equilibration.

## Monte Carlo

```{code-cell} python
from pylj.mc import MCSimulation

simulation = MCSimulation.initialise(number_of_atoms=16, temperature=300, box=20, seed=1, **model)
viewer = sample.Energy(simulation)
for _ in range(5000):
    simulation.step()
    if simulation.steps % 10 == 0:
        simulation.sample()
    if simulation.steps % 500 == 0:
        viewer.update(simulation)
```

`initialise` takes the same arguments without `timestep`. The temperature is stored as `simulation.temperature` and used by the acceptance rule. `step()` proposes moving one atom, chosen at random, to a uniform random position in the box, and accepts the move by the Metropolis criterion. `simulation.energy` is the running potential energy and `simulation.accepted` the number of accepted moves. `sample()` recomputes the energy exactly and appends `step` and `potential_energy` to `simulation.samples`.

The step can be written out:

```python
from pylj import mc

proposal = simulation.propose()
if mc.accept(proposal.energy_change, simulation.temperature, rng=simulation.rng):
    simulation.apply(proposal)
```

`propose()` returns a `Proposal` holding the trial positions and the energy change the move would cause. `mc.accept` returns `True` for a move that does not raise the energy, and otherwise with probability `exp(-energy_change / (k_B T))`. `apply()` makes the proposal the current configuration and adds its energy change to `energy`; it raises `ValueError` if the proposal was made from a configuration that is no longer current.

## Units

`initialise` takes `box` and `cut_off` in Angstrom and `temperature` in kelvin, and the viewers take a drawn `diameter` in Angstrom. Potentials take SI units and `Species` takes atomic mass units. Every quantity a simulation reports is in SI units.
