# Simulations

`MDSimulation` and `MCSimulation` share a base class, `Simulation`, and are built the same way. Both hold `configuration`, `pair_potentials`, `cut_off`, `rng`, `steps` and `samples`, and both have `step()`, `sample()` and `restart()`.

## Building a simulation

```python
simulation = MDSimulation.initialise(
    number_of_atoms=16,
    temperature=300,
    box=30,
    species=[argon],
    pair_potentials={(argon, argon): lj},
    init_conf="square",
    placement_temperature=None,
    timestep=1e-14,
    cut_off=None,
    seed=None,
)
```

Every argument is given by keyword. `number_of_atoms`, `temperature` in kelvin and `box`, the side of the square periodic box in Angstrom, are required, as are `species`, a sequence of `Species`, and `pair_potentials`, a mapping from each pair of species to the potential between them. `init_conf` is `"square"`, a square lattice, or `"metropolis"`, Metropolis insertion of one atom at a time at `placement_temperature`, which defaults to `temperature`. `timestep` is in seconds and applies to molecular dynamics only. `cut_off` is in Angstrom and defaults to 15 or half the box, whichever is smaller; it may not exceed half the box. `seed` seeds `simulation.rng`, which draws the initial velocities and the Monte Carlo moves.

A configuration whose pair energy is not finite or exceeds ten $k_B T$ per atom is refused with `ValueError`, as is a potential whose energy at the cut-off is larger than $k_B T$.

The constructors take a ready configuration instead: `MDSimulation(configuration, pair_potentials, cut_off=None, timestep=1e-14, seed=None)` with an `MDConfiguration`, and `MCSimulation(configuration, pair_potentials, temperature, cut_off=None, seed=None)` with a `Configuration`, all in SI units.

## The configuration

`simulation.configuration` is the current state. `position` is an `(N, 2)` array in metres, `box` the side in metres, `species` and `species_index` name each atom's species, and `masses` is in kilograms. `pairs(pair_potentials, cut_off, forces=False)` evaluates every pair and returns their distances, separations, energies and, if asked, radial forces; `potential_energy`, `forces` and `virial` take the same arguments. An `MDConfiguration` adds `velocity` and `unwrapped`, the positions without periodic wrapping, and `kinetic_energy()`, `temperature()`, which divides the kinetic energy by $(N - 1) k_B$ because the centre of mass is held at rest, and `msd(initial)`.

A configuration is never changed in place. `replace(**changes)` returns a copy with some arrays changed.

## Molecular dynamics

`step()` advances the configuration by one timestep with Velocity-Verlet and updates `forces`, the net force on each atom, and `steps`. `time` is `steps * timestep`. `integrate()` is the method `step()` calls; a subclass overrides it to use a different integrator.

`heat_bath(bath_temperature)` rescales the velocities so that the instantaneous temperature is `bath_temperature`.

`sample()` appends one entry to each array of `samples`, an `MDSamples`: `step`, `temperature`, `pressure`, `potential_energy`, `kinetic_energy` and `msd`, with `total_energy` derived from them. The pressure is the virial pressure, `(2 K + sum(f r)) / (2 L^2)`, in newtons per metre. All are in SI units.

`step()` raises `ValueError` if an atom moves further than half the cut-off in one step, which means the timestep is too long or the run has diverged.

## Monte Carlo

`temperature` is the temperature the acceptance rule uses. `energy` is the running potential energy and `accepted` the number of accepted moves.

`step()` is `propose()`, `mc.accept()` and `apply()`:

```python
from pylj import mc

proposal = simulation.propose()
if mc.accept(proposal.energy_change, simulation.temperature, rng=simulation.rng):
    simulation.apply(proposal)
```

`propose()` moves one atom, chosen at random, to a uniform random position in the box, and returns a `Proposal` holding the trial positions, the energy change the move would cause and the configuration it was made from. `mc.accept(energy_change, temperature, rng=...)` returns `True` for a move that does not raise the energy, and otherwise with probability `exp(-energy_change / (k_B T))`. `apply()` makes the proposal the current configuration and adds its energy change to `energy`; it raises `ValueError` if the proposal was made from a configuration that is no longer current.

`sample()` recomputes the energy exactly and appends `step` and `potential_energy` to `samples`, an `MCSamples`.

## Restarting

`restart()` returns a new simulation that continues from the current configuration with `steps` at zero, an empty `samples` and a copy of the random number generator, so that a production run can be recorded separately from equilibration:

```python
for _ in range(1000):
    simulation.step()
production = simulation.restart()
for _ in range(5000):
    production.step()
    production.sample()
```

## Units

`initialise` takes `box` and `cut_off` in Angstrom and `temperature` in kelvin, and the viewers take a drawn `diameter` in Angstrom. Potentials take SI units and `Species` takes atomic mass units. Every quantity a simulation reports is in SI units.
