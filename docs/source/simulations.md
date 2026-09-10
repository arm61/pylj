# Simulations

`MDSimulation` and `MCSimulation` share a base class, `Simulation`, and are built the same way. Both hold `configuration`, `model`, `cut_off`, `rng`, `steps`, `samples` and `trajectory`, and both have `step()`, `sample()` and `restart()`.

## Building a simulation

```python
simulation = MDSimulation.initialise(
    model,
    number_of_atoms=16,
    temperature=300,
    box=30,
    init_conf="square",
    placement_temperature=None,
    timestep=1e-14,
    cut_off=None,
    seed=None,
)
```

`model`, a `Model` of the species and the potential between each pair of them, is the one positional argument; the rest are given by keyword. Three are required: `number_of_atoms`; `temperature`, in kelvin; and `box`, the side of the square periodic box in Angstrom. `init_conf` selects the starting positions: `"square"` places the atoms on a square lattice, `"triangular"` on a triangular one, and `"metropolis"` inserts them one at a time by Metropolis acceptance at `placement_temperature`, which defaults to `temperature`. A triangular lattice fills the box, so the number of atoms has to be a number of columns times an even number of rows. Dividing the columns by the rows has to give something near `sqrt(3) / 2`, and `max_strain` is how far away it may sit, as a fraction of that ratio. At the default of 0.05 the counts up to 300 are 30, 56, 90, 120, 168, 224, 270 and 288, and any other count raises `ValueError` naming ones that fit. `timestep` is in seconds and applies to molecular dynamics only. `cut_off` is in Angstrom and defaults to 15 or half the box, whichever is smaller; it may not exceed half the box. `seed` seeds `simulation.rng`, which draws the initial velocities and the Monte Carlo moves.

A configuration whose pair energy is not finite or exceeds ten $k_B T$ per atom is refused with `ValueError`, as is a potential whose energy at the cut-off is larger than $k_B T$.

The constructors take a ready configuration instead: `MDSimulation(configuration, model, cut_off=None, timestep=1e-14, seed=None)` with an `MDConfiguration`, and `MCSimulation(configuration, model, temperature, cut_off=None, seed=None)` with a `Configuration`, all in SI units.

## The configuration

`simulation.configuration` is the current state. `position` is an `(N, 2)` array in metres, `box` the side in metres, `species` and `species_index` name each atom's species, and `masses` is in kilograms. `pairs(model, cut_off, forces=False)` evaluates every pair and returns their distances, separations, energies and, if asked, radial forces; `potential_energy`, `forces` and `virial` take the same arguments. An `MDConfiguration` adds `velocity` and `unwrapped`, the positions without periodic wrapping, and `kinetic_energy()`, `temperature()`, which divides the kinetic energy by $(N - 1) k_B$ because the centre of mass is held at rest, and `msd(initial)`.

A configuration is never changed in place. `replace(**changes)` returns a copy with some arrays changed.

## Molecular dynamics

`step()` advances the configuration by one timestep with Velocity-Verlet and updates `forces`, the net force on each atom, and `steps`. `time` is `steps * timestep`. `integrate()` is the method `step()` calls; a subclass overrides it to use a different integrator.

`heat_bath(bath_temperature)` rescales the velocities so that the instantaneous temperature is `bath_temperature`.

`sample()` records the configuration in `trajectory` and appends one entry to each array of `samples`, an `MDSamples`: `step`, `temperature`, `pressure`, `potential_energy`, `kinetic_energy` and `msd`, with `total_energy` derived from them. The pressure is the virial pressure, `(2 K + sum(f r)) / (2 L^2)`, in newtons per metre. All are in SI units.

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

`sample()` records the configuration in `trajectory`, recomputes the energy exactly and appends `step` and `potential_energy` to `samples`, an `MCSamples`.

## Restarting

`restart()` returns a new simulation that continues from the current configuration with `steps` at zero, an empty `samples` and `trajectory`, and a copy of the random number generator, so that a production run can be recorded separately from equilibration:

```python
for _ in range(1000):
    simulation.step()
production = simulation.restart()
for _ in range(5000):
    production.step()
    production.sample()
```

## Watching a run

A loop that only calls `step()` shows nothing until it finishes. Print the step count every so often to see it working:

```python
for _ in range(20000):
    simulation.step()
    simulation.heat_bath(300)
    if simulation.steps % 2000 == 0:
        print(simulation.steps, simulation.configuration.temperature())
```

A viewer does this too: the figure redraws every time `update()` is called.

Molecular dynamics manages a few thousand steps a second for twenty-five atoms, about a thousand for a hundred, and under a hundred for four hundred. Twenty thousand steps of four hundred atoms therefore takes minutes. Monte Carlo manages ten to twenty thousand a second at any of those sizes, because a step moves one atom.

## Trajectory

`sample()` also records the current configuration in `simulation.trajectory`, one frame per sample. A frame is a `Configuration`, so `simulation.trajectory[-1].position` is the last sampled positions, and `simulation.trajectory.position` is every frame's, an array of shape `(frames, N, 2)`. Slicing gives a trajectory, so `simulation.trajectory[100:]` is the run after the first hundred frames. `restart()` starts an empty trajectory.

A configuration is kept only if it was sampled, so memory grows with the number of samples and not with the number of steps. A molecular dynamics frame takes about fifty bytes per atom: a hundred atoms sampled a thousand times is five megabytes, and sampled a hundred thousand times is half a gigabyte.

Two analyses are computed on demand, from one frame or averaged over a trajectory. `rdf(bins=100, r_max=None)` returns the bin centres in metres and g(r), which is one where the atoms are spread as evenly as an ideal gas; `r_max` defaults to half the box. `structure_factor(q_max=None)` returns the wavevector magnitudes in inverse metres and S(q) at each, which is one where the atoms are spread as evenly as an ideal gas. S(q) is evaluated at the wavevectors `2 pi (h, k) / L` commensurate with the box, for integer `h` and `k` not both zero. The amplitude of a wavevector is the sum of `exp(i q . r)` over the atoms, and S(q) is the square of its modulus divided by the number of atoms. Wavevectors of equal magnitude are averaged together, so the result holds one value per magnitude. `q_max` defaults to six times `2 pi sqrt(N) / L`, the wavevector that matches the mean spacing between the atoms, which grows as the square root of the density. How far a nearest neighbour sits is set by the potential instead, so a dilute configuration is drawn up to a smaller multiple of its first peak, where its S(q) is close to one throughout. Two runs at different densities are drawn over different ranges; give both the same `q_max` to compare them directly.

```python
r, gr = simulation.configuration.rdf()   # the configuration now
r, gr = simulation.trajectory[100:].rdf()  # averaged over the run after equilibration
q, s = simulation.trajectory.structure_factor()
```

## Units

`initialise` takes `box` and `cut_off` in Angstrom and `temperature` in kelvin, and the viewers take a drawn `diameter` in Angstrom. Potentials take SI units and `Species` takes atomic mass units. Every quantity a simulation reports is in SI units.
