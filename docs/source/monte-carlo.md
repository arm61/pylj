---
file_format: mystnb
kernelspec:
  name: python3
---

# Monte Carlo

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

Monte Carlo reaches the same equilibrium properties as molecular dynamics by a different route. There are no velocities and no clock. Instead the method proposes a change to the configuration, accepts or rejects it by a rule that depends on the change in energy and the temperature, and repeats. The configurations it visits are distributed according to the Boltzmann probability, so averages over them are equilibrium averages. This chapter builds the loop from the energy up, and then shows it inside pylj.

The algorithm:

1. Place the particles.
2. Calculate the energy of the configuration.
3. Propose a change.
4. Calculate the energy change it would cause.
5. Accept the change with the Metropolis rule, or leave the configuration as it is.
6. Sample whatever is being measured.
7. Go to step 3.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from pylj import mc, sample
from pylj.constants import BOLTZMANN
from pylj.mc import MCSimulation
from pylj.potentials import LennardJones, Species, SquareWell

argon = Species(mass=39.948, name="argon")
lj = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
model = dict(species=[argon], pair_potentials={(argon, argon): lj})
```

## Initialisation

Placement is as for molecular dynamics: a square lattice, or Metropolis insertion with `init_conf="metropolis"`. There are no velocities to draw. The temperature is a parameter of the acceptance rule rather than a property of the particles, so the simulation stores it.

```{code-cell} python
simulation = MCSimulation.initialise(16, 300, 20, seed=0, **model)
viewer = sample.JustCell(simulation)
print(f"temperature {simulation.temperature} K")
```

## The energy

The energy of a configuration is the sum of its pair energies. Sixteen particles make $N(N-1)/2 = 120$ pairs, and `pairs` evaluates all of them:

```{code-cell} python
pairs = simulation.configuration.pairs(simulation.pair_potentials, simulation.cut_off)
print(pairs.energy.size, "pairs")
print(f"energy {pairs.energy.sum() * 1e21:.2f} zJ")
```

Pairs further apart than the cut-off, here half the 20 Angstrom box, contribute zero. The simulation keeps a running total, `energy`, so that it does not need to sum every pair after every move:

```{code-cell} python
print(f"stored energy {simulation.energy * 1e21:.2f} zJ")
```

## A proposal

The change pylj proposes is the simplest: one particle, chosen at random, is given a new position drawn uniformly from the box. `propose` returns the proposed positions and the energy change the move would cause, and leaves the configuration untouched. It does steps 3 and 4 of the algorithm together.

```{literalinclude} ../../pylj/mc.py
:pyobject: MCSimulation.propose
```

The energy change is found without re-evaluating every pair. Only the moved particle's interactions change, so its energy with the others at the new position, minus its energy with them at the old position, is the change. `insertion_energy` is that sum for one particle against a configuration, and `without` is the configuration with the particle taken out.

A proposal also records the configuration it was made from, because its energy change is measured against that configuration. Applying it after another move has changed the configuration would add an energy change that no longer applies, so `apply` refuses it. Propose again after each accepted move.

```{code-cell} python
proposal = simulation.propose()
moved = np.flatnonzero(np.any(proposal.position != simulation.configuration.position, axis=1))
print(f"particle {moved[0]} would move to ({proposal.position[moved[0], 0] * 1e10:.2f}, {proposal.position[moved[0], 1] * 1e10:.2f}) Angstrom")
print(f"energy change {proposal.energy_change * 1e21:+.2f} zJ")
```

## The Metropolis rule

Accepting only moves that lower the energy would find a minimum and stop. The particles should instead visit configurations with a probability proportional to the Boltzmann factor, $\exp(-E / k_B T)$, which allows the energy to rise. The Metropolis rule does this: a move that does not raise the energy is always accepted, and a move that raises it by $\Delta E$ is accepted with probability

$$
\exp\!\left(-\frac{\Delta E}{k_B T}\right),
$$

by drawing a uniform random number $n$ between 0 and 1 and accepting if $n$ is below that probability. The rule is short:

```{literalinclude} ../../pylj/mc.py
:pyobject: accept
```

The probability falls off with the size of the rise measured against $k_B T$, so at higher temperature larger rises are accepted more often:

```{code-cell} python
rise = np.linspace(0, 6e-21, 100)
fig, ax = plt.subplots(figsize=(4, 3))
for temperature in (100, 300, 1000):
    ax.plot(rise * 1e21, np.exp(-rise / (BOLTZMANN * temperature)), label=f"{temperature} K")
ax.set_xlabel("energy rise / zJ")
ax.set_ylabel("acceptance probability")
ax.legend()
fig.tight_layout()
```

## The loop

A step is: propose, decide, and apply if accepted. `apply` makes the proposed positions the configuration and adds the energy change to the running total. A rejected proposal is dropped and the configuration is unchanged. The loop below is written by hand, and draws from the simulation's own random number generator so the run is reproducible. `sample()` records the energy the viewer plots; consecutive configurations differ by one particle, so sampling every tenth step loses little, and the viewer redraws every two hundred:

```{code-cell} python
simulation = MCSimulation.initialise(16, 300, 20, seed=0, **model)
viewer = sample.Energy(simulation)
accepted = 0
for _ in range(2000):
    proposal = simulation.propose()
    if mc.accept(proposal.energy_change, simulation.temperature, rng=simulation.rng):
        simulation.apply(proposal)
        accepted += 1
    simulation.steps += 1
    if simulation.steps % 10 == 0:
        simulation.sample()
    if simulation.steps % 200 == 0:
        viewer.update(simulation)
print(f"accepted {accepted} of {simulation.steps} moves")
```

The energy pane shows the run settling. The lattice is not an equilibrium configuration: the energy falls over the first few hundred steps as the particles find each other's wells, and then fluctuates about a steady value. Sixteen particles fluctuate hard, so the fall is easier to see in an average than by eye. Averages are taken over the steady part, and the settling steps are discarded.

`step()` does the same three things, and counts the accepted moves as `accepted`:

```{literalinclude} ../../pylj/mc.py
:pyobject: MCSimulation.step
```

```{code-cell} python
simulation = MCSimulation.initialise(16, 300, 20, seed=0, **model)
for _ in range(2000):
    simulation.step()
    if simulation.steps % 10 == 0:
        simulation.sample()
print(f"accepted {simulation.accepted} of {simulation.steps} moves")
```

The two runs accept the same number of moves because they draw the same random numbers in the same order. About one move in eight is accepted: a uniform trial position usually lands the particle near another one, where the energy rise is large.

`sample()` records the step and the potential energy, and before recording it recomputes the energy exactly from every pair, so that rounding in the running total cannot accumulate over a long run.

## The two methods agree

The molecular dynamics chapter held a temperature with a thermostat; the Monte Carlo rule holds it by construction. Run both on the same sixteen particles in the same box at 300 K, discard the settling steps, and compare the mean potential energy over the rest. A Monte Carlo step moves one particle while a dynamics step moves them all, so the Monte Carlo run is given twice as many steps:

```{code-cell} python
from pylj.md import MDSimulation

dynamics = MDSimulation.initialise(16, 300, 20, seed=0, **model)
for _ in range(2000):
    dynamics.step()
    dynamics.heat_bath(300)
production = dynamics.restart()
for _ in range(10000):
    production.step()
    production.heat_bath(300)
    if production.steps % 10 == 0:
        production.sample()
by_dynamics = production.samples.potential_energy.mean()

monte_carlo = MCSimulation.initialise(16, 300, 20, seed=0, **model)
for _ in range(2000):
    monte_carlo.step()
production = monte_carlo.restart()
for _ in range(20000):
    production.step()
    if production.steps % 10 == 0:
        production.sample()
by_monte_carlo = production.samples.potential_energy.mean()

print(f"mean potential energy by molecular dynamics {by_dynamics * 1e21:.2f} zJ")
print(f"mean potential energy by Monte Carlo        {by_monte_carlo * 1e21:.2f} zJ")
print(f"difference {abs(by_dynamics - by_monte_carlo) / abs(by_monte_carlo):.1%}")
```

`restart()` begins a fresh record from the current configuration, so the samples cover only the settled run. The two methods reach the same average by different routes: one by following the motion, the other by drawing configurations with the Boltzmann weight. The remaining difference does not shrink with longer runs. Sixteen particles is a small system, and rescaling the velocities at every step holds the temperature in a slightly different way from the Metropolis rule; with more particles and a gentler thermostat the two draw closer.

## A potential only Monte Carlo can use

The square well is a hard core of diameter $\sigma$ surrounded by a well of depth $\epsilon$ out to $\lambda \sigma$. Its energy changes in steps, so its force is zero everywhere except at the two walls, where it is infinite. Molecular dynamics cannot integrate that. Monte Carlo needs only energies, and a proposal that puts two particles inside the core has an infinite energy change, which the rule always rejects.

```{code-cell} python
well = SquareWell(epsilon=1.5e-21, sigma=3e-10, lambda_=1.5)
well_model = dict(species=[argon], pair_potentials={(argon, argon): well})
simulation = MCSimulation.initialise(25, 300, 30, seed=0, **well_model)
for _ in range(5000):
    simulation.step()
viewer = sample.RDF(simulation)
for _ in range(20000):
    simulation.step()
    if simulation.steps % 200 == 0:
        viewer.update(simulation)
viewer.average()
print(f"accepted {simulation.accepted} of {simulation.steps} moves")
```

The viewer is built after five thousand settling steps, so its average covers only the settled run. The radial distribution function is zero inside 3 Angstrom, where the core forbids any pair, highest just outside it, where the well holds pairs together, and steps down at 4.5 Angstrom, the outer edge of the well. The axis is in metres, with a factor of 1e-9 printed in its corner, so 3 Angstrom reads as 0.3.

Monte Carlo gives equilibrium averages without dynamics. When the question is how fast something happens, the molecular dynamics chapter's method is the one to use; when the question is what the equilibrium looks like, either method answers it, and Monte Carlo answers it for potentials with no force at all. The next chapter takes molecular dynamics back out and uses it to measure the pressure of argon against the ideal gas law.
