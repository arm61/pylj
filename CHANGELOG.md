# Changelog

All notable changes to pylj are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `simulation.Samples`, the record a simulation's `sample()` appends to, with one array per quantity and `step`, whose `add()` takes exactly one value per array so the arrays stay aligned; `md.MDSamples` adds `temperature`, `pressure`, `potential_energy`, `kinetic_energy`, `msd` and the derived `total_energy`; `mc.MCSamples` adds `potential_energy`. A simulation holds it as `samples`.
- `pylj.configuration`: `Configuration`, an immutable snapshot of `position` (shape `(N, 2)`, metres), `species`, `species_index` and `box`, with `masses` in kilograms, `pairs`, `potential_energy`, `forces`, `virial` and `insertion_energy` taking the pair potentials and the cut-off explicitly, and `replace` and `without` for copies; `MDConfiguration`, adding `velocity` and `unwrapped` with `kinetic_energy`, `temperature` and `msd`; and `PairData`, the per-pair `distance`, `separation`, `energy` and `radial_force` arrays with the `virial` derived from them. `pylj.placement`: `place_square` and `place_metropolis`. `pylj.simulation`: `Simulation`, the base class with `configuration`, `pair_potentials`, `cut_off`, `rng`, `steps`, `samples` and `restart()`.
- `md.MDSimulation` and `mc.MCSimulation`. `initialise(number_of_particles, temperature, box, *, species, pair_potentials, init_conf='square', placement_temperature=None, cut_off=None, seed=None)` builds one from a model, with the box and the cut-off in Angstrom (`MDSimulation.initialise` also takes `timestep`); the constructors take a ready configuration in SI units. `step()` advances the simulation and `steps`; `sample()` records the samples; `restart()` continues from the current configuration. `MDSimulation` has `forces`, the net force on each particle, `timestep`, `time` (`steps * timestep`), `integrate()`, which applies Velocity-Verlet and which a subclass overrides for a different integrator, and `heat_bath()`. `MCSimulation` has `temperature`, `energy`, `accepted`, `propose()` and `apply()`.
- A `cut_off` larger than half the box raises `ValueError`.
- `pylj.sample` is a package: pane classes, one per plot, and a `Viewer` that lays out a list of panes. The eight viewer names are subclasses of it, and custom viewers combine panes or add a new one.
- `samples.step` on every simulation, recorded by `sample()`, so viewers work at any sampling cadence.
- `pylj.constants`, taking the Boltzmann constant and the atomic mass unit from `scipy.constants`.
- ruff and mypy configuration in `pyproject.toml`, a `dev` extra, and continuous integration on Python 3.11 to 3.14.
- `seed` on `MDSimulation.initialise`, `MCSimulation.initialise` and the simulation constructors, and `Simulation.rng`, the `numpy.random.Generator` that places an initial configuration, draws the initial velocities and makes Monte Carlo moves. The same seed reproduces the same run.
- `pylj.potentials`: `Species`, a frozen dataclass of mass and name; the `PairPotential` interface, `energies(dr)` and `forces(dr)` on an array of separations, the force being the signed radial `-dE/dr`; and `LennardJones(*, epsilon, sigma)`, `Buckingham(*, a, b, c)` and `SquareWell(*, epsilon, sigma, lambda_, max_val)`, keyword-only with physical parameters (#57). `Species` rejects a non-positive or non-finite mass, `potentials.check_positive_finite` is the shared check, and each potential rejects a parameter that is not positive and finite (`Buckingham` allows `c` of zero, and `SquareWell` needs `lambda_` greater than one). `LennardJones` is infinite at zero separation. Every potential has `min_separation`, the separation below which it is not to be trusted, zero unless the potential sets it; a configuration gives a pair closer than that infinite energy, and raises if asked for its force. `Buckingham` sets it to the top of its short-range barrier, below which the formula falls to minus infinity, and its `energies` and `forces` are the formula at every separation; a `Buckingham` whose energy is still rising at 100 Angstrom, so that no barrier holds its particles apart, raises `ValueError`.
- `pairwise.pair_potential`, `pairwise.species_pairs` and `pairwise.minimum_image`.
- `CellPane(diameter=...)` and a `diameter` keyword on every named viewer, in Angstrom, one value or one per species; by default particles are drawn at the separation of the minimum of their species' own pair energy.
- `mc.accept`, the Metropolis criterion on an energy change, and `mc.Proposal`, a proposed configuration with its energy change.
- `MDSimulation` and `MCSimulation` refuse an initial configuration whose pair energy is not finite or stores more than `simulation.INITIAL_ENERGY_LIMIT` (ten) k_B T per particle, as an overlapping lattice does; the message gives the stored energy per particle in k_B T.
- `MDSimulation` and `MCSimulation` refuse a pair potential whose energy at the cut-off is not finite or is larger in magnitude than k_B T, since the cut-off assumes the interaction has died away there; parameters in the wrong units are one way to trip it.
- `placement_temperature` on `MDSimulation.initialise` and `MCSimulation.initialise`: the temperature of the Metropolis acceptance used to place an initial configuration, by default the run temperature.

### Changed

- A box length outside 4 to 600 Angstrom raises `ValueError` rather than `AttributeError`.
- `md.heat_bath(configuration, bath_temperature)` returns the configuration with its velocities rescaled so that the instantaneous temperature is the bath temperature; it previously took the temperature sample array and rescaled towards its cumulative mean. A non-positive bath temperature, or a configuration at rest or with a non-finite temperature, raises `ValueError` (#76).
- Python 3.11 or later is required. scipy is a dependency; Cython is not.
- The initialisers compute the initial forces, so the first integration step uses real accelerations.
- Viewers are built before their display is opened, and a viewer whose panes need molecular dynamics samples refuses a Monte Carlo simulation.
- Viewers and panes take a simulation and read its `configuration` and `samples`; the radial distribution and scattering panes compute the pair distances when they draw, and the scattering pane includes the self-scattering term `N` in the Debye sum, so the intensity is never negative. The energy pane plots the total energy, potential plus kinetic, for a molecular dynamics simulation; the `Interactions` viewer shows it in place of the force pane.
- The radial distribution function is normalised by the ideal-gas shell count with r at bin centres; the speed histogram is drawn in its own bins; the pressure axis is labelled in N m^-1.
- `JustCell` no longer takes a `scale` argument. `Viewer.average()` raises on a viewer whose panes keep no history. `CellPlus.update` rejects half-supplied custom data.
- The atomic mass unit used for initial velocities is the CODATA value; initial velocities and computed temperatures move by up to 4e-5 relative.
- Pair distances and forces are computed with vectorised NumPy. `pairwise.dist(position, box)` takes `(N, 2)` positions and returns the distances and the `(M, 2)` separations; `pairwise.calculate_pressure(virial, box, kinetic_energy)` is the instantaneous virial pressure, `(2 K + sum(f r)) / (2 L^2)`, whose kinetic term averages `(N - 1) k_B T / L^2` because the centre of mass is held at rest; it previously used the sampled temperature, which is defined over `N - 1` degrees of freedom, with `N` in the ideal term.
- `md.velocity_verlet(configuration, forces, timestep, pair_potentials, cut_off)` returns the next configuration and the forces at it, and raises `ValueError` if a particle moves further than half the cut-off in one step, which means the timestep is too long or the run has diverged; `md.update_positions(configuration, accelerations, timestep)` and `md.update_velocities(velocity, accelerations, next_accelerations, timestep)` work on `(N, 2)` arrays.
- `MDConfiguration.unwrapped` holds the positions without periodic wrapping, advanced by the integrator, and `msd` reads them (#74).
- `MDConfiguration.temperature` divides the kinetic energy by (N - 1) k_B, since the 2N - 2 velocity components left once the centre-of-mass motion is removed carry k_B T / 2 each; the energy pane's kinetic term follows. Initial velocities are drawn from a normal distribution. `MDSimulation.initialise` requires at least two particles.
- `MDSimulation.initialise` and `MCSimulation.initialise` take keyword-only `species`, a sequence of `Species`, and `pair_potentials`, a mapping from each pair of species (in either order, including each species with itself) to a `PairPotential`, in place of `mass`, `constants`, `forcefield` and `diameter`. Particles are assigned to the species in turn; `Configuration.masses` holds each particle's mass, and `Configuration.species` with `Configuration.species_index` names each particle's species. A missing pair raises `ValueError`; a potential class in place of an instance raises `TypeError`. A non-positive or non-finite temperature, and a cross pair supplied in both orders, each raise `ValueError`.
- Initial velocities are drawn with each particle's own thermal width and the mass-weighted centre-of-mass velocity is removed.
- The Monte Carlo loop proposes a configuration and applies it on acceptance, leaving the configuration untouched otherwise; a trial move costs O(N). `MCSimulation.sample` recomputes the energy exactly before recording.
- `mc.Proposal` holds `position`, shape `(N, 2)`, in place of `xposition` and `yposition`, and `source`, the configuration it was proposed from; `MCSimulation.apply` refuses a proposal whose `source` is no longer the current configuration.
- `init_conf` is a keyword argument, `'square'` by default, taking `'square'` or `'metropolis'`; an unknown value raises `ValueError`. `'metropolis'` seats particles by sequential Metropolis insertion, each trial position accepted on its interaction energy with the particles already placed, in place of the `'random'` rejection-sampled placement; it works for any potential, including a hard core. `'square'` places on the lattice without an overlap check.

### Fixed

- `Energy` and `Phase` viewers crashed under NumPy 2.
- `Interactions`, `Phase` and `Scattering` crashed if built before the first sample.
- Two-type systems drew the other type's particles at the origin.
- Pair energies in multi-type systems were counted once per type pair (#81).
- The Metropolis criterion, `mc.accept`, draws a fresh random number for every uphill change (a downhill change is accepted without a draw); one was reused for the life of the process (#78).
- The square-well hard core tested epsilon rather than sigma (part of #80).
- A diameter given in metres, a non-positive or non-finite diameter, or a potential with no energy minimum to size the particles by is refused with a clear message.
- `'metropolis'` initial configurations do not overlap at the placement temperature (#82).
- A pair potential's energy and force stored their result on the potential, overwriting the method and breaking a second call on the same instance (#79).
- The Buckingham potential raised under NumPy 2.1 or later for an integer, 0-d array or non-float NumPy scalar separation, and the Lennard-Jones and square-well forms failed on integer input (#83).
- The mean squared displacement was wrong unless sampled on every integration step, and non-zero before the first step (#74).
- Initial velocities carried a centre-of-mass drift that never decayed and added a ballistic term to the mean squared displacement, and were not at the requested temperature (#75).
- The square-well potential drives a Monte Carlo simulation: its energies are evaluated without asking for a force, and `MDSimulation.initialise` refuses it with a clear message (#80).

### Removed

- `pylj.util` and `System`; the structured particle array and `particle_dt`, whose integer `types` field `Configuration.species_index` replaces.
- `md.initialise`, `mc.initialise`, `md.initialize`, `mc.initialize`, `md.sample` and `mc.sample`; `md.calculate_temperature` and `md.calculate_msd`, which are methods on the configuration now.
- `pairwise.compute_force`, `pairwise.update_accelerations` and `pairwise.heat_bath`.
- The `temperature_sample`, `pressure_sample`, `force_sample`, `msd_sample` and `energy_sample` arrays, which `samples` replaces, and the sum of the pair forces, which has no physical meaning.
- `pylj/sample.py`, `point_size` on forcefields, `pairwise.create_dist_identifiers`, `MANIFEST.in`, and the Code Climate upload from CI.
- `pairwise.separation`, `pairwise.pbc_correction`, `pairwise.second_law`, and the deprecated `pairwise.lennard_jones_energy` and `pairwise.lennard_jones_force` wrappers.
- `mc.select_random_particle`, `mc.get_new_particle`, `mc.reject`, `mc.metropolis`, and the identity `mc.accept(new_energy)`.
- `pylj.forcefields` and its `mixing` and `diameter` members; cross-species potentials are entries in `pair_potentials`.
- The `'random'` initial configuration, replaced by `'metropolis'`.
