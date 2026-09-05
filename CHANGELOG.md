# Changelog

All notable changes to pylj are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `pylj.sample` is a package: pane classes, one per plot, and a `Viewer` that lays out a list of panes. The seven viewer names are subclasses of it, and custom viewers combine panes or add a new one.
- `System.simulation`, `'md'` or `'mc'`, set by the initialisers.
- `step_sample` on `System`, recorded by `md.sample` and `mc.sample`, so viewers work at any sampling cadence.
- A `diameter` property on every forcefield (the separation at the pair-potential minimum) and a `diameter` argument on `md.initialise` and `mc.initialise` to override it, in Angstrom. Particles are drawn to scale.
- `pylj.constants`, taking the Boltzmann constant and the atomic mass unit from `scipy.constants`.
- ruff and mypy configuration in `pyproject.toml`, a `dev` extra, and continuous integration on Python 3.11 to 3.14.
- `System.restart()` returns a new system that continues from the current configuration with step and time at zero, empty sample arrays and the mean squared displacement measured from the copied positions, for starting a production run after equilibration.
- `seed` on `md.initialise`, `mc.initialise` and `System`, and `System.rng`, the `numpy.random.Generator` that places an initial configuration, draws the initial velocities and makes Monte Carlo moves. The same seed reproduces the same run.
- `pylj.potentials`: `Species`, a frozen dataclass of mass and name; the `PairPotential` interface, `energies(dr)` and `forces(dr)` on an array of separations, the force being the signed radial `-dE/dr`; and `LennardJones(*, epsilon, sigma)`, `Buckingham(*, a, b, c)` and `SquareWell(*, epsilon, sigma, lambda_, max_val)`, keyword-only with physical parameters (#57). `Species` rejects a non-positive or non-finite mass.
- `pairwise.compute_energy`, which evaluates the pair distances and energies without calling `forces`, so a potential with no finite force drives Monte Carlo; `System.compute_energy` uses it.
- `pairwise.pair_potential` and `pairwise.particle_masses`.
- `CellPane(diameter=...)` and a `diameter` keyword on every named viewer, in Angstrom, one value or one per species; by default particles are drawn at the separation of the minimum of their species' own pair energy.
- `mc.accept`, the Metropolis criterion on an energy change; `mc.Proposal`, a proposed configuration with its energy change; `pairwise.particle_energy`, one particle's interaction energy with a set of others; `System.propose`, `System.apply` and `System.energy`, the total pair energy of the current configuration.
- `util.check_initial_energy`, used by `md.initialise` and `mc.initialise`, refuses an initial configuration whose pair energy is not finite or stores more than `util.INITIAL_ENERGY_LIMIT` (ten) k_B T per particle, as an overlapping lattice does; the message gives the temperature it could heat to.
- `System` refuses a pair potential whose energy at the cut-off is not finite or is larger in magnitude than k_B T, since the cut-off assumes the interaction has died away there; parameters in the wrong units are one way to trip it. `mc.accept` refuses a NaN energy change.
- `placement_temperature` on `md.initialise`, `mc.initialise` and `System`: the temperature of the Metropolis acceptance used to place an initial configuration, by default the run temperature.

### Changed

- `md.heat_bath(particles, mass, bath_temperature)` rescales on the instantaneous temperature; it previously took the temperature sample array and rescaled towards its cumulative mean. `System.heat_bath(bath_temperature)` is unchanged. A non-positive bath temperature, or a system with zero or non-finite temperature, raises `ValueError` (#76).
- Python 3.11 or later is required. scipy is a dependency; Cython is not.
- `System.__init__` takes keyword-only arguments after `mass`, including the required `simulation`.
- The initialisers compute the initial forces, so the first integration step uses real accelerations.
- Viewers are built before their display is opened, and a viewer whose panes need molecular dynamics samples refuses a Monte Carlo system.
- The energy pane plots potential plus kinetic energy, (N - 1) k_B T; the radial distribution function is normalised by the ideal-gas shell count with r at bin centres; the speed histogram is drawn in its own bins; the pressure axis is labelled in N m^-1.
- `md.initialize` and `mc.initialize` pass every argument through.
- `JustCell` no longer takes a `scale` argument. `Viewer.average()` raises on a viewer whose panes keep no history. `CellPlus.update` rejects half-supplied custom data.
- The atomic mass unit used for initial velocities is the CODATA value; initial velocities and computed temperatures move by up to 4e-5 relative.
- `energy` and `force` on the forcefields return a Python float for scalar input, rather than `np.float64`; a one-element list, such as to `square_well.energy`, returns a one-element array rather than a float.
- `pairwise.compute_force` evaluates each forcefield only on the pairs of its own types, instead of passing every forcefield the full distance array with the other types' entries zeroed.
- Pair distances and accelerations are computed with vectorised NumPy. `pairwise.dist` returns `(dr, dx, dy)` and no longer takes or returns particle types; `pairwise.calculate_pressure` takes the pair distances and forces directly, `(distances, forces, box_length, number_of_particles, temperature)`, and `md.sample` feeds it the forces stored by the last force evaluation rather than recomputing them.
- The particle dtype carries `xunwrapped` and `yunwrapped`, positions without periodic wrapping, advanced by `md.velocity_verlet`; `xprevious_position`, `yprevious_position`, `xpbccount` and `ypbccount` are gone. `md.calculate_msd(particles, initial_particles)` reads the unwrapped positions and no longer takes `box_length`; `md.update_positions` takes and returns the unwrapped positions in place of the previous positions.
- `md.calculate_temperature` divides the kinetic energy by (N - 1) k_B, since the 2N - 2 velocity components left once the centre-of-mass motion is removed carry k_B T / 2 each; the energy pane's kinetic term follows. Initial velocities are drawn from a normal distribution. `md.initialise` requires at least two particles.
- `md.initialise`, `mc.initialise` and `System` take keyword-only `species`, a sequence of `Species`, and `pair_potentials`, a mapping from each pair of species (in either order, including each species with itself) to a `PairPotential`, in place of `mass`, `constants`, `forcefield` and `diameter`. Particles are assigned to the species in turn; `System.masses` holds each particle's mass and `System.species` and `System.pair_potentials` the model. A missing pair raises `ValueError`; a potential class in place of an instance raises `TypeError`. A non-positive or non-finite temperature, and a cross pair supplied in both orders, each raise `ValueError`.
- The particle dtype's `types` field is the integer index of each particle's species.
- `pairwise.compute_force(particles, box_length, cut_off, pair_potentials, species)` resolves each pair of species to its potential and divides each pair force by the receiving particle's own mass; `pairwise.update_accelerations` takes the mass of each particle. `md.velocity_verlet` takes `pair_potentials` and `species`. `md.calculate_temperature` and `md.heat_bath` accept one mass per particle.
- Initial velocities are drawn with each particle's own thermal width and the mass-weighted centre-of-mass velocity is removed.
- The Monte Carlo loop proposes a configuration and applies it on acceptance, leaving the configuration untouched otherwise; a trial move costs O(N). `System.mc_sample` recomputes the pair distances and energies and sets `energy` to the exact total before recording. `System.init_temp` is `System.temperature`.
- `init_conf` takes `'square'` or `'metropolis'`. `'metropolis'` seats particles by sequential Metropolis insertion, each trial position accepted on its interaction energy with the particles already placed, in place of the `'random'` rejection-sampled placement; it works for any potential, including a hard core. `'square'` places on the lattice without an overlap check.

### Fixed

- `Energy` and `Phase` viewers crashed under NumPy 2.
- `Interactions`, `Phase` and `Scattering` crashed if built before the first sample.
- Two-type systems drew the other type's particles at the origin.
- Pair energies in multi-type systems were counted once per type pair (#81).
- The Metropolis criterion, `mc.accept`, draws a fresh random number for every uphill change (a downhill change is accepted without a draw); one was reused for the life of the process (#78).
- The square-well hard core tested epsilon rather than sigma (part of #80), and `square_well.energy` failed on integer input.
- A custom forcefield without a `diameter` property, a diameter given in metres, or a non-positive or non-finite diameter (whether from the forcefield or passed by the caller) is refused with a clear message.
- `'metropolis'` initial configurations do not overlap (#82).
- `energy` and `force` on the forcefields no longer store their result on `self`, overwriting the bound method and breaking a second call on the same instance (#79).
- `buckingham.energy` and `buckingham.force` raised under NumPy 2.1 or later when the separation was an integer, a 0-d array, or a NumPy scalar that is not a float subclass (#83). `lennard_jones` and `lennard_jones_sigma_epsilon` also failed on integer input.
- The mean squared displacement was wrong unless sampled on every integration step, and non-zero before the first step (#74).
- Initial velocities carried a centre-of-mass drift that never decayed and added a ballistic term to the mean squared displacement, and were not at the requested temperature (#75).
- The square-well potential drives a Monte Carlo simulation: its energies are evaluated without asking for a force, and `md.initialise` refuses it with a clear message (#80).

### Removed

- `pairwise.heat_bath`; `md.heat_bath` is the implementation (#76).
- `pylj/sample.py`, `point_size` on forcefields, `System.type_identifiers`, `pairwise.create_dist_identifiers`, `MANIFEST.in`, and the Code Climate upload from CI.
- `pairwise.separation`, `pairwise.pbc_correction`, `pairwise.second_law`, and the deprecated `pairwise.lennard_jones_energy` and `pairwise.lennard_jones_force` wrappers.
- `System.select_random_particle`, `new_random_position`, `metropolis`, `accept` and `reject`, with `old_energy`, `new_energy`, `position_store` and `random_particle`; `mc.select_random_particle`, `mc.get_new_particle`, `mc.reject`, `mc.metropolis`, and the identity `mc.accept(new_energy)`; the unused `energy` field of the particle dtype.
- `pylj.forcefields` and its `mixing` and `diameter` members; cross-species potentials are entries in `pair_potentials`. `md.compute_force` (a wrapper of `pairwise.compute_force`), `util.Particle`, `System.setup_types`, `System.setup_diameters` and `System.diameters`.
- The `'random'` initial configuration, replaced by `'metropolis'`.
