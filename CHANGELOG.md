# Changelog

All notable changes to pylj are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `simulation.Samples`, the record a simulation's `sample()` appends to, with one array per quantity and `step`, whose `add()` takes exactly one value per array so the arrays stay aligned; `md.MDSamples` adds `temperature`, `pressure`, `potential_energy`, `kinetic_energy`, `msd` and the derived `total_energy`; `mc.MCSamples` adds `potential_energy`. A simulation holds it as `samples`.
- `pylj.configuration`: `Configuration`, an immutable snapshot of `positions` (shape `(N, 2)`, Angstrom), `species`, `species_index` and `box`, with `number_of_atoms`, `masses` in atomic mass units, and `pairs`, `potential_energy`, `forces`, `virial` and `insertion_energy` taking the model and the cut-off explicitly, and `replace` and `without` for copies; `MDConfiguration`, adding `velocities` and `unwrapped` with `kinetic_energy`, `temperature` and `msd`; and `PairData`, the per-pair `distances`, `separations`, `energies` and `radial_forces` arrays with the `virial` derived from them. `pylj.placement`: `place_square`, `place_metropolis`, and `place`, which builds the initial configuration for `initialise`. `pylj.simulation`: `Simulation`, the base class with `configuration`, `model`, `cut_off`, `rng`, `steps`, `samples`, `trajectory` and `restart()`. Assigning `configuration` recomputes the forces of a molecular dynamics simulation or the energy of a Monte Carlo one, and a configuration with a different number of atoms or a different box raises `ValueError`.
- `md.MDSimulation` and `mc.MCSimulation`. `initialise(model, *, number_of_atoms, temperature, box, init_conf='square', placement_temperature=None, max_strain=0.05, cut_off=None, seed=None)` builds one from a model (`MDSimulation.initialise` also takes `timestep=0.01`, and `MCSimulation.initialise` takes `max_displacement=0.5`); the constructors, `MDSimulation(configuration, model, *, cut_off=None, timestep=0.01, seed=None)` and `MCSimulation(configuration, model, temperature, *, cut_off=None, max_displacement=0.5, seed=None)`, take a ready configuration. `step()` advances the simulation and `steps`; `sample()` records the samples; `restart()` continues from the current configuration. `MDSimulation` has `forces`, the net force on each atom, `timestep`, `time` (`steps * timestep`), `initial_configuration`, which the mean squared displacement is measured from, `integrate()`, which applies Velocity-Verlet and which a subclass overrides for a different integrator, and `heat_bath()`. `MCSimulation` has `temperature`, `max_displacement`, `energy`, `accepted`, `propose()` and `apply()`.
- `cut_off` on the `initialise` methods and the constructors, by default 15 Angstrom or half the box, whichever is smaller; a cut-off larger than half the box raises `ValueError`.
- `pylj.sample` is a package: pane classes, one per plot, and a `Viewer` that lays out a list of panes. The eight viewer names are subclasses of it, and custom viewers combine panes or add a new one.
- `samples.step` on every simulation, recorded by `sample()`, so viewers work at any sampling cadence.
- `pylj.constants`: `BOLTZMANN` in kJ/mol/K, and `KJ_PER_MOL`, one kJ/mol in amu Angstrom^2/ps^2, both derived from `scipy.constants`.
- ruff and mypy configuration in `pyproject.toml`, and `dev` and `docs` extras.
- `seed` on `MDSimulation.initialise`, `MCSimulation.initialise` and the simulation constructors, and `Simulation.rng`, the `numpy.random.Generator` that places an initial configuration, draws the initial velocities and makes Monte Carlo moves. The same seed reproduces the same run.
- `pylj.potentials`: `Species`, a frozen dataclass of mass and name; the `PairPotential` interface, `energies(dr)` and `forces(dr)` on an array of separations, the force being the signed radial `-dE/dr`; and `LennardJones(*, epsilon, sigma)`, `Buckingham(*, a, b, c)` and `SquareWell(*, epsilon, sigma, lambda_, max_val=np.inf)`, keyword-only with physical parameters (#57). `Species` rejects a non-positive or non-finite mass, `potentials.check_positive_finite` and `potentials.check_non_negative_finite` are the shared checks, and each potential rejects a parameter that is not positive and finite (`Buckingham` allows `c` of zero, and `SquareWell` needs `lambda_` greater than one and a positive `max_val`). `LennardJones` is infinite at zero separation. Every potential has `min_separation`, the separation below which it is unphysical, zero unless the potential sets it; a configuration gives a pair closer than that infinite energy, and raises if asked for its force. `Buckingham` sets it to the top of its short-range barrier, below which the formula falls to minus infinity, and its `energies` and `forces` are the formula at every separation; a `Buckingham` whose energy is still rising at 100 Angstrom, so that no barrier holds its atoms apart, raises `ValueError`.
- `pairwise.species_pairs` and `pairwise.minimum_image`.
- `CellPane(diameter=...)` and a `diameter` keyword on every named viewer, in Angstrom, one value or one per species; by default atoms are drawn at the separation of the minimum of their species' own pair energy. A diameter that is not positive and finite, or a potential with no energy minimum to size the atoms by, is refused with a clear message.
- `mc.accept(energy_change, temperature, *, rng=None)`, the Metropolis criterion on an energy change, which raises `ValueError` if the change is not a number, and `mc.Proposal`, a proposed move: the proposed `positions`, shape `(N, 2)`, the `energy_change`, and the `source` configuration it was proposed from. `MCSimulation.apply` refuses a proposal whose `source` is no longer the current configuration.
- `placement_temperature` on `MDSimulation.initialise` and `MCSimulation.initialise`: the temperature of the Metropolis acceptance used to place an initial configuration, by default the run temperature.
- `md.at_rest`, which returns a copy of a configuration with its centre of mass at rest; `MDSimulation` applies it to whatever it is built from, so `simulation.configuration` is a copy rather than the object passed in.
- `pylj.model.Model`, the species and the potential between each pair of them, validated when it is built and read-only after; `Model.single(species, potential)` builds the one-species case, and `Model.potential(one, other)` looks a pair up in either order. A model, and any simulation holding one, can be pickled and deep-copied, so a temperature scan can be handed to `multiprocessing` and a finished run saved to disk.
- `LennardJones`, `Buckingham` and `SquareWell` print as the constructor call that built them, so a model shows its parameters in a notebook.
- `placement.place_triangular` and `init_conf='triangular'`, which start a simulation from the triangular lattice a two-dimensional solid settles into, with `max_strain` on `place` and on both `initialise` methods setting how far the fitted lattice may sit from `placement.TRIANGULAR_RATIO`, as a fraction of that ratio, up to `placement.MOST_STRAIN`.
- `pylj.scattering`, the wavevectors commensurate with a box and the structure factor of a configuration at them.
- `pylj.trajectory.Trajectory`, the configurations a simulation has sampled, held as `simulation.trajectory`; `sample()` appends the current configuration and `restart()` starts an empty one. Indexing gives a frame, slicing gives a trajectory, and `positions` gives the `(frames, N, 2)` array. A molecular dynamics frame carries the time it was sampled at, in `times`, and `msd(max_lag=None)` averages the mean squared displacement over every time origin.
- `Configuration.rdf(bins=100, r_max=None)` and `Configuration.structure_factor(q_max=None)`, and the same two methods on `Trajectory` averaged over its frames, so g(r) and S(q) are available as arrays without building a viewer. S(q) is evaluated at the wavevectors commensurate with the box, and `q_max` defaults to six times `2 pi sqrt(N) / L`. A `q_max` below the smallest wavevector the box has, or one needing more than `scattering.MOST_WAVEVECTORS`, raises `ValueError`.

### Changed

- pylj takes and reports every quantity in Angstrom, picoseconds, kJ/mol, atomic mass units and kelvin, and computes in those units. 1.5.2 took the box in Angstrom, the mass in atomic mass units, and the forcefield constants and the timestep in SI, and reported SI.
- The package is built from `pyproject.toml`; `setup.py`, `docker/` and `TODO.md` are removed. Releases publish to PyPI through trusted publishing from a GitHub release. The version lives in `pylj.__version__` only, in place of the `util.__version__()` function, and `util.__cite__()` is `pylj.__cite__()`; the runtime dependency on `jupyter` is now `ipython`, since only the viewers' redrawing needs a notebook.
- The documentation is a set of pages on running a simulation, custom potentials and viewers, with the module reference, built with Sphinx and myst-nb (#85, #58).
- A box below `placement.SMALLEST_BOX`, 4 Angstrom, raises `ValueError` rather than `AttributeError`, and a box above 600 Angstrom is accepted.
- `md.heat_bath(configuration, bath_temperature)` returns the configuration with its velocities rescaled so that the instantaneous temperature is the bath temperature; it previously took the temperature sample array and rescaled towards its cumulative mean. A negative or non-finite bath temperature, a configuration with a non-finite temperature, or a configuration at rest with a bath temperature above zero, raises `ValueError` (#76).
- Python 3.11 or later is required. scipy is a dependency; Cython is not.
- The initialisers compute the initial forces, so the first integration step uses real accelerations.
- Viewers are built before their display is opened, and a viewer whose panes need molecular dynamics samples refuses a Monte Carlo simulation.
- `sample.environment(panes, size)` returns the figure and its axes, `(fig, axes)`, and leaves displaying the figure to the viewer; it returned `(fig, ax, hfig)` and displayed the figure itself.
- Viewers and panes take a simulation and read its `configuration` and `samples`, and a viewer given anything else raises `TypeError`; the radial distribution pane computes the pair distances when it draws. The scattering pane plots the structure factor S(q) against the wavevectors commensurate with the box; 1.5.2 plotted an intensity from the Debye sum `sin(qr) / (qr)` over pair distances, labelled `I(q)`. The energy pane plots the total energy, potential plus kinetic, for a molecular dynamics simulation; the `Interactions` viewer shows it in place of the force pane.
- The radial distribution function is normalised by the ideal-gas shell count with r at bin centres; the speed histogram is drawn in its own bins.
- `JustCell` no longer takes a `scale` argument.
- Initial velocities and computed temperatures use CODATA constants; they move by up to 4e-5 relative.
- Pair distances and forces are computed with vectorised NumPy. `pairwise.dist(positions, box)` takes `(N, 2)` positions and returns the distances and the `(M, 2)` separations; `pairwise.calculate_pressure(virial, box, kinetic_energy)` is the instantaneous virial pressure, `(2 K + sum(f r)) / (2 L^2)`.
- `md.velocity_verlet(configuration, forces, timestep, model, cut_off)` returns the next configuration and the forces at it, and raises `ValueError` if an atom moves further than half the cut-off in one step, which means the timestep is too long or the run has diverged; `md.update_positions(configuration, accelerations, timestep)` and `md.update_velocities(velocities, accelerations, next_accelerations, timestep)` work on `(N, 2)` arrays.
- `MDConfiguration.unwrapped` holds the positions without periodic wrapping, advanced by the integrator, and `msd` reads them (#74).
- `MDConfiguration.temperature` divides the kinetic energy by (N - 1) k_B, since the 2N - 2 velocity components left once the centre-of-mass motion is removed carry k_B T / 2 each. Initial velocities are drawn from a normal distribution. `MDSimulation.initialise` requires at least two atoms.
- `MDSimulation.initialise` and `MCSimulation.initialise` replace `md.initialise` and `mc.initialise`. A `Model` is the one positional argument, in place of `mass`, `constants` and `forcefield`; `number_of_atoms`, `box` and `timestep` replace `number_of_particles`, `box_length` and `timestep_length`; and every other argument is a keyword, so a call names the number of atoms, the temperature and the box. Atoms are assigned to the model's species in turn; `Configuration.masses` holds each atom's mass, and `Configuration.species` with `Configuration.species_index` names each atom's species. A negative or non-finite temperature raises `ValueError`.
- Initial velocities are drawn with each atom's own thermal width and the mass-weighted centre-of-mass velocity is removed.
- A Monte Carlo move displaces one atom, chosen at random, by a random distance of up to `max_displacement` along each axis; it previously moved the atom to a random position anywhere in the box, which at liquid and solid densities is almost always rejected.
- The Monte Carlo loop proposes a configuration and applies it on acceptance, leaving the configuration untouched otherwise; a trial move costs O(N). `MCSimulation.sample` recomputes the energy exactly before recording.
- `init_conf` is a keyword argument, `'square'` by default, taking `'square'`, `'triangular'` or `'metropolis'`; an unknown value raises `ValueError`. `'metropolis'` seats atoms by sequential Metropolis insertion, each trial position accepted on its interaction energy with the atoms already placed; it works for any potential, including a hard core.
- The `'square'` start spreads its columns and rows across the box: `ceil(sqrt(N))` columns and as many rows as the atoms need, in place of a square grid of `ceil(sqrt(N))` sites a side filled a column at a time, which left the spare sites as an empty strip. Alternate rows run in opposite directions, so two species form a chessboard rather than stripes.
- The radial distribution function pane shows its y axis, so the level g(r) = 1 can be read.
- The scattering pane shows its y axis, so the level S(q) = 1 can be read.
- `ScatteringPane` and the `Scattering` viewer take `q_max`, the largest wavevector magnitude drawn, so two runs can be drawn over the same axis.
- The cell pane draws an atom that overhangs an edge of the box again at the opposite edge, where the periodic boundary puts the overhanging part; atoms were previously clipped at the edge.
- Pane axis labels use the same font size as the tick labels, and a series held constant by the thermostat is shown with a margin of one per cent of its value.
- The `Phase`, `RDF` and `Scattering` viewers no longer keep a history of what they have drawn. `Viewer.average(simulation)`, on every viewer, and `Pane.average(ax, simulation)` draw the mean over the simulation's trajectory.

### Fixed

- `Energy` and `Phase` viewers crashed under NumPy 2.
- `Interactions`, `Phase` and `Scattering` crashed if built before the first sample.
- Two-type systems drew the other type's atoms at the origin.
- `CellPlus` could not be built: its constructor called `update` without the custom data that `update` required.
- Pair energies in multi-type systems were counted once per type pair (#81).
- The Metropolis criterion reused one random number for the life of the process (#78).
- The square-well hard core tested epsilon rather than sigma (part of #80).
- The `'random'` initial configuration could place atoms on top of one another (#82); `'metropolis'` replaces it.
- A pair potential's energy and force stored their result on the potential, overwriting the method and breaking a second call on the same instance (#79).
- The Buckingham potential raised under NumPy 2.1 or later for an integer, 0-d array or non-float NumPy scalar separation, and the Lennard-Jones and square-well forms failed on integer input (#83).
- The mean squared displacement was wrong unless sampled on every integration step, and non-zero before the first step (#74).
- Initial velocities carried a centre-of-mass drift that never decayed and added a ballistic term to the mean squared displacement, and were not at the requested temperature (#75).
- The square-well potential drives a Monte Carlo simulation: its energies are evaluated without asking for a force, and `MDSimulation.initialise` refuses it with a clear message (#80).

### Removed

- The example notebooks under `examples/`, the root `requirements.txt` and `docs/update_docs.sh`; the documentation is built from the `docs` extra.
- `pylj.util` and `System`; the structured atom array and `particle_dt`, whose `types` field `Configuration.species_index` replaces.
- `md.initialise`, `mc.initialise`, `md.initialize`, `mc.initialize`, `md.sample` and `mc.sample`; `md.calculate_temperature` and `md.calculate_msd`, which are methods on the configuration now.
- `md.compute_force`, `pairwise.compute_force`, `pairwise.update_accelerations` and `pairwise.heat_bath`.
- The `temperature_sample`, `pressure_sample`, `force_sample`, `msd_sample` and `energy_sample` arrays, which `samples` replaces, and the sum of the pair forces, which has no physical meaning.
- The `setup_*` and `update_*` drawing functions in `pylj.sample`, which the panes replace.
- `point_size` on forcefields, `pairwise.create_dist_identifiers`, `MANIFEST.in`, and the Code Climate upload from CI.
- `pairwise.separation`, `pairwise.pbc_correction`, `pairwise.second_law`, and the deprecated `pairwise.lennard_jones_energy` and `pairwise.lennard_jones_force` wrappers.
- `mc.select_random_particle`, `mc.get_new_particle`, `mc.reject`, `mc.metropolis`, and the identity `mc.accept(new_energy)`.
- `pylj.forcefields` and its `mixing` methods; cross-species potentials are entries in a `Model`.
- The `'random'` initial configuration, replaced by `'metropolis'`.
