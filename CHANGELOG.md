# Changelog

All notable changes to pylj are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `pylj.sample` is a package: pane classes, one per plot, and a `Viewer` that lays out a list of panes. The seven viewer names are subclasses of it, and custom viewers combine panes or add a new one.
- `System.simulation`, `'md'` or `'mc'`, set by the initialisers.
- `step_sample` on `System`, recorded by `md.sample` and `mc.sample`, so viewers work at any sampling cadence.
- A `diameter` property on every forcefield (the separation at the pair-potential minimum) and a `diameter` argument on `md.initialise` and `mc.initialise` to override it, in Angstrom. Particles are drawn to scale.
- `System.cores`, the separation at which each forcefield's pair energy falls to zero, found numerically from its `energy` method. Initial configurations keep particles at least this far apart.
- `pylj.constants`, taking the Boltzmann constant and the atomic mass unit from `scipy.constants`.
- ruff and mypy configuration in `pyproject.toml`, a `dev` extra, and continuous integration on Python 3.11 to 3.14.

### Changed

- `md.heat_bath(particles, mass, bath_temperature)` rescales on the instantaneous temperature; it previously took the temperature sample array and rescaled towards its cumulative mean. `System.heat_bath(bath_temperature)` is unchanged. A non-positive bath temperature, or a system with zero or non-finite temperature, raises `ValueError` (#76).
- Python 3.11 or later is required. scipy is a dependency; Cython is not.
- `System.__init__` takes keyword-only arguments after `mass`, including the required `simulation`.
- The initialisers compute the initial forces, so the first integration step uses real accelerations.
- Viewers are built before their display is opened, and a viewer whose panes need molecular dynamics samples refuses a Monte Carlo system.
- The energy pane plots potential plus N k_B T; the radial distribution function is normalised by the ideal-gas shell count with r at bin centres; the speed histogram is drawn in its own bins; the pressure axis is labelled in N m^-1.
- `md.initialize` and `mc.initialize` pass every argument through.
- `JustCell` no longer takes a `scale` argument. `Viewer.average()` raises on a viewer whose panes keep no history. `CellPlus.update` rejects half-supplied custom data.
- The atomic mass unit used for initial velocities is the CODATA value; initial velocities and computed temperatures move by up to 4e-5 relative.
- `energy` and `force` on the forcefields return a Python float for scalar input, rather than `np.float64`; a one-element list, such as to `square_well.energy`, returns a one-element array rather than a float.
- `pairwise.compute_force` evaluates each forcefield only on the pairs of its own types, instead of passing every forcefield the full distance array with the other types' entries zeroed.
- Pair distances and accelerations are computed with vectorised NumPy. `pairwise.dist` returns `(dr, dx, dy)` and no longer takes or returns particle types; `pairwise.calculate_pressure` takes the pair distances and forces directly, `(distances, forces, box_length, number_of_particles, temperature)`, and `md.sample` feeds it the forces stored by the last force evaluation rather than recomputing them.

### Fixed

- `Energy` and `Phase` viewers crashed under NumPy 2.
- `Interactions`, `Phase` and `Scattering` crashed if built before the first sample.
- Two-type systems drew the other type's particles at the origin.
- Pair energies in multi-type systems were counted once per type pair (#81).
- `mc.metropolis` reused one random number for the life of the process (#78).
- The square-well hard core tested epsilon rather than sigma (part of #80), and `square_well.energy` failed on integer input.
- A custom forcefield without a `diameter` property, a diameter given in metres, or a non-positive or non-finite diameter (whether from the forcefield or passed by the caller) is refused with a clear message.
- Random initial configurations no longer overlap: particles are placed at least their repulsive-core separation apart (#82).
- The square lattice refuses a spacing below the repulsive core, with the largest count or the smallest box that fits, rounded up so the suggested box is accepted, in the message (#82).
- A forcefield whose `energy` does not return one value per separation, whose pair energy is positive at every grid point (suggesting its constants are in the wrong units) is refused with a clear message.
- `energy` and `force` on the forcefields no longer store their result on `self`, overwriting the bound method and breaking a second call on the same instance (#79).
- `buckingham.energy` and `buckingham.force` raised under NumPy 2.1 or later when the separation was an integer, a 0-d array, or a NumPy scalar that is not a float subclass (#83). `lennard_jones` and `lennard_jones_sigma_epsilon` also failed on integer input.

### Removed

- `pairwise.heat_bath`; `md.heat_bath` is the implementation (#76).
- `pylj/sample.py`, `point_size` on forcefields, `System.type_identifiers`, `pairwise.create_dist_identifiers`, `MANIFEST.in`, and the Code Climate upload from CI.
- `pairwise.separation`, `pairwise.pbc_correction`, `pairwise.second_law`, and the deprecated `pairwise.lennard_jones_energy` and `pairwise.lennard_jones_force` wrappers.
