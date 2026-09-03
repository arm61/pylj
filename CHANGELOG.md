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

### Fixed

- `Energy` and `Phase` viewers crashed under NumPy 2.
- `Interactions`, `Phase` and `Scattering` crashed if built before the first sample.
- Two-type systems drew the other type's particles at the origin.
- Pair energies in multi-type systems were counted once per type pair (#81).
- `mc.metropolis` reused one random number for the life of the process (#78).
- The square-well hard core tested epsilon rather than sigma (part of #80), and `square_well.energy` failed on integer input.
- A custom forcefield without a `diameter` property, a diameter given in metres, or a non-positive diameter is refused with a clear message.
- `System.random` placed particles uniformly with no separation check, so a first step could give overlapping particles and unphysical accelerations; it now places each particle by rejection sampling, keeping every pair at least the mean of their diameters apart, and raises `ValueError` if a placement cannot be found after 1000 attempts (#82).
- `System.square` had the same defect at higher density: it spaced particles by `box_length / ceil(sqrt(n))` with no check, silently overlapping them above a threshold particle count. It now raises `ValueError` if that spacing is less than the largest drawn diameter (#82).

### Removed

- `pairwise.heat_bath`; `md.heat_bath` is the implementation (#76).
- `pylj/sample.py`, `point_size` on forcefields, `System.type_identifiers`, `pairwise.create_dist_identifiers`, `MANIFEST.in`, and the Code Climate upload from CI.
