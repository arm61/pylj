# Teaching with pylj

This page is for lecturers setting work with pylj. It says what pylj needs, what it can and cannot simulate, what each chapter assumes of the reader, which parameters are safe to vary, and how long the chapters take to run.

## What pylj needs

pylj needs Python 3.11 or later. It depends on numpy, scipy, matplotlib and Jupyter, all of which pip installs with it:

```bash
pip install git+https://github.com/arm61/pylj
```

The figures redraw in place as a simulation runs, so the work has to be done in a notebook. On a hosted notebook service such as Google Colab, a departmental JupyterHub or Binder, students install pylj from the first cell of the notebook:

```python
%pip install git+https://github.com/arm61/pylj
%matplotlib inline
```

The `%matplotlib inline` line selects the figure backend the viewers draw through. Every notebook needs it, whether pylj was installed beforehand or from the first cell.

## What it does and does not do

pylj simulates particles in two dimensions, in a square box with periodic boundaries. The box may be from 4 to 600 Angstrom on a side: below 4 Angstrom the box holds only one particle, and above 600 Angstrom the particles are too small to see.

The particles interact through pair potentials and nothing else. pylj packages the Lennard-Jones, Buckingham and square-well potentials, and a student can write their own in a short class, as the *Bring your own potential* chapter shows. A simulation may hold one species or a mixture of several.

Molecular dynamics integrates the motion with Velocity-Verlet, and the only thermostat is velocity rescaling: `heat_bath` sets the kinetic energy to the temperature asked for. Monte Carlo makes one kind of move, which relocates a single particle to a uniformly random position in the box. There are no bonds, no charges, no walls and no pressure control, so pylj cannot run molecules, ionic systems, surfaces or a constant-pressure ensemble.

The practical ceiling is about a hundred particles, which run at about a thousand steps a second on a laptop; twenty particles run at several thousand. Drawing is the slow part, not the physics, so a viewer should be updated every fifty to a hundred steps rather than every step.

## Which chapters need what

The chapters build on each other, and each is one notebook to run from top to bottom.

- **pylj**, the opening chapter, needs no Python beyond reading a `for` loop, and no physics beyond the idea that atoms attract and repel.
- **Particles and potentials** needs functions, array arithmetic in numpy, and the `**model` idiom for passing a dictionary of keyword arguments, which it explains and every later chapter uses. It assumes the reader has seen a potential energy curve, and introduces the Lennard-Jones form, the periodic box and the minimum image convention.
- **Molecular dynamics** needs functions, loops and f-strings. It assumes Newton's second law and the idea of kinetic energy, and builds the Velocity-Verlet integrator and a thermostat from them.
- **Monte Carlo** needs the same Python. It assumes the Boltzmann distribution, and builds the Metropolis rule from it.
- **The ideal gas law** needs loops, f-strings and a helper function with default arguments. Its physics is the Maxwell-Boltzmann distribution, the radial distribution function, the virial pressure, the second virial coefficient and the van der Waals equation. It uses `scipy.integrate.quad`, `scipy.optimize.brentq` and `scipy.optimize.curve_fit`, each explained where it appears.
- **The ideal gas law from first principles** needs only two short functions. It assumes the partition function of statistical mechanics and derives the law from it.
- **Bring your own potential** needs the reader to write a Python class with two methods. Its physics is a pair potential and its derivative.
- **Viewers and panes** also needs a class. It assumes no physics beyond the earlier chapters, and its subject is matplotlib rather than simulation.

## Parameters that are safe to vary, and the guards

The chapters run at temperatures from 100 to 1000 K, in boxes from 20 to 150 Angstrom, with 9 to 100 particles. Those ranges are safe to explore in any combination that keeps the particles from being packed shoulder to shoulder. The cut-off is 15 Angstrom by default, or half the box when the box is smaller than 30 Angstrom, and may be set to anything up to half the box.

Three guards refuse a run rather than let it produce nonsense. Each raises `ValueError` with a message that names the remedy.

The first is the initial energy limit. A starting configuration holding more than ten $k_B T$ of potential energy per particle is refused, because that energy is released as motion over the first few steps and heats the run. The message says how much energy per particle the configuration holds and that its particles are too close together for the temperature. A student meets it when they ask for too many particles in too small a box, or place particles at too low a temperature. The remedies are fewer particles, a larger box, or `init_conf='metropolis'`, which places particles one at a time by energy rather than on a lattice.

The second is the cut-off check. Truncating the interaction at the cut-off assumes it has died away there, so pylj checks that each pair energy at the cut-off is finite and within $k_B T$ of zero. The message says how many $k_B T$ the potential still is at the cut-off and asks the reader to check the parameter units. That is almost always the cause: a well depth in kilojoules per mole, or a separation in Angstrom where metres and joules are expected. When the cut-off is already half the box, the only remedy is a larger box.

The third is the displacement check, made on every molecular dynamics step. If any particle moves further than half the cut-off in one step, the run is refused. The message gives the distance moved and the timestep, and says the timestep is too long or the run has already diverged. The default timestep of ten femtoseconds is safe for argon at the temperatures the chapters use; a student who raises it, or who runs a much lighter species, will meet this guard.

## Compute times

All the chapters together execute in about a minute on a laptop. The chapters divide that time as follows.

| Chapter | Time to execute |
| --- | --- |
| The ideal gas law | about 25 s |
| Monte Carlo | about 20 s |
| Molecular dynamics | about 8 s |
| Each of the other five chapters | under 5 s |

The longest single cells are the comparison of the two methods in *Monte Carlo*, at about 10 s, the two radial distribution functions in *The ideal gas law*, at about 8 s each, and the pressure sweep in the same chapter, at about 6 s. A student running a chapter from top to bottom therefore waits well under a minute.

## Citing pylj

pylj is released under the MIT licence, so it may be used, modified and redistributed in teaching freely, as long as the copyright notice and licence text travel with it.

If you use pylj in teaching, please cite the paper describing it:

> McCluskey, A. R., Morgan, B. J., Edler, K. J., and Parker, S. C. (2018). "pylj: A teaching tool for classical atomistic simulation." *Journal of Open Source Education*, 1(2), 19. https://doi.org/10.21105/jose.00019

Calling `pylj.__cite__()` opens that paper in a browser. On a hosted notebook the browser is on the server rather than the student's machine, so the call does nothing visible.
