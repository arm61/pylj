# Simulations

## Pictures and animations

Once a run has recorded frames, `view()` and `animate()` show what the atoms were doing. `view()` draws a single configuration, with each atom as a disc drawn to scale. Because the box is periodic, an atom that overhangs an edge is drawn again at the opposite edge. Without arguments `view()` draws the current configuration, and `frame` picks a recorded frame instead:

```python
simulation.view()          # the atoms now
simulation.view(frame=-1)  # the last recorded frame
```

To put several frames side by side, draw each one on its own axes with `ax`:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, frame in zip(axes, [0, 100, 199]):
    simulation.view(frame=frame, ax=ax)
plt.show()
```

`animate()` plays the recorded frames in sequence. In a notebook it appears as a player, with a slider and with buttons to play, pause and step through the frames. `frames` selects which frames to play, as a slice or as a list of frame indices, and `fps` sets the speed, 30 frames per second by default:

```python
simulation.animate(frames=slice(0, 100))
```

How smooth the motion looks depends on how far the atoms move between frames, and so on how often the run was sampled. Every frame is stored in the notebook, so long animations are large: 100 frames of 168 atoms take about 2.7 MB. Rather than build an animation larger than matplotlib's `animation.embed_limit` (20 MB by default), `animate()` raises `ValueError`. A progress bar shows while the frames are drawn, and `progress=False` turns it off.

Both methods size and colour the atoms in the same way. By default an atom is drawn with a diameter equal to half the separation at which two atoms of its species have their lowest pair energy. For Lennard-Jones atoms that diameter is 2^(1/6) sigma / 2. `diameter` and `colour` override the defaults, each taking either one value for every atom or a sequence with one value per atom. A sequence of colours can pick out one atom to follow through an animation:

```python
colour = ["lightgrey"] * simulation.configuration.number_of_atoms
colour[0] = "red"
simulation.animate(colour=colour)
```

## Units

pylj takes and reports every quantity in Angstrom, picoseconds, kJ/mol, atomic mass units and kelvin, so velocities are in Angstrom per picosecond, forces in kJ/mol per Angstrom, the two-dimensional pressure in kJ/mol per Angstrom squared, and a diffusion coefficient from the mean squared displacement in Angstrom squared per picosecond. A mass times a velocity squared is in amu Angstrom^2/ps^2; `pylj.constants.KJ_PER_MOL`, about 100, is one kJ/mol in those units.
