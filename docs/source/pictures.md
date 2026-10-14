---
file_format: mystnb
kernelspec:
  name: python3
---

# Pictures and animations

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
%config InlineBackend.figure_format = "retina"
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

Once a run has recorded frames, `view()` and `animate()` show what the atoms were doing. `view()` draws a single configuration, with each atom as a disc drawn to scale. Because the box is periodic, an atom that overhangs an edge is drawn again at the opposite edge. Without arguments `view()` draws the current configuration, and `frame` picks a recorded frame instead:

The examples draw a short molecular dynamics run of argon, sampled every five steps:

```{code-cell} python
import matplotlib.pyplot as plt

from pylj.md import MDSimulation
from pylj.model import Model
from pylj.potentials import LennardJones, Species

argon = Species(mass=39.948, name="argon")
model = Model.single(argon, LennardJones(epsilon=0.9497, sigma=3.372))
simulation = MDSimulation.initialise(
    model, number_of_atoms=30, temperature=300, box=20, init_conf="triangular", seed=1
)
for _ in range(600):
    simulation.step()
    if simulation.steps % 50 == 0:
        simulation.rescale_velocities(300)
    if simulation.steps % 5 == 0:
        simulation.sample()
```

```{code-cell} python
simulation.view()          # the atoms now
simulation.view(frame=-1)  # the last recorded frame
plt.show()
```

To put several frames side by side, draw each one on its own axes with `ax`:

```{code-cell} python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, frame in zip(axes, [0, 60, 119]):
    simulation.view(frame=frame, ax=ax)
plt.show()
```

`animate()` plays the recorded frames in sequence. In a notebook it appears as a player, with a slider and with buttons to play, pause and step through the frames. `frames` selects which frames to play, as a slice or as a list of frame indices, and `fps` sets the speed, 30 frames per second by default:

```{code-cell} python
:tags: [remove-stderr]
simulation.animate(frames=slice(0, 60))
```

How smooth the motion looks depends on how far the atoms move between frames, and so on how often the run was sampled. Every frame is stored in the notebook, so long animations are large: 100 frames of 168 atoms take about 2.7 MB. Rather than build an animation larger than matplotlib's `animation.embed_limit` (20 MB by default), `animate()` raises `ValueError`. A progress bar shows while the frames are drawn, and `progress=False` turns it off.

Both methods size and colour the atoms in the same way. By default an atom is drawn with a diameter equal to half the separation at which two atoms of its species have their lowest pair energy. For Lennard-Jones atoms that diameter is 2^(1/6) sigma / 2. `diameter` and `colour` override the defaults, each taking either one value for every atom or a sequence with one value per atom. A sequence of colours can pick out one atom to follow through an animation:

```{code-cell} python
:tags: [remove-stderr]
colour = ["lightgrey"] * simulation.configuration.number_of_atoms
colour[0] = "red"
simulation.animate(frames=slice(0, 60), colour=colour)
```
