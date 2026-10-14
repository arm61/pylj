---
file_format: mystnb
kernelspec:
  name: python3
---

# Units

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
%config InlineBackend.figure_format = "retina"
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

pylj takes and reports every quantity in Angstrom, picoseconds, kJ/mol, atomic mass units and kelvin, so velocities are in Angstrom per picosecond, forces in kJ/mol per Angstrom, the two-dimensional pressure in kJ/mol per Angstrom squared, and a diffusion coefficient from the mean squared displacement in Angstrom squared per picosecond. A mass times a velocity squared is in amu Angstrom^2/ps^2; `pylj.constants.KJ_PER_MOL`, about 100, is one kJ/mol in those units.

`pylj.constants` holds the two constants pylj converts with:

```{code-cell} python
from pylj.constants import BOLTZMANN, KJ_PER_MOL

print(f"BOLTZMANN = {BOLTZMANN} kJ/mol/K")
print(f"KJ_PER_MOL = {KJ_PER_MOL} amu Angstrom^2/ps^2")
```
