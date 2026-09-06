---
file_format: mystnb
kernelspec:
  name: python3
---

# The ideal gas law from first principles

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
```

The previous chapter measured the pressure of a simulated gas and compared it with $pA = N k_B T$. This chapter derives that law from the partition function of $N$ particles that do not interact, first in three dimensions and then in two.

## The partition function

For $N$ identical particles of mass $m$ in a volume $V$ at temperature $T$, with no interactions, the partition function is

$$
Q = \frac{V^N}{N!\,\Lambda^{3N}},
$$

where $\Lambda$ is the thermal de Broglie wavelength,

$$
\Lambda = \left(\frac{h^2}{2 \pi m k_B T}\right)^{1/2}.
$$

$\Lambda$, with $h$ Planck's constant, is the length scale below which quantum mechanics matters. For argon at room temperature it is far smaller than the spacing between atoms, which is why a classical simulation is justified:

```{code-cell} python
import numpy as np
from scipy.constants import h
from pylj.constants import ATOMIC_MASS_UNIT, BOLTZMANN

mass = 39.948 * ATOMIC_MASS_UNIT


def thermal_wavelength(temperature, mass):
    return np.sqrt(h**2 / (2 * np.pi * mass * BOLTZMANN * temperature))


print(f"argon at 293 K: {thermal_wavelength(293, mass) * 1e10:.3f} Angstrom")
print(f"an electron at 293 K: {thermal_wavelength(293, 9.1093837e-31) * 1e10:.1f} Angstrom")
```

## From the partition function to the pressure

Thermodynamic quantities come from the logarithm of $Q$:

$$
\ln Q = N \ln V - \ln N! - 3N \ln \Lambda.
$$

The $N!$ counts the ways of labelling identical particles, which do not give distinct states. It is the first term that matters here, because it is the only one that depends on the volume. The pressure is

$$
p = k_B T \left(\frac{\partial \ln Q}{\partial V}\right)_T = k_B T \frac{N}{V},
$$

which is the ideal gas law, $pV = N k_B T$. The terms in $N!$ and $\Lambda$ do not depend on $V$ and drop out of the derivative.

## Two dimensions

In two dimensions a particle has two translational degrees of freedom instead of three. The volume becomes an area $A$, and $\Lambda^3$ becomes $\Lambda^2$, because the power of $\Lambda$ counts the momentum integrals, one for each velocity component:

$$
Q = \frac{A^N}{N!\,\Lambda^{2N}}, \qquad
\ln Q = N \ln A - \ln N! - 2N \ln \Lambda.
$$

The pressure in two dimensions is a force per unit length, and it is the derivative with respect to area:

$$
p = k_B T \left(\frac{\partial \ln Q}{\partial A}\right)_T = \frac{N k_B T}{A}.
$$

This is the ideal gas law the previous chapter tested. The simulation measures the pressure from the kinetic energy, and each of the $2N$ velocity components carries $k_B T / 2$, so the kinetic term of its pressure is $K / A = N k_B T / A$, the same law by a different route. Holding the centre of mass at rest fixes two of those components, which is why the simulation measured $(N - 1) k_B T / A$. Written as a function, and evaluated for the argon at STP the previous chapter ran:

```{code-cell} python
def ideal_pressure(number, temperature, area):
    return number * BOLTZMANN * temperature / area


area = (150e-10) ** 2
print(f"20 particles: {ideal_pressure(20, 273.15, area):.3e} N/m")
print(f"19 particles: {ideal_pressure(19, 273.15, area):.3e} N/m")
```

These are the two ideal lines the previous chapter printed, and its measured pressure sat on the second. The derivation assumed the particles do not interact and have no size. Both assumptions fail for the Lennard-Jones argon of the previous chapter as the density rises, which is where its measured pressures left the law.
