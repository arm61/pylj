"""The species and pair potentials shared across the test suite.

``ARGON_MODEL`` and ``MIXTURE_MODEL`` are ``Model`` instances for the
``initialise`` factories.
"""

from pylj.model import Model
from pylj.potentials import Buckingham, LennardJones, Species, SquareWell

ARGON = Species(mass=39.948, name="argon")
# A heavier atom with a 5 Angstrom core and the argon well depth.
LARGER = Species(mass=80.0, name="larger")

LJ_ARGON = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
LJ_LARGER = LennardJones(epsilon=1.577e-21, sigma=5.0e-10)
LJ_ARGON_LARGER = LennardJones(epsilon=1.577e-21, sigma=4.186e-10)

ARGON_MODEL = Model.single(ARGON, LJ_ARGON)
# A Buckingham form for argon, with its short-range barrier, and so its
# min_separation, at 0.78 Angstrom.
BUCKINGHAM_ARGON = Buckingham(a=1.69e-15, b=3.66e10, c=1.02e-77)
BUCKINGHAM_MODEL = Model.single(ARGON, BUCKINGHAM_ARGON)
# A hard-core square well with a 3 Angstrom core and a well out to 4.5.
WELL = SquareWell(epsilon=1.5e-21, sigma=3e-10, lambda_=1.5)
WELL_MODEL = Model.single(ARGON, WELL)

# Hard cores of three different diameters, one per species pair.
WELL_MIXTURE_MODEL = Model(
    (ARGON, LARGER),
    {
        (ARGON, ARGON): WELL,
        (LARGER, LARGER): SquareWell(epsilon=1.5e-21, sigma=5e-10, lambda_=1.5),
        (ARGON, LARGER): SquareWell(epsilon=1.5e-21, sigma=4e-10, lambda_=1.5),
    },
)

MIXTURE_MODEL = Model(
    (ARGON, LARGER),
    {(ARGON, ARGON): LJ_ARGON, (LARGER, LARGER): LJ_LARGER, (ARGON, LARGER): LJ_ARGON_LARGER},
)
