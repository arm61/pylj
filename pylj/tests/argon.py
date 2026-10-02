"""The species and pair potentials shared across the test suite.

``ARGON_MODEL`` and ``MIXTURE_MODEL`` are ``Model`` instances for the
``initialise`` factories.
"""

from pylj.model import Model
from pylj.potentials import Buckingham, LennardJones, Species, SquareWell

ARGON = Species(mass=39.948, name="argon")
# A heavier atom with a 5 Angstrom core and the argon well depth.
LARGER = Species(mass=80.0, name="larger")

LJ_ARGON = LennardJones(epsilon=0.9497, sigma=3.372)
LJ_LARGER = LennardJones(epsilon=0.9497, sigma=5.0)
LJ_ARGON_LARGER = LennardJones(epsilon=0.9497, sigma=4.186)

ARGON_MODEL = Model.single(ARGON, LJ_ARGON)
# A Buckingham form for argon, with its short-range barrier, and so its
# min_separation, at 0.78 Angstrom.
BUCKINGHAM_ARGON = Buckingham(a=1.0177e6, b=3.66, c=6142.6)
BUCKINGHAM_MODEL = Model.single(ARGON, BUCKINGHAM_ARGON)
# A hard-core square well with a 3 Angstrom core and a well out to 4.5.
WELL = SquareWell(epsilon=0.9033, sigma=3.0, lambda_=1.5)
WELL_MODEL = Model.single(ARGON, WELL)

# Hard cores of three different diameters, one per species pair.
WELL_MIXTURE_MODEL = Model(
    (ARGON, LARGER),
    {
        (ARGON, ARGON): WELL,
        (LARGER, LARGER): SquareWell(epsilon=0.9033, sigma=5.0, lambda_=1.5),
        (ARGON, LARGER): SquareWell(epsilon=0.9033, sigma=4.0, lambda_=1.5),
    },
)

MIXTURE_MODEL = Model(
    (ARGON, LARGER),
    {(ARGON, ARGON): LJ_ARGON, (LARGER, LARGER): LJ_LARGER, (ARGON, LARGER): LJ_ARGON_LARGER},
)
