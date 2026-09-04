"""The species and pair potentials shared across the test suite.

``ARGON_MODEL`` and ``MIXTURE_MODEL`` unpack into the ``species`` and
``pair_potentials`` keywords of the initialisers and ``System``.
"""

from typing import TypedDict

from pylj.potentials import LennardJones, PairPotential, Species


class Model(TypedDict):
    species: list[Species]
    pair_potentials: dict[tuple[Species, Species], PairPotential]


ARGON = Species(mass=39.948, name="argon")
# A heavier particle with a 5 Angstrom core and the argon well depth.
LARGER = Species(mass=80.0, name="larger")

LJ_ARGON = LennardJones(epsilon=1.577e-21, sigma=3.372e-10)
LJ_LARGER = LennardJones(epsilon=1.577e-21, sigma=5.0e-10)
LJ_ARGON_LARGER = LennardJones(epsilon=1.577e-21, sigma=4.186e-10)

ARGON_MODEL: Model = {
    "species": [ARGON],
    "pair_potentials": {(ARGON, ARGON): LJ_ARGON},
}
MIXTURE_MODEL: Model = {
    "species": [ARGON, LARGER],
    "pair_potentials": {
        (ARGON, ARGON): LJ_ARGON,
        (LARGER, LARGER): LJ_LARGER,
        (ARGON, LARGER): LJ_ARGON_LARGER,
    },
}
