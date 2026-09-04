import numpy as np
import pytest

from pylj.potentials import Species


class TestSpecies:
    def test_carries_mass_and_name(self):
        argon = Species(mass=39.948, name="argon")
        assert argon.mass == 39.948
        assert argon.name == "argon"

    def test_name_defaults_to_empty(self):
        assert Species(mass=39.948).name == ""

    def test_is_frozen(self):
        argon = Species(mass=39.948, name="argon")
        with pytest.raises(Exception):
            argon.mass = 1.0

    def test_is_hashable_so_it_can_key_a_mapping(self):
        argon = Species(mass=39.948, name="argon")
        xenon = Species(mass=131.29, name="xenon")
        interactions = {(argon, argon): 1, (argon, xenon): 2}
        assert interactions[(argon, argon)] == 1
