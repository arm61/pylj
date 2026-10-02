import numpy as np

from pylj.constants import BOLTZMANN, KJ_PER_MOL


class TestConstants:
    def test_boltzmann_is_the_gas_constant_in_kilojoules(self):
        np.testing.assert_allclose(BOLTZMANN, 8.314462618e-3, rtol=1e-9)

    def test_a_kilojoule_per_mole_is_a_hundred_amu_angstrom_squared_per_ps_squared(self):
        np.testing.assert_allclose(KJ_PER_MOL, 100.0, rtol=1e-8)
