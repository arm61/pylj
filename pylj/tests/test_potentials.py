import numpy as np
import pytest

from pylj.potentials import PairPotential, Species, buckingham, lennard_jones, square_well


class TestSpecies:
    def test_carries_mass_and_name(self):
        argon = Species(mass=39.948, name="argon")
        assert argon.mass == 39.948
        assert argon.name == "argon"

    def test_name_defaults_to_empty(self):
        assert Species(mass=39.948).name == ""

    def test_is_frozen(self):
        argon = Species(mass=39.948, name="argon")
        with pytest.raises(Exception):  # noqa: B017
            argon.mass = 1.0

    def test_is_hashable_so_it_can_key_a_mapping(self):
        argon = Species(mass=39.948, name="argon")
        xenon = Species(mass=131.29, name="xenon")
        interactions = {(argon, argon): 1, (argon, xenon): 2}
        assert interactions[(argon, argon)] == 1


class TestPairPotential:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            PairPotential()

    def test_a_subclass_must_define_energies_and_forces(self):
        class Incomplete(PairPotential):
            pass

        with pytest.raises(TypeError):
            Incomplete()


class TestLennardJones:
    def test_constructor_is_keyword_only(self):
        with pytest.raises(TypeError):
            lennard_jones(1.65e-21, 3.4e-10)

    def test_energy_zero_at_sigma(self):
        lj = lennard_jones(epsilon=1.65e-21, sigma=3.4e-10)
        np.testing.assert_allclose(lj.energies(np.array([3.4e-10])), [0.0], atol=1e-30)

    def test_energy_minimum_is_minus_epsilon(self):
        lj = lennard_jones(epsilon=1.65e-21, sigma=3.4e-10)
        r_min = 2 ** (1 / 6) * 3.4e-10
        np.testing.assert_allclose(lj.energies(np.array([r_min])), [-1.65e-21], rtol=1e-6)

    def test_force_is_the_negative_energy_gradient(self):
        lj = lennard_jones(epsilon=1.65e-21, sigma=3.4e-10)
        r = np.array([3.0e-10, 4.0e-10, 5.0e-10])
        h = r * 1e-6
        numerical = -(lj.energies(r + h) - lj.energies(r - h)) / (2 * h)
        np.testing.assert_allclose(lj.forces(r), numerical, rtol=1e-4)

    def test_force_sign_is_repulsive_then_attractive(self):
        lj = lennard_jones(epsilon=1.65e-21, sigma=3.4e-10)
        r_min = 2 ** (1 / 6) * 3.4e-10
        assert lj.forces(np.array([0.9 * r_min]))[0] > 0
        assert lj.forces(np.array([1.5 * r_min]))[0] < 0

    def test_returns_an_array_for_an_array(self):
        lj = lennard_jones(epsilon=1.65e-21, sigma=3.4e-10)
        assert lj.energies(np.array([3e-10, 4e-10])).shape == (2,)
        assert lj.forces(np.array([3e-10, 4e-10])).shape == (2,)


class TestBuckingham:
    def test_constructor_is_keyword_only(self):
        with pytest.raises(TypeError):
            buckingham(1e-16, 3e10, 1e-77)

    def test_energy_matches_the_formula(self):
        bk = buckingham(a=1e-16, b=3e10, c=1e-77)
        r = np.array([3e-10, 4e-10])
        expected = 1e-16 * np.exp(-3e10 * r) - 1e-77 / r**6
        np.testing.assert_allclose(bk.energies(r), expected, rtol=1e-12)

    def test_force_is_the_negative_energy_gradient(self):
        bk = buckingham(a=1e-16, b=3e10, c=1e-77)
        r = np.array([3.0e-10, 4.0e-10, 5.0e-10])
        h = r * 1e-6
        numerical = -(bk.energies(r + h) - bk.energies(r - h)) / (2 * h)
        np.testing.assert_allclose(bk.forces(r), numerical, rtol=1e-4)


class TestSquareWell:
    def test_constructor_is_keyword_only(self):
        with pytest.raises(TypeError):
            square_well(1.65e-21, 3.4e-10, 1.5)

    def test_energy_is_a_step(self):
        sw = square_well(epsilon=1.65e-21, sigma=3.4e-10, lambda_=1.5, max_val=1e5)
        # inside the core, in the well, and beyond the well
        energies = sw.energies(np.array([3.0e-10, 4.0e-10, 6.0e-10]))
        np.testing.assert_allclose(energies, [1e5, -1.65e-21, 0.0])

    def test_force_raises(self):
        sw = square_well(epsilon=1.65e-21, sigma=3.4e-10, lambda_=1.5)
        with pytest.raises(ValueError, match="Monte Carlo"):
            sw.forces(np.array([4.0e-10]))
