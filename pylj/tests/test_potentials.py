from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pylj.potentials import Buckingham, LennardJones, PairPotential, Species, SquareWell


class TestSpecies:
    def test_carries_mass_and_name(self):
        argon = Species(mass=39.948, name="argon")
        assert argon.mass == 39.948
        assert argon.name == "argon"

    def test_name_defaults_to_empty(self):
        assert Species(mass=39.948).name == ""

    def test_is_frozen(self):
        argon = Species(mass=39.948, name="argon")
        with pytest.raises(FrozenInstanceError):
            argon.mass = 1.0

    def test_is_hashable_so_it_can_key_a_mapping(self):
        argon = Species(mass=39.948, name="argon")
        xenon = Species(mass=131.29, name="xenon")
        interactions = {(argon, argon): 1, (argon, xenon): 2}
        assert interactions[(argon, argon)] == 1

    def test_rejects_a_non_positive_or_non_finite_mass(self):
        for mass in (0.0, -1.0, np.inf, np.nan):
            with pytest.raises(ValueError, match="mass must be positive"):
                Species(mass=mass)


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
            LennardJones(1.65e-21, 3.4e-10)

    def test_rejects_a_non_positive_or_non_finite_parameter(self):
        for name in ("epsilon", "sigma"):
            for bad in (0.0, -1.0, np.inf, np.nan):
                parameters = {"epsilon": 1.65e-21, "sigma": 3.4e-10, name: bad}
                with pytest.raises(ValueError, match=name):
                    LennardJones(**parameters)

    def test_energy_zero_at_sigma(self):
        lj = LennardJones(epsilon=1.65e-21, sigma=3.4e-10)
        np.testing.assert_allclose(lj.energies(np.array([3.4e-10])), [0.0], atol=1e-30)

    def test_energy_minimum_is_minus_epsilon(self):
        lj = LennardJones(epsilon=1.65e-21, sigma=3.4e-10)
        r_min = 2 ** (1 / 6) * 3.4e-10
        np.testing.assert_allclose(lj.energies(np.array([r_min])), [-1.65e-21], rtol=1e-6)

    def test_force_is_the_negative_energy_gradient(self):
        lj = LennardJones(epsilon=1.65e-21, sigma=3.4e-10)
        r = np.array([3.0e-10, 4.0e-10, 5.0e-10])
        h = r * 1e-6
        numerical = -(lj.energies(r + h) - lj.energies(r - h)) / (2 * h)
        np.testing.assert_allclose(lj.forces(r), numerical, rtol=1e-4)

    def test_force_sign_is_repulsive_then_attractive(self):
        lj = LennardJones(epsilon=1.65e-21, sigma=3.4e-10)
        r_min = 2 ** (1 / 6) * 3.4e-10
        assert lj.forces(np.array([0.9 * r_min]))[0] > 0
        assert lj.forces(np.array([1.5 * r_min]))[0] < 0

    def test_returns_an_array_for_an_array(self):
        lj = LennardJones(epsilon=1.65e-21, sigma=3.4e-10)
        assert lj.energies(np.array([3e-10, 4e-10])).shape == (2,)
        assert lj.forces(np.array([3e-10, 4e-10])).shape == (2,)

    def test_is_infinite_at_zero_separation(self):
        # Coincident atoms cost infinite energy, so a Metropolis trial
        # there is rejected rather than compared as NaN.
        lj = LennardJones(epsilon=1.65e-21, sigma=3.4e-10)
        assert lj.energies(np.array([0.0]))[0] == np.inf
        assert lj.forces(np.array([0.0]))[0] == np.inf


class TestBuckingham:
    def test_constructor_is_keyword_only(self):
        with pytest.raises(TypeError):
            Buckingham(1e-16, 3e10, 1e-77)

    def test_rejects_a_non_positive_or_non_finite_parameter(self):
        good = {"a": 1e-16, "b": 3e10, "c": 1e-77}
        for name in ("a", "b"):
            for bad in (0.0, -1.0, np.inf, np.nan):
                with pytest.raises(ValueError, match=name):
                    Buckingham(**{**good, name: bad})
        for bad in (-1e-77, np.inf, np.nan):
            with pytest.raises(ValueError, match="c must"):
                Buckingham(**{**good, "c": bad})

    def test_rejects_a_repulsion_too_weak_to_form_a_barrier(self):
        # With A fourteen orders of magnitude too small the barrier would
        # sit at about 500 Angstrom, beyond any separation a simulation
        # reaches: every pair would be inside it.
        with pytest.raises(ValueError, match="too weak"):
            Buckingham(a=1e-30, b=1e5, c=1e-77)

    def test_energy_matches_the_formula(self):
        bk = Buckingham(a=1e-16, b=3e10, c=1e-77)
        r = np.array([3e-10, 4e-10])
        expected = 1e-16 * np.exp(-3e10 * r) - 1e-77 / r**6
        np.testing.assert_allclose(bk.energies(r), expected, rtol=1e-12)

    def test_force_is_the_negative_energy_gradient(self):
        bk = Buckingham(a=1e-16, b=3e10, c=1e-77)
        r = np.array([3.0e-10, 4.0e-10, 5.0e-10])
        h = r * 1e-6
        numerical = -(bk.energies(r + h) - bk.energies(r - h)) / (2 * h)
        np.testing.assert_allclose(bk.forces(r), numerical, rtol=1e-4)

    def test_min_separation_is_the_top_of_the_barrier(self):
        bk = Buckingham(a=1e-16, b=3e10, c=1e-77)
        r = bk.min_separation
        assert 0 < r < 3e-10
        # Zero to the root finder's tolerance, against a force scale of A B.
        np.testing.assert_allclose(bk.forces(np.array([r])), [0.0], atol=1e-6 * 1e-16 * 3e10)
        assert bk.energies(np.array([0.99 * r]))[0] < bk.energies(np.array([r]))[0]
        assert bk.energies(np.array([1.01 * r]))[0] < bk.energies(np.array([r]))[0]

    def test_is_the_formula_inside_the_barrier(self):
        # The formula itself collapses to minus infinity at short range; the
        # potential reports it as it is and leaves the simulation to keep
        # pairs outside min_separation.
        bk = Buckingham(a=1e-16, b=3e10, c=1e-77)
        inside = np.array([0.5 * bk.min_separation])
        expected = 1e-16 * np.exp(-3e10 * inside) - 1e-77 / inside**6
        np.testing.assert_allclose(bk.energies(inside), expected, rtol=1e-12)
        assert expected[0] < 0
        assert bk.energies(np.array([0.0]))[0] == -np.inf

    def test_a_form_without_a_barrier_is_valid_everywhere(self):
        bk = Buckingham(a=1e-16, b=3e10, c=0.0)
        assert bk.min_separation == 0.0
        assert np.isfinite(bk.energies(np.array([1e-12]))[0])

    def test_other_potentials_are_valid_everywhere(self):
        assert LennardJones(epsilon=1.65e-21, sigma=3.4e-10).min_separation == 0.0
        assert SquareWell(epsilon=1.65e-21, sigma=3.4e-10, lambda_=1.5).min_separation == 0.0


class TestSquareWell:
    def test_constructor_is_keyword_only(self):
        with pytest.raises(TypeError):
            SquareWell(1.65e-21, 3.4e-10, 1.5)

    def test_rejects_a_non_positive_or_non_finite_parameter(self):
        good = {"epsilon": 1.65e-21, "sigma": 3.4e-10, "lambda_": 1.5}
        for name in ("epsilon", "sigma"):
            for bad in (0.0, -1.0, np.inf, np.nan):
                with pytest.raises(ValueError, match=name):
                    SquareWell(**{**good, name: bad})
        for bad in (1.0, 0.5, np.inf, np.nan):
            with pytest.raises(ValueError, match="lambda_"):
                SquareWell(**{**good, "lambda_": bad})
        for bad in (0.0, -1e-21, np.nan):
            with pytest.raises(ValueError, match="max_val"):
                SquareWell(**{**good, "max_val": bad})
        assert SquareWell(**good, max_val=1e-20).max_val == 1e-20

    def test_energy_is_a_step(self):
        sw = SquareWell(epsilon=1.65e-21, sigma=3.4e-10, lambda_=1.5, max_val=1e5)
        # inside the core, in the well, and beyond the well
        energies = sw.energies(np.array([3.0e-10, 4.0e-10, 6.0e-10]))
        np.testing.assert_allclose(energies, [1e5, -1.65e-21, 0.0])

    def test_energies_keep_the_shape_of_the_separations(self):
        # A separation of any shape, including a scalar 0-d array, comes back
        # with the same shape, as for every other potential.
        sw = SquareWell(epsilon=1.65e-21, sigma=3.4e-10, lambda_=1.5)
        assert sw.energies(np.array(4.0e-10)).shape == ()
        assert sw.energies(np.array([3.0e-10, 4.0e-10])).shape == (2,)

    def test_force_raises(self):
        sw = SquareWell(epsilon=1.65e-21, sigma=3.4e-10, lambda_=1.5)
        with pytest.raises(ValueError, match="Monte Carlo"):
            sw.forces(np.array([4.0e-10]))
