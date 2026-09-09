import unittest

import numpy as np
from numpy.testing import assert_allclose

from pylj.scattering import default_q_max, shell_average, wavevectors


def reference(position, wavevector):
    """|F(q)|^2 / N for each wavevector, one Python loop at a time."""
    n = len(position)
    out = np.empty(len(wavevector))
    for i, q in enumerate(wavevector):
        amplitude = sum(np.exp(1j * np.dot(q, r)) for r in position)
        out[i] = abs(amplitude) ** 2 / n
    return out


class TestWavevectors(unittest.TestCase):
    def test_magnitudes_are_the_commensurate_values_in_order(self):
        box = 20e-10
        q, _, _ = wavevectors(box, 5 * 2 * np.pi / box)
        expected = 2 * np.pi / box * np.sqrt([1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25])
        assert_allclose(q, expected)

    def test_lowest_shells_have_the_expected_multiplicities(self):
        box = 20e-10
        q, _, shell = wavevectors(box, 5 * 2 * np.pi / box)
        counts = np.bincount(shell)
        self.assertEqual(list(counts[:4]), [4, 4, 4, 8])

    def test_every_wavevector_matches_the_magnitude_of_its_shell(self):
        box = 20e-10
        q, wavevector, shell = wavevectors(box, 4 * 2 * np.pi / box)
        assert_allclose(np.linalg.norm(wavevector, axis=1), q[shell])

    def test_the_origin_is_absent(self):
        box = 20e-10
        _, wavevector, _ = wavevectors(box, 4 * 2 * np.pi / box)
        self.assertFalse((np.linalg.norm(wavevector, axis=1) == 0).any())

    def test_no_wavevector_exceeds_q_max(self):
        box = 20e-10
        q_max = 3.5 * 2 * np.pi / box
        _, wavevector, _ = wavevectors(box, q_max)
        self.assertLessEqual(np.linalg.norm(wavevector, axis=1).max(), q_max)


class TestShellAverage(unittest.TestCase):
    def test_matches_a_direct_sum_over_wavevectors(self):
        box = 20e-10
        rng = np.random.default_rng(0)
        position = rng.uniform(0, box, size=(12, 2))
        q, wavevector, shell = wavevectors(box, 3 * 2 * np.pi / box)
        direct = reference(position, wavevector)
        expected = np.bincount(shell, weights=direct) / np.bincount(shell)
        assert_allclose(shell_average(position, wavevector, shell), expected)

    def test_is_never_negative(self):
        box = 20e-10
        rng = np.random.default_rng(1)
        position = rng.uniform(0, box, size=(40, 2))
        q, wavevector, shell = wavevectors(box, 8 * 2 * np.pi / box)
        self.assertTrue((shell_average(position, wavevector, shell) >= 0).all())

    def test_one_atom_scatters_as_itself(self):
        box = 20e-10
        q, wavevector, shell = wavevectors(box, 3 * 2 * np.pi / box)
        assert_allclose(shell_average(np.array([[1e-10, 2e-10]]), wavevector, shell), 1.0)


class TestDefaultQMax(unittest.TestCase):
    def test_is_six_times_the_mean_spacing_wavevector(self):
        assert_allclose(default_q_max(100, 40e-10), 6 * 2 * np.pi * 10 / 40e-10)
