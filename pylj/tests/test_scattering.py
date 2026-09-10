import unittest

import numpy as np
from numpy.testing import assert_allclose

from pylj.scattering import check_q_max, default_q_max, shell_average, wavevectors


def reference(position, box, index):
    """|F(q)|^2 / N for each wavevector, one Python loop at a time."""
    n = len(position)
    unit = 2 * np.pi / box
    out = np.empty(len(index))
    for i, pair in enumerate(index):
        q = unit * pair
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
        q, index, shell = wavevectors(box, 4 * 2 * np.pi / box)
        unit = 2 * np.pi / box
        assert_allclose(unit * np.linalg.norm(index, axis=1), q[shell])

    def test_the_origin_is_absent(self):
        box = 20e-10
        _, index, _ = wavevectors(box, 4 * 2 * np.pi / box)
        self.assertFalse((np.linalg.norm(index, axis=1) == 0).any())

    def test_no_wavevector_exceeds_q_max(self):
        box = 20e-10
        q_max = 3.5 * 2 * np.pi / box
        _, index, _ = wavevectors(box, q_max)
        unit = 2 * np.pi / box
        self.assertLessEqual(unit * np.linalg.norm(index, axis=1).max(), q_max)


class TestShellAverage(unittest.TestCase):
    def test_matches_a_direct_sum_over_wavevectors(self):
        box = 20e-10
        rng = np.random.default_rng(0)
        position = rng.uniform(0, box, size=(12, 2))
        q, index, shell = wavevectors(box, 3 * 2 * np.pi / box)
        direct = reference(position, box, index)
        expected = np.bincount(shell, weights=direct) / np.bincount(shell)
        assert_allclose(shell_average(position, box, index, shell), expected)

    def test_is_never_negative(self):
        box = 20e-10
        rng = np.random.default_rng(1)
        position = rng.uniform(0, box, size=(40, 2))
        q, index, shell = wavevectors(box, 8 * 2 * np.pi / box)
        self.assertTrue((shell_average(position, box, index, shell) >= 0).all())

    def test_one_atom_scatters_as_itself(self):
        box = 20e-10
        q, index, shell = wavevectors(box, 3 * 2 * np.pi / box)
        assert_allclose(shell_average(np.array([[1e-10, 2e-10]]), box, index, shell), 1.0)


class TestDefaultQMax(unittest.TestCase):
    def test_is_six_times_the_mean_spacing_wavevector(self):
        assert_allclose(default_q_max(100, 40e-10), 6 * 2 * np.pi * 10 / 40e-10)


class TestWavevectorRange(unittest.TestCase):
    def test_keeps_every_wavevector_up_to_q_max(self):
        box = 20e-10
        unit = 2 * np.pi / box
        # A q_max between two shells: sqrt(13) is inside it, sqrt(16) is not.
        q, index, _ = wavevectors(box, 3.8 * unit)
        assert_allclose(q[-1], unit * np.sqrt(13))
        self.assertEqual(len(index), 4 + 4 + 4 + 8 + 4 + 4 + 8 + 8)

    def test_a_non_integer_default_keeps_its_outermost_shell(self):
        box = 20e-10
        unit = 2 * np.pi / box
        q_max = default_q_max(17, box)
        q, _, _ = wavevectors(box, q_max)
        self.assertGreater(q.max(), 0.999 * q_max - unit)
        self.assertLessEqual(q.max(), q_max)


class TestCheckQMax(unittest.TestCase):
    def test_accepts_the_smallest_wavevector_of_the_box(self):
        box = 20e-10
        check_q_max(2 * np.pi / box, box)

    def test_names_the_units_for_a_q_max_below_the_box(self):
        box = 20e-10
        with self.assertRaisesRegex(ValueError, "below.*smallest wavevector"):
            check_q_max(8.0, box)

    def test_an_exact_multiple_keeps_its_outermost_shell(self):
        box = 20e-10
        unit = 2 * np.pi / box
        for multiple in (3, 5, 7, 10):
            with self.subTest(multiple=multiple):
                q, _, _ = wavevectors(box, multiple * unit)
                assert_allclose(q.max(), multiple * unit)
