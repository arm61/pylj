from numpy.testing import assert_almost_equal, assert_equal
from pylj import forcefields
import unittest


class TestForcefields(unittest.TestCase):
    def test_lennard_jones_energy(self):
        a = forcefields.lennard_jones(2.0, [1.0, 1.0])
        assert_almost_equal(a, -0.015380859)

    def test_lennard_jones_force(self):
        a = forcefields.lennard_jones(2.0, [1.0, 1.0], force=True)
        assert_almost_equal(a, -0.045410156)

    def test_lennard_jones_sigma_epsilon_energy(self):
        a = forcefields.lennard_jones_sigma_epsilon(2.0, [1.0, 0.25])
        assert_almost_equal(a, -0.015380859)

    def test_lennard_jones_sigma_epsilon_force(self):
        a = forcefields.lennard_jones_sigma_epsilon(
            2.0, [1.0, 0.25], force=True)
        assert_almost_equal(a, -0.045410156)

    def test_buckingham_energy(self):
        a = forcefields.buckingham(2.0, [1.0, 1.0, 1.0])
        assert_almost_equal(a, 0.1197103832)

    def test_buckingham_force(self):
        a = forcefields.buckingham(2.0, [1.0, 1.0, 1.0], force=True)
        assert_almost_equal(a, 0.08846028324)

    def test_square_well_energy(self):
        a = forcefields.square_well(2.0, [1.0, 1.5, 2.0])
        assert_equal(a, -1.0)
        b = forcefields.square_well(0.5, [1.0, 2.0, 1.25])
        assert_equal(b, float('inf'))
        c = forcefields.square_well(3.0, [0.5, 1.5, 1.25])
        assert_equal(c, 0)
        d = forcefields.square_well([2.0, 0.5], [1.0, 1.5, 2.0])
        assert_equal(d, [-1.0, float('inf')])
        e = forcefields.square_well([3.0, 3.0, 0.25], [1.0, 1.5, 1.25])
        assert_equal(e, [0, 0, float('inf')])
        f = forcefields.square_well(
            [3.0, 3.0, 0.25], [1.0, 1.5, 1.25], max_val=5000)
        assert_equal(f, [0, 0, 5000])

    def test_square_well_force(self):
        with self.assertRaises(ValueError):
            forcefields.square_well(
                2.0, [1.0, 1.5, 2.0], force=True)
        with self.assertRaises(ValueError):
            forcefields.square_well(
                [2.0], [1.0, 1.5, 2.0], force=True)


if __name__ == '__main__':
    unittest.main(exit=False)
