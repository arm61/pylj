from numpy.testing import assert_almost_equal, assert_equal
from pylj import forcefields
import unittest


class TestForcefields(unittest.TestCase):
    def test_lennard_jones_energy(self):
        a = forcefields.lennard_jones(2.0, [1.0, 1.0])
        assert_almost_equal(a.energy, -0.015380859)
        b = forcefields.lennard_jones([2.0, 4.5], [1.0, 1.0])
        assert_almost_equal(b.energy, [-0.015380859, -0.00012041])
        c = forcefields.lennard_jones([2.0, 4.5], [0.5, 3.5])
        assert_almost_equal(c.energy, [-0.05456543, -0.00042149])
        d = forcefields.lennard_jones([1.0, 1.5, 20.0], [5.0, 3.5])
        assert_almost_equal(d.energy, [1.50000000, -0.268733500, -5.46874988e-08])
        e = forcefields.lennard_jones([100.0, 200.0, 500.0], [100.0, 300.0])
        assert_almost_equal(e.energy, [0, 0, 0])

    def test_lennard_jones_force(self):
        a = forcefields.lennard_jones(2.0, [1.0, 1.0])
        assert_almost_equal(a.force, -0.045410156)
        b = forcefields.lennard_jones([2.0, 4.0, 6.0], [1.0, 1.0])
        assert_almost_equal(b.force, [-0.045410156, -3.66032124e-04, -2.14325517e-05])
        c = forcefields.lennard_jones([2.0, 4.0, 6.0], [1.5, 4.0])
        assert_almost_equal(c.force, [-0.185302734, -1.46457553e-03, -8.57325038e-05])
        d = forcefields.lennard_jones([150.0, 300.0, 500.0], [200.0, 500.0])
        assert_almost_equal(d.force, [-1.7558299e-12, -1.3717421e-14, -3.8400000e-16])

    def test_lennard_jones_sigma_epsilon_energy(self):
        a = forcefields.lennard_jones_sigma_epsilon(2.0, [1.0, 0.25])
        assert_almost_equal(a.energy, -0.015380859)
        b = forcefields.lennard_jones_sigma_epsilon([2.0, 1.0], [1.0, 0.25])
        assert_almost_equal(b.energy, [-0.015380859, 0])
        c = forcefields.lennard_jones_sigma_epsilon(
            [2.0, 1.0, 1.5], [0.5, 0.75])
        assert_almost_equal(c.energy, [-0.0007322, -0.0461425, -0.0041095])
        d = forcefields.lennard_jones_sigma_epsilon(
            [400.0, 500.0, 600.0], [5e-10, 9e-9])
        assert_almost_equal(d.energy, [0, 0, 0])

    def test_lennard_jones_sigma_epsilon_force(self):
        a = forcefields.lennard_jones_sigma_epsilon(2.0, [1.0, 0.25])
        assert_almost_equal(a.force, -0.045410156)
        b = forcefields.lennard_jones_sigma_epsilon([2.0, 4.0], [1.0, 0.25])
        assert_almost_equal(b.force, [-0.0454102, -0.000366])
        c = forcefields.lennard_jones_sigma_epsilon([3.0, 4.0], [3.0, 1.0])
        assert_almost_equal(c.force, [8.0, -0.6877549])

    def test_buckingham_energy(self):
        a = forcefields.buckingham(2.0, [1.0, 1.0, 1.0])
        assert_almost_equal(a.energy, 0.1197103832)
        b = forcefields.buckingham([2.0], [1.0, 1.0, 1.0])
        assert_almost_equal(b.energy, 0.1197103832)
        c = forcefields.buckingham([2.0, 4.0], [1.0, 1.0, 1.0])
        assert_almost_equal(c.energy, [0.1197103832, 0.0180715])
        d = forcefields.buckingham([2.0, 4.0, 5.0], [0.01, 0.01, 0.01])
        assert_almost_equal(d.energy, [0.0096457, 0.0096055, 0.0095117])

    def test_buckingham_force(self):
        a = forcefields.buckingham(2.0, [1.0, 1.0, 1.0])
        assert_almost_equal(a.force, 0.08846028324)
        b = forcefields.buckingham([2.0], [1.0, 1.0, 1.0])
        assert_almost_equal(b.force, 0.08846028324)
        c = forcefields.buckingham([2.0, 1.0, 4.0], [1.5, 0.1, 2.0])
        assert_almost_equal(c.force, [0.0290596, -11.8642744, 0.0998156])

    def test_square_well_energy(self):
        a = forcefields.square_well(2.0, [1.0, 1.5, 2.0])
        assert_equal(a.energy, -1.0)
        b = forcefields.square_well(0.5, [1.0, 2.0, 1.25])
        assert_equal(b.energy, float('inf'))
        c = forcefields.square_well(3.0, [0.5, 1.5, 1.25])
        assert_equal(c.energy, 0)
        d = forcefields.square_well([2.0, 0.5], [1.0, 1.5, 2.0])
        assert_equal(d.energy, [-1.0, float('inf')])
        e = forcefields.square_well([3.0, 3.0, 0.25], [1.0, 1.5, 1.25])
        assert_equal(e.energy, [0, 0, float('inf')])
        f = forcefields.square_well([3.0, 3.0, 0.25], [1.0, 1.5, 1.25], max_val=5000)
        assert_equal(f.energy, [0, 0, 5000])

    def test_square_well_force(self):
        a = forcefields.square_well(2.0, [1.0, 1.5, 2.0])
        with self.assertRaises(ValueError):
            a.update_force()
        b = forcefields.square_well([2.0], [1.0, 1.5, 2.0])
        with self.assertRaises(ValueError):
            b.update_force()


if __name__ == '__main__':
    unittest.main(exit=False)
