import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pylj import forcefields


class TestForcefields(unittest.TestCase):
    def test_lennard_jones(self):
        a = forcefields.lennard_jones([1.0, 1.0])
        assert_almost_equal(a.energy(2.0), -0.015380859)
        assert_almost_equal(a.force(2.0), -0.045410156)
        a.mixing([2.0, 3.0])
        assert_almost_equal([a.a, a.b], [1.4239324, 1.7379922])
        b = forcefields.lennard_jones([1.0, 1.0])
        assert_almost_equal(b.energy([2.0, 4.5],), [-0.015380859, -0.00012041])
        assert_almost_equal(b.force([2.0, 4.0]), [-0.045410156, -3.66032124e-04])
        c = forcefields.lennard_jones([0.5, 3.5])
        assert_almost_equal(c.energy([2.0, 4.5],), [-0.05456543, -0.00042149])
        assert_almost_equal(c.force([2.0, 4.5],), [-0.1633301, -0.000562])
        d = forcefields.lennard_jones( [5.0, 3.5])
        assert_almost_equal(d.energy([1.0, 1.5, 20.0]),
                            [1.50000000, -0.268733500, -5.46874988e-08])
        assert_almost_equal(d.force([1.0, 1.5, 20.0]),
                            [3.9000000e+01, -9.2078707e-01, -1.6406249e-08])
        e = forcefields.lennard_jones([100.0, 300.0])
        assert_almost_equal(e.energy([100.0, 200.0, 500.0]), [0, 0, 0])
        assert_almost_equal(e.force([100.0, 200.0, 500.0]), [0, 0, 0])
        with self.assertRaises(IndexError):
            forcefields.lennard_jones([1.0, 1.0, 1.0])

    def test_lennard_jones_sigma_epsilon(self):
        a = forcefields.lennard_jones_sigma_epsilon([1.0, 0.25])
        assert_almost_equal(a.energy(2.0), -0.015380859)
        assert_almost_equal(a.force(2.0), -0.045410156)
        b = forcefields.lennard_jones_sigma_epsilon([1.0, 0.25])
        assert_almost_equal(b.energy([2.0, 1.0]), [-0.015380859, 0])
        assert_almost_equal(b.force([2.0, 4.0]), [-0.0454102, -0.000366])
        c = forcefields.lennard_jones_sigma_epsilon([0.5, 0.75])
        assert_almost_equal(c.energy([2.0, 1.0, 1.5]), [-0.0007322, -0.0461425, -0.0041095])
        assert_almost_equal(c.force([2.0, 1.0, 1.5]), [-0.0021962, -0.2724609, -0.0164157])
        d = forcefields.lennard_jones_sigma_epsilon([5e-10, 9e-9])
        assert_almost_equal(d.energy([400.0, 500.0, 600.0]), [0, 0, 0])
        assert_almost_equal(d.force([400.0, 500.0, 600.0]), [0, 0, 0])
        e = forcefields.lennard_jones_sigma_epsilon([1.0, 1.0])
        e.mixing([4.0, 4.0])
        assert_almost_equal([e.sigma, e.epsilon], [2.5, 2.0])
        with self.assertRaises(IndexError):
            forcefields.lennard_jones_sigma_epsilon([1.0, 1.0, 1.0])

    def test_buckingham(self):
        a = forcefields.buckingham([1.0, 1.0, 1.0])
        assert_almost_equal(a.energy(2.0), 0.1197103832)
        assert_almost_equal(a.force(2.0), 0.08846028324)
        a.mixing([4.0, 4.0, 4.0])
        assert_almost_equal([a.a, a.b, a.c], [2.0, 2.0, 2.0])
        b = forcefields.buckingham([1.0, 1.0, 1.0])
        assert_almost_equal(b.energy([2.0]), 0.1197103832)
        assert_almost_equal(b.force([2.0]), 0.08846028324)
        c = forcefields.buckingham([1.0, 1.0, 1.0])
        assert_almost_equal(c.energy([2.0, 4.0]), [0.1197103832, 0.0180715])
        assert_almost_equal(c.force([2.0, 4.0]), [0.0884603, 0.0179494])
        d = forcefields.buckingham([0.01, 0.01, 0.01])
        assert_almost_equal(d.energy([2.0, 4.0, 5.0]), [0.0096457, 0.0096055, 0.0095117])
        assert_almost_equal(d.force([2.0, 4.0, 5.0]),
                            [-3.7073013e-04,  9.2416835e-05,  9.4354942e-05])
        with self.assertRaises(IndexError):
            forcefields.buckingham([1.0, 1.0])

    def test_square_well(self):
        a = forcefields.square_well([1.0, 1.5, 2.0])
        assert_equal(a.energy(2.0), -1.0)
        with self.assertRaises(ValueError):
            a.force()
        b = forcefields.square_well([1.0, 2.0, 1.25])
        assert_equal(b.energy(0.5), float('inf'))
        c = forcefields.square_well([0.5, 1.5, 1.25])
        assert_equal(c.energy(3.0), 0)
        d = forcefields.square_well([1.0, 1.5, 2.0])
        assert_equal(d.energy([2.0, 0.5]), [-1.0, float('inf')])
        e = forcefields.square_well([1.0, 1.5, 1.25])
        assert_equal(e.energy([3.0, 3.0, 0.25]), [0, 0, float('inf')])
        f = forcefields.square_well([1.0, 1.5, 1.25], max_val=5000)
        assert_equal(f.energy([3.0, 3.0, 0.25]), [0, 0, 5000])
        h = forcefields.square_well([1.0, 1.5, 2.0])
        assert_equal(h.energy(1.2), float('inf'))
        i = forcefields.square_well([1.0, 1.5, 2.0])
        assert_equal(i.energy(1.6), -1.0)
        with self.assertRaises(IndexError):
            forcefields.square_well([1.0, 1.0])

    def test_lennard_jones_sigma_epsilon_diameter(self):
        a = forcefields.lennard_jones_sigma_epsilon([1.0, 0.25])
        assert_almost_equal(a.diameter, 2 ** (1 / 6))

    def test_lennard_jones_diameter(self):
        a = forcefields.lennard_jones([1.363e-134, 9.273e-78])
        assert_almost_equal(a.diameter * 1e10, 3.78, decimal=2)

    def test_buckingham_diameter_is_at_the_well(self):
        a = forcefields.buckingham([1.69e-15, 3.66e10, 1.01e-77])
        assert_almost_equal(a.diameter * 1e10, 3.83, decimal=1)

    def test_square_well_energy_takes_an_integer_array(self):
        a = forcefields.square_well([1.0, 1.5, 2.0])
        energy = a.energy(np.arange(1, 6))
        assert_equal(energy.dtype, np.dtype(float))
        assert_almost_equal(energy, [np.inf, -1.0, 0.0, 0.0, 0.0])

    def test_square_well_energy_takes_a_python_integer(self):
        a = forcefields.square_well([1.0, 1.5, 2.0])
        assert_equal(a.energy(2), forcefields.square_well([1.0, 1.5, 2.0]).energy(2.0))

    def test_square_well_diameter(self):
        a = forcefields.square_well([1.0, 1.5, 2.0])
        assert_equal(a.diameter, 1.5)

    def test_buckingham_diameter_rejects_constants_without_a_well(self):
        a = forcefields.buckingham([1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            _ = a.diameter

    def test_lennard_jones_energy_and_force_are_repeatable(self):
        a = forcefields.lennard_jones([1.0, 1.0])
        assert_almost_equal(a.energy(2.0), a.energy(2.0))
        assert_almost_equal(a.force(2.0), a.force(2.0))

    def test_lennard_jones_sigma_epsilon_energy_and_force_are_repeatable(self):
        a = forcefields.lennard_jones_sigma_epsilon([1.0, 0.25])
        assert_almost_equal(a.energy(2.0), a.energy(2.0))
        assert_almost_equal(a.force(2.0), a.force(2.0))

    def test_buckingham_energy_and_force_are_repeatable(self):
        a = forcefields.buckingham([1.0, 1.0, 1.0])
        assert_almost_equal(a.energy(2.0), a.energy(2.0))
        assert_almost_equal(a.force(2.0), a.force(2.0))

    def test_square_well_energy_is_repeatable(self):
        a = forcefields.square_well([1.0, 1.5, 2.0])
        assert_equal(a.energy(2.0), a.energy(2.0))

    def _assert_energy_accepts_float_numpy_scalar_list_and_array(self, a):
        # int, np.float32 and a 0-d array all raised on master (#83): a
        # Python int hit "Integers to negative integer powers are not
        # allowed" for the LJ variants, and np.float32/a 0-d array took
        # buckingham's array branch, where np.where on a 0-d array raises
        # under NumPy 2.
        e_int = a.energy(2)
        e_float = a.energy(2.0)
        e_np_scalar = a.energy(np.float64(2.0))
        e_np_float32 = a.energy(np.float32(2.0))
        e_0d_array = a.energy(np.array(2.0))
        e_list = a.energy([2.0])
        e_array = a.energy(np.array([2.0]))
        self.assertIs(type(e_int), float)
        self.assertIs(type(e_float), float)
        self.assertIs(type(e_np_scalar), float)
        self.assertIs(type(e_np_float32), float)
        self.assertIs(type(e_0d_array), float)
        self.assertEqual(np.shape(e_list), (1,))
        self.assertEqual(np.shape(e_array), (1,))
        assert_almost_equal(e_float, e_int)
        assert_almost_equal(e_float, e_np_scalar)
        assert_almost_equal(e_float, e_np_float32)
        assert_almost_equal(e_float, e_0d_array)
        assert_almost_equal(e_float, e_list[0])
        assert_almost_equal(e_float, e_array[0])

    def _assert_force_accepts_float_numpy_scalar_list_and_array(self, a):
        f_int = a.force(2)
        f_float = a.force(2.0)
        f_np_scalar = a.force(np.float64(2.0))
        f_np_float32 = a.force(np.float32(2.0))
        f_0d_array = a.force(np.array(2.0))
        f_list = a.force([2.0])
        f_array = a.force(np.array([2.0]))
        self.assertIs(type(f_int), float)
        self.assertIs(type(f_float), float)
        self.assertIs(type(f_np_scalar), float)
        self.assertIs(type(f_np_float32), float)
        self.assertIs(type(f_0d_array), float)
        self.assertEqual(np.shape(f_list), (1,))
        self.assertEqual(np.shape(f_array), (1,))
        assert_almost_equal(f_float, f_int)
        assert_almost_equal(f_float, f_np_scalar)
        assert_almost_equal(f_float, f_np_float32)
        assert_almost_equal(f_float, f_0d_array)
        assert_almost_equal(f_float, f_list[0])
        assert_almost_equal(f_float, f_array[0])

    def test_lennard_jones_accepts_float_numpy_scalar_list_and_array(self):
        a = forcefields.lennard_jones([1.0, 1.0])
        self._assert_energy_accepts_float_numpy_scalar_list_and_array(a)
        self._assert_force_accepts_float_numpy_scalar_list_and_array(a)

    def test_lennard_jones_sigma_epsilon_accepts_float_numpy_scalar_list_and_array(self):
        a = forcefields.lennard_jones_sigma_epsilon([1.0, 0.25])
        self._assert_energy_accepts_float_numpy_scalar_list_and_array(a)
        self._assert_force_accepts_float_numpy_scalar_list_and_array(a)

    def test_buckingham_accepts_float_numpy_scalar_list_and_array(self):
        a = forcefields.buckingham([1.0, 1.0, 1.0])
        self._assert_energy_accepts_float_numpy_scalar_list_and_array(a)
        self._assert_force_accepts_float_numpy_scalar_list_and_array(a)

    def test_square_well_energy_accepts_float_numpy_scalar_list_and_array(self):
        a = forcefields.square_well([1.0, 1.5, 2.0])
        self._assert_energy_accepts_float_numpy_scalar_list_and_array(a)


if __name__ == '__main__':
    unittest.main(exit=False)
