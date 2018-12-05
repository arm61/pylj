from numpy.testing import assert_almost_equal
from pylj import forcefields
import unittest


class TestForcefields(unittest.TestCase):
    def test_lennard_jones_energy(self):
        a = forcefields.lennard_jones(2., [1., 1.])
        assert_almost_equal(a, -0.015380859)

    def test_lennard_jones_force(self):
        a = forcefields.lennard_jones(2., [1., 1.], force=True)
        assert_almost_equal(a, -0.045410156)

    def test_buckingham_energy(self):
        a = forcefields.buckingham(2., [1., 1., 1.])
        assert_almost_equal(a, 0.1197103832)

    def test_buckingham_force(self):
        a = forcefields.buckingham(2., [1., 1., 1.], force=True)
        assert_almost_equal(a, 0.08846028324)
