from numpy.testing import assert_almost_equal
from pylj import comp
import unittest
import numpy as np


class TestPairwise(unittest.TestCase):
    def test_dist(self):
        xpos = np.array([0, 1])
        ypos = np.array([0, 1])
        box_length = 5.
        dr, dx, dy = comp.dist(xpos, ypos, box_length)
        assert_almost_equal(dr, [np.sqrt(2)])
        assert_almost_equal(dx, [-1])
        assert_almost_equal(dy, [-1])
