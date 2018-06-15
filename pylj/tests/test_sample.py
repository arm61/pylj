from numpy.testing import assert_almost_equal, assert_equal
from pylj import sample, md
import unittest

class TestScattering(unittest.TestCase):
    def test_scattering(self):
        a = md.initialise(2, 300, 8, 'square')
        b = sample.Scattering(a)
        assert_equal(b.ax.shape, (2, 2))

class TestInteractions(unittest.TestCase):
    def test_interactions(self):
        a = md.initialise(2, 300, 8, 'square')
        b = sample.Interactions(a)
        assert_equal(b.ax.shape, (2, 2))

class TestEnergy(unittest.TestCase):
    def test_energy(self):
        a = md.initialise(2, 300, 8, 'square')
        b = sample.Energy(a)
        assert_equal(b.ax.shape, (2,))

class TestRDF(unittest.TestCase):
    def test_rdf(self):
        a = md.initialise(2, 300, 8, 'square')
        b = sample.RDF(a)
        assert_equal(b.ax.shape, (2,))

class TestSample(unittest.TestCase):
    def test_environment(self):
        a, b = sample.environment(2)
        assert_equal(b.shape, (2,))
        a, b = sample.environment(4)
        assert_equal(b.shape, (2, 2))


