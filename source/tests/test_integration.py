import unittest
from source.integration import GaussianIntegration

class TestGaussianIntegration(unittest.TestCase):

    def test_integrate(self):
        integral = GaussianIntegration(5).integrate(lambda x, y: 1)
        self.assertAlmostEqual(integral, 0.5)
        integral = GaussianIntegration(3).integrate(lambda x, y: x+y)
        self.assertAlmostEqual(integral, 1/3)
