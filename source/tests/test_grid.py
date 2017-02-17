import unittest
import numpy as np
from source.grid import Grid


class TestGrid(unittest.TestCase):
    static_file = "source/tests/fixtures/example.vtk"

    def test_get_vertices_matrix(self):
        grid = Grid(TestGrid.static_file)
        self.assertTrue((grid.get_vertices_matrix() == \
            [[0, 0], [1, 0], [1, 1], [0, 1]]).all())

    def test_get_connectivity_matrix(self):
        grid = Grid(TestGrid.static_file)
        self.assertTrue((grid.get_connectivity_matrix() == \
            [[0, 1, 2], [3, 0, 2]]).all())
