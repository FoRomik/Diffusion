import unittest
import numpy as np
from source.fem import Fem


class TestFem(unittest.TestCase):
    vertices_matrix = np.array(
        [[0., 0.],
        [0.5, 0.],
        [1., 0.],
        [0., 0.5],
        [0.5, 0.5],
        [1., 0.5],
        [0., 1.],
        [0.5, 1.],
        [1., 1.]])

    connectivity_matrix = np.array(
        [[0, 1, 4],
        [3, 4, 7],
        [1, 2, 5],
        [4, 5, 8],
        [0, 4, 3],
        [3, 7, 6],
        [1, 5, 4],
        [4, 8, 7]])

    def test_assembly(self):
        fem = Fem(TestFem.vertices_matrix, TestFem.connectivity_matrix)
        fem.assembly([3, 4, 7], lambda x, y: 1, lambda x, y:1, 5)
        global_stiffness = [
            [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ,  0.5, -0.5,  0. ,  0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. , -0.5,  1. ,  0. ,  0. , -0.5,  0. ],
            [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ,  0. , -0.5,  0. ,  0. ,  0.5,  0. ],
            [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]]
        global_load = [
            [ 0.        ],
            [ 0.        ],
            [ 0.        ],
            [ 0.04166667],
            [ 0.04166667],
            [ 0.        ],
            [ 0.        ],
            [ 0.04166667],
            [ 0.        ]]

        for i in range(len(global_stiffness)):
            for j in range(len(global_stiffness[0])):
                self.assertAlmostEqual(
                    fem.global_stiffness[i][j], global_stiffness[i][j])
        for i in range(len(global_load)):
            self.assertAlmostEqual(fem.global_load[i][0], global_load[i][0])

    def test_get_mapping_matrix(self):
        result = np.array([[0.5, 0.5],[0., 0.5]])
        fem = Fem(TestFem.vertices_matrix, TestFem.connectivity_matrix)
        self.assertTrue((fem.get_mapping_matrix([0, 1, 4]) == result).all())

    def test_get_boundary_array(self):
        fem = Fem(TestFem.vertices_matrix, TestFem.connectivity_matrix)
        boundary_array = [0, 1, 2, 3, 5, 6, 7, 8]
        self.assertListEqual(boundary_array, fem.get_boundary_array())
