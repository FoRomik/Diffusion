import unittest
import numpy as np
from source.fem import Fem


class Test_fem(unittest.TestCase):
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

    def test_local_assembly(self):
        fem = Fem(Test_fem.vertices_matrix, Test_fem.connectivity_matrix)
        fem.local_assembly([0,1,4], lambda x, y: 1, lambda x, y: 1)
        local_stiffness =[
            [ 0.5, -0.5, 0.],
            [-0.5, 1., -0.5],
            [ 0., -0.5, 0.5]]
        local_load = [
            [0.04166667],
            [0.04166667],
            [0.04166667]]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    fem.local_stiffness[i][j],
                    local_stiffness[i][j])
        for j in range(3):
            self.assertAlmostEqual(fem.local_load[i][0], local_load[i][0])

    def test_global_assembly(self):
        fem = Fem(Test_fem.vertices_matrix, Test_fem.connectivity_matrix)
        fem.global_assembly([0,1,2], [[1,1,1],[1,1,1],[1,1,1]], [[1],[1],[1]])
        global_stiffness = [
            [ 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [ 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [ 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [ 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [ 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [ 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [ 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [ 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [ 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        global_load = [[ 1.],[ 1.],[ 1.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.]]
        for i in range(9):
            for j in range(9):
                self.assertEqual(
                    fem.global_stiffness[i][j],
                    global_stiffness[i][j])
        for j in range(9):
            self.assertEqual(fem.global_load[i][0], global_load[i][0])

    def test_get_mapping_matrix(self):
        result = np.array([[0.5, 0.5],[0., 0.5]])
        fem = Fem(Test_fem.vertices_matrix, Test_fem.connectivity_matrix)
        for i in range(2):
            for j in range(2):
                self.assertEqual(
                    fem.get_mapping_matrix([0, 1, 4])[i][j],
                        result[i][j])

    def test_get_boundary_array(self):
        fem = Fem(Test_fem.vertices_matrix, Test_fem.connectivity_matrix)
        boundary_array = [0, 1, 2, 3, 5, 6, 7, 8]
        self.assertListEqual(boundary_array, fem.get_boundary_array())
