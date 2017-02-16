import unittest
import numpy as np
from source.fem import Fem


class Test_Fem(unittest.TestCase):
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
        fem = Fem(Test_Fem.vertices_matrix, Test_Fem.connectivity_matrix)
        fem.local_assembly([0,1,4], lambda x, y: 1, lambda x, y: 1, 5)
        local_stiffness =[
            [ 0.5, -0.5, 0.],
            [-0.5, 1., -0.5],
            [ 0., -0.5, 0.5]]

        local_load = [
            [0.04166667],
            [0.04166667],
            [0.04166667]]

        self.assertTrue((fem.local_stiffness == local_stiffness).all())
        for i in range(3):
            self.assertAlmostEqual(fem.local_load[i][0], local_load[i][0])

    def test_global_assembly(self):
        fem = Fem(Test_Fem.vertices_matrix, Test_Fem.connectivity_matrix)
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
        self.assertTrue((fem.global_stiffness == global_stiffness).all())
        self.assertTrue((fem.global_load == global_load).all())

    def test_get_mapping_matrix(self):
        result = np.array([[0.5, 0.5],[0., 0.5]])
        fem = Fem(Test_Fem.vertices_matrix, Test_Fem.connectivity_matrix)
        self.assertTrue((fem.get_mapping_matrix([0, 1, 4]) == result).all())

    def test_get_boundary_array(self):
        fem = Fem(Test_Fem.vertices_matrix, Test_Fem.connectivity_matrix)
        boundary_array = [0, 1, 2, 3, 5, 6, 7, 8]
        self.assertListEqual(boundary_array, fem.get_boundary_array())
