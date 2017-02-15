"""
This module...
"""

import numpy as np
#import scipy.sparse as sparse
from source import triangle


class Fem(object):
    """
    This class...
    """
    def __init__(self, vertices_matrix, connectivity_matrix):
        self.vertices_matrix = vertices_matrix
        self.connectivity_matrix = connectivity_matrix
        self.boundary_array = self.get_boundary_array()

        self.local_stiffness = []
        self.local_load = []
        vertices_number = len(self.vertices_matrix)
        self.global_stiffness = np.zeros((vertices_number, vertices_number))  # (make sparse!)
        self.global_load = np.zeros((vertices_number, 1))

    def solve(self, sigma, function, integration_order):
        """
        This is the main method of the Fem class which solves the diffusion
        equation.
        """
        for element in self.connectivity_matrix:
            self.local_assembly(element, sigma, function)
            self.global_assembly(
                element, self.local_stiffness, self.local_load)
        self.apply_boundary()
        return np.linalg.solve(
            self.global_stiffness, self.global_load).flatten()

    def local_assembly(self, element, sigma, function):
        """
        This method...
        """
        mapping_matrix = self.get_mapping_matrix(element)
        inverse_transpose = np.linalg.inv(mapping_matrix).T
        determinant = abs(np.linalg.det(mapping_matrix))
        self.local_stiffness = triangle.get_local_stiffness(
            inverse_transpose, determinant, sigma)
        self.local_load = triangle.get_local_load(
            inverse_transpose, determinant, function)

    def global_assembly(self, element, stiffness, load):
        """
        This method...
        """
        self.global_stiffness[
            np.transpose(np.array([element])), element] += stiffness
        self.global_load[element] += load


    def get_mapping_matrix(self, element):
        """
        This method...
        """
        [[x_0, y_0], [x_1, y_1], [x_2, y_2]] = \
            self.vertices_matrix[element, :]
        return np.array([[x_1 - x_0, x_2 - x_0], [y_1 - y_0, y_2 - y_0]])

    def get_boundary_array(self):
        """
        Performing an algorithm on connectivity
        matrix.
        """
        edges = set()
        boundaries = set()
        for triangle in self.connectivity_matrix:
            triangle = sorted(triangle)
            for i in [(0, 1), (0, 2), (1, 2)]:
                if (triangle[i[0]], triangle[i[1]]) not in edges:
                    edges.add((triangle[i[0]], triangle[i[1]]))
                else: edges.remove((triangle[i[0]], triangle[i[1]]))
        for edge in edges:
            for vertex in edge:
                boundaries.add(vertex)
        return list(boundaries)

    def apply_boundary(self):
        """
        This method...
        """
        self.global_stiffness[self.boundary_array, :] = 0
        self.global_stiffness[:, self.boundary_array] = 0
        self.global_stiffness[
            np.transpose(np.array([self.boundary_array])),
            self.boundary_array] = np.identity(len(self.boundary_array))

        self.global_load[self.boundary_array] = 0
