"""
This module...
"""

import numpy as np
import scipy.sparse as sparse
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
        self.vertices_number = len(self.vertices_matrix)
        self.global_stiffness = np.zeros((
            self.vertices_number, self.vertices_number))  # (make sparse!)
        self.global_load = np.zeros((self.vertices_number, 1))

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
        partial_solution =  np.linalg.solve(
            self.global_stiffness, self.global_load).flatten()
        return self.modify_solution(partial_solution)

    def local_assembly(self, element, sigma, function):
        """
        This method computes local stiffness matrix and local load vector.
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
        This method adds local stiffness matrix and local load vector to
        global stiffness matrix and global load vector.
        """
        self.global_stiffness[
            np.transpose(np.array([element])), element] += stiffness
        self.global_load[element] += load


    def get_mapping_matrix(self, element):
        """
        This method computes mapping matrix from original to reference
        triangle.
        """
        [[x_0, y_0], [x_1, y_1], [x_2, y_2]] = \
            self.vertices_matrix[element, :]
        return np.array([[x_1 - x_0, x_2 - x_0], [y_1 - y_0, y_2 - y_0]])

    def get_boundary_array(self):
        """
        This method performs an algorithm on connectivity matrix to find nodes
        on the edge of graph.
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
        This method modifies global stiffness matrix and load vector so the
        boundary conditions are applied.
        """
        P = np.delete(
            np.identity(self.vertices_number), self.boundary_array, axis=1)
        self.global_stiffness = np.dot(
            np.dot(np.transpose(P), self.global_stiffness), P)
        self.global_load = np.dot(np.transpose(P), self.global_load)

    def modify_solution(self, solution):
        """
        This method modifies the solution so boundary vertices are included and
        are set to 0.
        """
        interior = np.delete(
            np.arange(self.vertices_number), self.boundary_array)
        new_solution = np.zeros(self.vertices_number)
        new_solution[interior] = solution
        return new_solution
