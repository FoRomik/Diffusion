"""
This module solves diffusion equation using Finite elements method.
"""

import numpy as np
from source import triangle


class Fem(object):
    """
    This class of methods solves diffusion equation using FEM based on input
    vertices matrix and connectivity matrix, function f, function sigma and
    integration order.
    """
    def __init__(self, vertices_matrix, connectivity_matrix):
        self.vertices_matrix = vertices_matrix
        self.connectivity_matrix = connectivity_matrix
        self.boundary_array = self.get_boundary_array()

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
            self.assembly(element, sigma, function, integration_order)
        self.apply_boundary()
        partial_solution = np.linalg.solve(
            self.global_stiffness, self.global_load).flatten()
        return self.modify_solution(partial_solution)

    def assembly(self, element, sigma, function, integration_order):
        """
        This method computes local stiffness matrix and local load vector and
        adds them to global stiffness matrix and global load vector.
        """
        [[x_0, y_0], [x_1, y_1], [x_2, y_2]] = self.vertices_matrix[element, :]
        jacobian = np.array([[x_1 - x_0, x_2 - x_0], [y_1 - y_0, y_2 - y_0]])
        x_mapping = lambda x, y: x_0 + (x_1 - x_0)*x + (x_2 - x_0)*y
        y_mapping = lambda x, y: y_0 + (y_1 - y_0)*x + (y_2 - y_0)*y
        new_function = lambda x, y: function(x_mapping(x, y), y_mapping(x, y))
        new_sigma = lambda x, y: sigma(x_mapping(x,y), y_mapping(x, y))
        inverse_transpose = np.transpose(np.linalg.inv(jacobian))
        determinant = abs(np.linalg.det(jacobian))
        local_stiffness = triangle.get_local_stiffness(
            inverse_transpose, determinant, new_sigma, integration_order)
        local_load = triangle.get_local_load(
            determinant, new_function, integration_order)
        self.global_stiffness[
            np.transpose(np.array([element])), element] += local_stiffness
        self.global_load[element] += local_load

    def get_boundary_array(self):
        """
        This method performs an algorithm on connectivity matrix to find nodes
        on the edge of graph.
        """
        edges = set()
        boundaries = set()
        for element in self.connectivity_matrix:
            element = sorted(element)
            for i in [(0, 1), (0, 2), (1, 2)]:
                if (element[i[0]], element[i[1]]) not in edges:
                    edges.add((element[i[0]], element[i[1]]))
                else: edges.remove((element[i[0]], element[i[1]]))
        for edge in edges:
            for vertex in edge:
                boundaries.add(vertex)
        return list(boundaries)

    def apply_boundary(self):
        """
        This method modifies global stiffness matrix and load vector so the
        boundary conditions are applied.
        """
        interior_matrix = np.delete(
            np.identity(self.vertices_number), self.boundary_array, axis=1)
        self.global_stiffness = np.dot(
            np.dot(np.transpose(interior_matrix),
                   self.global_stiffness), interior_matrix)
        self.global_load = np.dot(
            np.transpose(interior_matrix), self.global_load)

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
