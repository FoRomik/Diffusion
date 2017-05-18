"""
This module solves diffusion equation using Finite elements method.
"""

import numpy as np


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
            self.vertices_number, self.vertices_number))
        self.global_load = np.zeros((self.vertices_number, 1))

    def solve(self, sigma, function, integration_order=4):
        """
        This is the main method of the Fem class which solves the diffusion
        equation.
        """
        for element in self.connectivity_matrix:
            self.assembly(element, sigma, function, integration_order)
        self.apply_boundary()

    def get_matrix_a(self):
        return self.global_stiffness

    def get_vector_b(self):
        return self.global_load

    def assembly(self, element, sigma, function, integration_order):
        """
        This method computes local stiffness matrix and local load vector and
        adds them to global stiffness matrix and global load vector.
        """
        new_function = self.transform_function(element, function)
        new_sigma = self.transform_function(element, sigma)
        jacobian = self.get_jacobian(element)
        inverse_transpose = np.transpose(np.linalg.inv(jacobian))
        determinant = abs(np.linalg.det(jacobian))
        local_stiffness = get_local_stiffness(
            inverse_transpose, determinant, new_sigma, integration_order)
        local_load = get_local_load(
            determinant, new_function, integration_order)
        self.global_stiffness[
            np.transpose(np.array([element])), element] += local_stiffness
        self.global_load[element] += local_load

    def get_jacobian(self, element):
        """
        This method computes Jacobian matrix based on input element.
        """
        [[x_0, y_0], [x_1, y_1], [x_2, y_2]] = self.vertices_matrix[element, :]
        return np.array([[x_1 - x_0, x_2 - x_0], [y_1 - y_0, y_2 - y_0]])

    def transform_function(self, element, function):
        """
        This method transforms function to reference triangle coordinate
        system.
        """
        [[x_0, y_0], [x_1, y_1], [x_2, y_2]] = self.vertices_matrix[element, :]
        x_mapping = lambda x, y: x_0 + (x_1 - x_0)*x + (x_2 - x_0)*y
        y_mapping = lambda x, y: y_0 + (y_1 - y_0)*x + (y_2 - y_0)*y
        return lambda x, y: function(x_mapping(x, y), y_mapping(x, y))

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


"""
This module contains information about reference triangle shape functions and
numerically integrates over the triangle to get local stiffness matrix and load
vector.
"""

import numpy as np


def get_shape_function(node):
    """
    This function returns shape function for each of triangle's node.
    """
    # z0 =(0,0) z1 =(1,0) z2 =(0,1)
    shape_function_array = np.array(
        [lambda x, y: 1-x-y, lambda x, y: x, lambda x, y: y])
    return shape_function_array[node]


def get_shape_function_derivative(node):
    """
    This function returns shape function derivative for each of triangle's
    node.
    """
    # z0 =(0,0) z1 =(1,0) z2 =(0,1)
    shape_function_matrix = np.array([[-1, -1], [1, 0], [0, 1]])
    return shape_function_matrix[node]


def get_local_stiffness(
        inverse_transpose, determinant, sigma, integration_order):
    """
    This function calculates local stiffness matrix for triangle based on
    triangle mapping information (inverse_transpose_matrix, determinant,
    sigma).
    """
    local_stiffness = np.zeros((3, 3))
    for i in range(3):
        for j in range(i+1):
            nabla_psi_1 = get_shape_function_derivative(i)
            nabla_psi_2 = get_shape_function_derivative(j)
            integral_constant = inverse_transpose.dot(
                nabla_psi_1).dot(inverse_transpose.dot(
                    nabla_psi_2))*determinant
            integral = integral_constant*GaussianIntegration(
                integration_order).integrate(sigma)
            local_stiffness[i][j] = integral
            local_stiffness[j][i] = integral
    return local_stiffness


def get_local_load(determinant, function, integration_order):
    """
    This function calculates local load vector for triangle based on triangle
    mapping information (inverse_transpose, determinant, function f).
    """
    local_load = np.zeros((3, 1))
    psi = lambda x, y: 0
    for i in range(3):
        psi = get_shape_function(i)
        local_load[i] = determinant*GaussianIntegration(
            integration_order).integrate(lambda x, y: psi(x, y)*function(x, y))
    return local_load

"""
This module performes a Gaussian integration on reference triangle.
"""
import numpy as np


class GaussianIntegration(object):
    """
    This class contains methods for calculating Gaussian integral of order n.
    """
    def __init__(self, integration_order):
        self.integration_order = integration_order
        self.weights = self.get_weights()

    def integrate(self, function):
        """
        This is the main method.
        """
        area = 0.5
        integral = 0.
        for point in self.weights:
            integral += function(point[0], point[1])*point[2]
        return integral*area

    def get_weights(self):
        """
        This method returns matrix of weights based on input integration order.
        """
        if self.integration_order == 1:
            weights = np.array(
                [[0.33333333333333, 0.33333333333333, 1.00000000000000]])
        elif self.integration_order == 2:
            weights = np.array(
                [[0.16666666666667, 0.16666666666667, 0.33333333333333],
                 [0.16666666666667, 0.66666666666667, 0.33333333333333],
                 [0.66666666666667, 0.16666666666667, 0.33333333333333]])
        elif self.integration_order == 3:
            weights = np.array(
                [[0.33333333333333, 0.33333333333333, -0.56250000000000],
                 [0.20000000000000, 0.20000000000000, 0.52083333333333],
                 [0.20000000000000, 0.60000000000000, 0.52083333333333],
                 [0.60000000000000, 0.20000000000000, 0.52083333333333]])
        elif self.integration_order == 4:
            weights = np.array(
                [[0.44594849091597, 0.44594849091597, 0.22338158967801],
                 [0.44594849091597, 0.10810301816807, 0.22338158967801],
                 [0.10810301816807, 0.44594849091597, 0.22338158967801],
                 [0.09157621350977, 0.09157621350977, 0.10995174365532],
                 [0.09157621350977, 0.81684757298046, 0.10995174365532],
                 [0.81684757298046, 0.09157621350977, 0.10995174365532]])
        elif self.integration_order == 5:
            weights = np.array(
                [[0.33333333333333, 0.33333333333333, 0.22500000000000],
                 [0.47014206410511, 0.47014206410511, 0.13239415278851],
                 [0.47014206410511, 0.05971587178977, 0.13239415278851],
                 [0.05971587178977, 0.47014206410511, 0.13239415278851],
                 [0.10128650732346, 0.10128650732346, 0.12593918054483],
                 [0.10128650732346, 0.79742698535309, 0.12593918054483],
                 [0.79742698535309, 0.10128650732346, 0.12593918054483]])
        elif self.integration_order == 6:
            weights = np.array(
                [[0.24928674517091, 0.24928674517091, 0.11678627572638],
                 [0.24928674517091, 0.50142650965818, 0.11678627572638],
                 [0.50142650965818, 0.24928674517091, 0.11678627572638],
                 [0.06308901449150, 0.06308901449150, 0.05084490637021],
                 [0.06308901449150, 0.87382197101700, 0.05084490637021],
                 [0.87382197101700, 0.06308901449150, 0.05084490637021],
                 [0.31035245103378, 0.63650249912140, 0.08285107561837],
                 [0.63650249912140, 0.05314504984482, 0.08285107561837],
                 [0.05314504984482, 0.31035245103378, 0.08285107561837],
                 [0.63650249912140, 0.31035245103378, 0.08285107561837],
                 [0.31035245103378, 0.05314504984482, 0.08285107561837],
                 [0.05314504984482, 0.63650249912140, 0.08285107561837]])
        elif self.integration_order == 7:
            weights = np.array(
                [[0.33333333333333, 0.33333333333333, -0.14957004446768],
                 [0.26034596607904, 0.26034596607904, 0.17561525743321],
                 [0.26034596607904, 0.47930806784192, 0.17561525743321],
                 [0.47930806784192, 0.26034596607904, 0.17561525743321],
                 [0.06513010290222, 0.06513010290222, 0.05334723560884],
                 [0.06513010290222, 0.86973979419557, 0.05334723560884],
                 [0.86973979419557, 0.06513010290222, 0.05334723560884],
                 [0.31286549600487, 0.63844418856981, 0.07711376089026],
                 [0.63844418856981, 0.04869031542532, 0.07711376089026],
                 [0.04869031542532, 0.31286549600487, 0.07711376089026],
                 [0.63844418856981, 0.31286549600487, 0.07711376089026],
                 [0.31286549600487, 0.04869031542532, 0.07711376089026],
                 [0.04869031542532, 0.63844418856981, 0.07711376089026]])
        elif self.integration_order == 8:
            weights = np.array(
                [[0.33333333333333, 0.33333333333333, 0.14431560767779],
                 [0.45929258829272, 0.45929258829272, 0.09509163426728],
                 [0.45929258829272, 0.08141482341455, 0.09509163426728],
                 [0.08141482341455, 0.45929258829272, 0.09509163426728],
                 [0.17056930775176, 0.17056930775176, 0.10321737053472],
                 [0.17056930775176, 0.65886138449648, 0.10321737053472],
                 [0.65886138449648, 0.17056930775176, 0.10321737053472],
                 [0.05054722831703, 0.05054722831703, 0.03245849762320],
                 [0.05054722831703, 0.89890554336594, 0.03245849762320],
                 [0.89890554336594, 0.05054722831703, 0.03245849762320],
                 [0.26311282963464, 0.72849239295540, 0.02723031417443],
                 [0.72849239295540, 0.00839477740996, 0.02723031417443],
                 [0.00839477740996, 0.26311282963464, 0.02723031417443],
                 [0.72849239295540, 0.26311282963464, 0.02723031417443],
                 [0.26311282963464, 0.00839477740996, 0.02723031417443],
                 [0.00839477740996, 0.72849239295540, 0.02723031417443]])
        else:
            raise "Error: bad input of n."
        return weights
