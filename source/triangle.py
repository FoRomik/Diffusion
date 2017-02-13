"""
This module...
"""

import numpy as np
import scipy.integrate as integrate


class Triangle(object):
    """
    This class...
    """

    def __init__(self, inverse_transpose, determinant):
        self.inverse_transpose = inverse_transpose
        self.determinant = determinant

    def get_local_stiffness(self, sigma):
        """
        This method...
        """
        local_stiffness = np.zeros((3, 3))
        for i in range(3):
            for j in range(i+1):
                nabla_psi_1 = get_shape_function_derivative(i)
                nabla_psi_2 = get_shape_function_derivative(j)
                integral_constant = self.inverse_transpose.dot(
                    nabla_psi_1).dot(self.inverse_transpose.dot(
                        nabla_psi_2))*self.determinant
                integral = integral_constant*integrate.dblquad(
                    sigma, 0, 1, lambda x: 0, lambda x: 1-x)[0]
                local_stiffness[i][j] = integral
                local_stiffness[j][i] = integral
        return local_stiffness

    def get_local_load(self, function):
        """
        This method...
        """
        local_load = np.zeros((3, 1))
        psi = lambda x, y: 0
        for i in range(3):
            psi = get_shape_function(i)
            local_load[i] = self.determinant*integrate.dblquad(
                lambda x, y: function(x, y)*psi(x, y), 0, 1,
                lambda x: 0, lambda x: 1-x)[0]
        return local_load

def get_shape_function(node):
    """
    This function...
    """
    # z0 =(0,0) z1 =(1,0) z2 =(0,1)
    shape_function_array = np.array(
        [lambda x, y: 1-x-y, lambda x, y: x, lambda x, y: y])
    return shape_function_array[node]

def get_shape_function_derivative(node):
    """
    This function...
    """
    # z0 =(0,0) z1 =(1,0) z2 =(0,1)
    shape_function_matrix = np.array([[-1, -1], [1, 0], [0, 1]])
    return shape_function_matrix[node]
