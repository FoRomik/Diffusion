"""
This module contains information about reference triangle shape functions and
numerically integrates over the triangle to get local stiffness matrix and load
vector.
"""

import numpy as np
from source.integration import GaussianIntegration


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
