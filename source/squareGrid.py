"""
This module creates triangulated square mesh.
"""

import numpy as np


class SquareGrid(object):
    """
    This class...
    """
    def __init__(self, columns, rows):
        self.vertices_matrix = create_vertices_matrix(columns, rows)
        self.connectivity_matrix = create_connectivity_matrix(columns, rows)
        self.boundary_array = create_boundary_array(columns, rows)

def create_vertices_matrix(columns, rows):
    """
    This function...
    """
    matrix = [[0 for i in range(2)] for j in range(columns*rows)]
    for i in range(columns*rows):
        for j in range(2):
            if j == 0:
                matrix[i][j] = i % columns/(columns - 1)
            else:
                matrix[i][j] = (i//columns)/(columns - 1)
    return np.array(matrix)

def create_connectivity_matrix(columns, rows):
    """
    This function...
    """
    matrix = [[0, 1, columns+1], [0, columns+1, columns]]
    i = len(matrix)
    j = len(matrix[0])
    matrix = np.kron(matrix, np.ones((columns-1, 1))) \
        + np.kron(np.ones([i, j]),
                  np.transpose(np.array([np.arange(columns-1)])))
    i = len(matrix)
    j = len(matrix[0])
    matrix = np.kron(matrix, np.ones((rows-1, 1))) \
        + np.kron(np.ones([i, j]),
                  np.transpose(np.array([np.arange(rows-1)]))*columns)
    return np.array(np.int_(matrix))

def create_boundary_array(columns, rows):
    """
    This function...
    """
    return np.array(np.hstack((
        np.arange(columns),  # bottom
        np.arange(columns, columns*rows, columns),  # left
        np.arange(2*columns-1, columns*rows, columns),  # right
        np.arange(columns*rows-columns+1, columns*rows-1))))  # top
