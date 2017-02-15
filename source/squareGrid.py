"""
This module creates triangulated square mesh.
"""

import numpy as np


class SquareGrid(object):
    """
    This class generates a mesh based on the number of input rows and columns.
    """
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows

    def get_vertices_matrix(self):
        """
        This method returns vertices matrix for specified square grid.
        """
        matrix = [[0 for i in range(2)] for j in range(self.columns*self.rows)]
        for i in range(self.columns*self.rows):
            for j in range(2):
                if j == 0:
                    matrix[i][j] = i % self.columns/(self.columns - 1)
                else:
                    matrix[i][j] = (i//self.columns)/(self.columns - 1)
        return np.array(matrix)

    def get_connectivity_matrix(self):
        """
        This method returns connectivity matrix for specified square grid.
        """
        matrix = [[0, 1, self.columns+1], [0, self.columns+1, self.columns]]
        i = len(matrix)
        j = len(matrix[0])
        matrix = np.kron(matrix, np.ones((self.columns-1, 1))) \
            + np.kron(np.ones([i, j]),
                np.transpose(np.array([np.arange(self.columns-1)])))
        i = len(matrix)
        j = len(matrix[0])
        matrix = np.kron(matrix, np.ones((self.rows-1, 1))) \
            + np.kron(np.ones([i, j]), np.transpose(
                np.array([np.arange(self.rows-1)]))*self.columns)
        return np.array(np.int_(matrix))
