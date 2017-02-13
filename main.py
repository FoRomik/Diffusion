"""
This module...
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from source.grid import Grid
from source.squareGrid import SquareGrid
from source.fem import Fem

def plot(columns, rows, sigma, function, integration_order=4):
    """
    This function...
    """
    solution = solve(columns, rows, sigma, function)
    grid = SquareGrid(columns, rows)

    vertices_matrix = grid.vertices_matrix
    triangles_matrix = grid.connectivity_matrix
    triangles = np.asarray(np.int_(triangles_matrix))

    fig = plt.figure()
    axis = fig.gca(projection='3d')
    axis.plot_trisurf(
        vertices_matrix[:, 0],
        vertices_matrix[:, 1],
        solution,
        triangles=triangles,
        cmap=plt.cm.seismic)
    plt.show()


def solve(columns, rows, sigma, function, integration_order=4):
    """
    This function...
    """
    grid = SquareGrid(columns, rows)
    fem = Fem(grid.vertices_matrix, grid.connectivity_matrix, grid.boundary_array)
    return fem.solve(function, sigma, integration_order)
