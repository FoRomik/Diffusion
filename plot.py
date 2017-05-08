"""
This module computes finite element method and plots the result.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from source.grid import Grid
from source.square import SquareGrid
from source.fem import Fem
from source.fdm import Fdm


def plotFdm(m, n, hx, hy, sigma, dx_sigma, dy_sigma, function):
    """
    This function plots solution of a FDM method on input rectangle.
    """
    grid = Fdm(m, n, hx, hy)
    solution = grid.solve(sigma, dx_sigma, dy_sigma, function)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(0, m)
    Y = np.arange(0, n)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()

def plotFem(vtk_file_name, sigma, function, integration_order=4):
    """
    This function plots solution of a FEM method on input grid.
    """
    grid = Grid(vtk_file_name)

    vertices_matrix = grid.get_vertices_matrix()
    connectivity_matrix = grid.get_connectivity_matrix()

    triangles = np.asarray(connectivity_matrix)

    fem = Fem(vertices_matrix, connectivity_matrix)
    solution = fem.solve(sigma, function, integration_order)

    fig = plt.figure()
    axis = fig.gca(projection='3d')
    axis.plot_trisurf(
        vertices_matrix[:, 0],
        vertices_matrix[:, 1],
        solution,
        triangles=triangles,
        cmap=plt.cm.seismic)
    plt.show()

def plotFem_square(columns, rows, sigma, function, integration_order=4):
    """
    This function plots solution of a FEM method on square generated grid.
    """
    grid = SquareGrid(columns, rows)

    vertices_matrix = grid.get_vertices_matrix()
    connectivity_matrix = grid.get_connectivity_matrix()

    triangles = np.asarray(np.int_(connectivity_matrix))

    fem = Fem(vertices_matrix, connectivity_matrix)
    solution = fem.solve(sigma, function, integration_order)

    fig = plt.figure()
    axis = fig.gca(projection='3d')
    axis.plot_trisurf(
        vertices_matrix[:, 0],
        vertices_matrix[:, 1],
        solution,
        triangles=triangles,
        cmap=plt.cm.seismic)
    plt.show()
