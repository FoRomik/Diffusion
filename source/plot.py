"""
This class of methods computes FDM/FEM and plots the result.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class Plot(object):
    """docstring for Plot."""
    def __init__(self, solution):
        self.solution = solution

    def FDM(self, X, Y):
        """
        This function plots solution of a FDM method on input rectangle.
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, self.solution, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def FEM(self, vertices_matrix, connectivity_matrix):
        """
        This function plots solution of a FEM method on input grid.
        """

        triangles = np.asarray(connectivity_matrix)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(
            vertices_matrix[:, 0],
            vertices_matrix[:, 1],
            self.solution,
            triangles=triangles,
            cmap=plt.cm.coolwarm)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
