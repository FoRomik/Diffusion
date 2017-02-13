import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from source.grid import Grid
from source.squareGrid import SquareGrid
from source.fem import Fem

def plot(m, n, sigma, f, integrationOrder=4):
    U = solve(m, n, sigma, f)
    grid = SquareGrid(m, n)

    verticesMatrix = grid.verticesMatrix
    trianglesMatrix = grid.connectivityMatrix
    triangles = np.asarray(np.int_(trianglesMatrix))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(
        verticesMatrix[:, 0],
        verticesMatrix[:, 1],
        U,
        triangles=triangles,
        cmap=plt.cm.seismic)
    plt.show()


def solve(m, n, sigma, f, integrationOrder=4):
    grid = SquareGrid(m, n)
    fem = Fem(grid.verticesMatrix, grid.connectivityMatrix, grid.boundaryArray)
    return fem.solve(f, sigma, integrationOrder)
