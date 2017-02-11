import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from source.grid import Grid
from source.squareGrid import SquareGrid
from source.fem import Fem

def solve(m,n,vtkFileName="example.vtk", f=1, sigma=1, integrationOrder=4):
    grid = SquareGrid(m, n)
    #grid = Grid(vtkFileName)
    return Fem(grid.verticesMatrix, grid.connectivityMatrix,
        grid.boundaryArray).solution(integrationOrder)

def plot(m, n, vtkFileName="example.vtk", f=1, sigma=1, integrationOrder=4):
    U = solve(m, n)
    grid = SquareGrid(m, n)

    verticesMatrix = grid.verticesMatrix
    trianglesMatrix = grid.connectivityMatrix
    triangles = np.asarray(np.int_(trianglesMatrix))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(verticesMatrix[:,0], verticesMatrix[:,1], U, triangles = triangles)
    plt.show()
