import numpy as np

class SquareGrid(object):
    def __init__(self,m,n):
        self.verticesMatrix = self.createVerticesMatrix(m, n)
        self.connectivityMatrix = self.createConnectivityMatrix(m, n)
        self.boundaryArray = self.createBoundaryArray(m, n)

    def createVerticesMatrix(self, m, n):
        matrix = [[0 for i in range(2)] for j in range(m*n)]
        for i in range(m*n):
            for j in range(2):
                if j == 0:
                    matrix[i][j] = i%m/(m - 1)
                else: matrix[i][j] = (i//m)/(m - 1)
        return np.array(matrix)

    def createConnectivityMatrix(self, m, n):
        matrix = [[0,1,m+1],[0,m+1,m]]
        i = len(matrix)
        j = len(matrix[0])
        matrix = np.kron(matrix, np.ones((m-1,1))) + np.kron(np.ones([i,j]),np.array([range(m-1)]).T)
        i = len(matrix)
        j = len(matrix[0])
        matrix = np.kron(matrix, np.ones((n-1,1))) + np.kron(np.ones([i,j]),np.array([range(0,n-1)]).T*m)
        return matrix

    def createBoundaryArray(self, m, n):
        return np.hstack((
                np.arange(m), # bottom
                np.arange(m,m*n,m), # left
                np.arange(2*m-1,m*n,m), # right
                np.arange(m*n-m+1,m*n-1)))  # top
