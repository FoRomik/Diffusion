import numpy as np

class SquareGrid(object):
    def __init__(self,m,n):
        self.verticesMatrix = np.array([[(j%m)/(m - 1) if i==0 else (j//m)/(m - 1) for i in range(2)]for j in range(m*n)])
        connectivityMatrix = [[0,1,m+1], [0,m+1,m]]
        connectivityMatrix = np.kron(connectivityMatrix, np.ones((m-1,1))) + np.kron(np.ones([len(connectivityMatrix),len(connectivityMatrix[0])]),np.transpose([range(0,m-1)]))
        self.connectivityMatrix = np.kron(connectivityMatrix,np.ones((n-1,1))) + np.kron(np.ones([len(connectivityMatrix),len(connectivityMatrix[0])]),np.dot(np.transpose([range(0,n-1)]),m))
        self.boundaryArray = np.hstack((np.arange(m),np.arange(m,m*n,m),np.arange(2*m-1,m*n,m),np.arange(m*n-m+1,m*n-1)))  # bottom, left, right, top
