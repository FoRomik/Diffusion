import numpy as np
import scipy

class Fem(object):
    def __init__(self, verticesMatrix, connectivityMatrix, boundaryArray, f=1, sigma=1):
        self.verticesMatrix = verticesMatrix
        self.trianglesMatrix = connectivityMatrix
        self.boundaryArray = boundaryArray

    def solution(self, integrationOrder):
        verticesNumber = len(self.verticesMatrix)

        K = np.zeros((verticesNumber,verticesNumber)) # Make this sparse!
        F = np.zeros((verticesNumber,1))

        for triangle in self.trianglesMatrix:
            nodes = triangle
            nodes = np.int_(nodes)
            Pe = np.hstack((np.ones((3,1)),self.verticesMatrix[nodes,:])) # 3 by 3 matrix with rows=[1 xcorner ycorner]
            Area = abs(np.linalg.det(Pe))/2 # area of triangle e = half of parallelogram area
            C = np.linalg.inv(Pe)
            grad = [C[1],C[2]]
            Ke = Area*np.dot(np.transpose(grad),grad)
            Fe = Area/3 # f(x,y)=1
            K[np.transpose([nodes]),nodes] += Ke
            F[nodes] += Fe

        K[self.boundaryArray,:] = 0
        K[:,self.boundaryArray] = 0
        F[self.boundaryArray] = 0;
        K[np.transpose([self.boundaryArray]),self.boundaryArray] = np.identity(len(self.boundaryArray))

        U = np.linalg.solve(K, F)
        U = np.transpose(U)[0]
        return U
