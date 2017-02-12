import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sparse


class Fem(object):
    def __init__(self, verticesMatrix, connectivityMatrix, boundaryArray,
                 f, sigma, integrationOrder=4):
        self.verticesMatrix = verticesMatrix
        self.trianglesMatrix = connectivityMatrix
        self.boundaryArray = boundaryArray

        verticesNumber = len(self.verticesMatrix)
        self.K = np.zeros((verticesNumber, verticesNumber))  # (make sparse!)
        self.F = np.zeros((verticesNumber, 1))

        for triangle in self.trianglesMatrix:
            Jinv = np.linalg.inv(self.getJ(triangle)).T
            Ke = self.getLocalStiffnessMatrix(Jinv, sigma)
            Fe = self.getLocalLoadVector(Jinv, f)
            self.K[[triangle].T, triangle] += Ke
            self.F[triangle] += Fe
        self.applyBoundary()

        self.solution = np.linalg.solve(K, F).T[0]

    def getJ(self, triangle):
        [[x0, y0], [x1, y1], [x2, y2]] = self.verticesMatrix[[triangle].T, :]
        return np.array([[x1 - x0, x2 - x0], [y1 - y0, y2 - y0]])

    def getLocalStiffnessMatrix(self):
        return Area*np.dot(np.transpose(grad), grad)

    def getLocalLoadVector(self):
        return Area/3  # f(x,y)=1

    def applyBoundary(self):
        self.K[self.boundaryArray, :] = 0
        self.K[:, self.boundaryArray] = 0
        self.K[[self.boundaryArray].T, self.boundaryArray] = np.identity(
            len(self.boundaryArray))

        self.F[self.boundaryArray] = 0
