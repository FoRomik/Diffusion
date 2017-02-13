import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sparse


class Fem(object):
    def __init__(self, verticesMatrix, connectivityMatrix, boundaryArray):
        self.verticesMatrix = verticesMatrix
        self.trianglesMatrix = connectivityMatrix
        self.boundaryArray = boundaryArray

        verticesNumber = len(self.verticesMatrix)
        self.K = np.zeros((verticesNumber, verticesNumber))  # (make sparse!)
        self.F = np.zeros((verticesNumber, 1))

    def solve(self, f, sigma, integrationOrder):
        for triangle in self.trianglesMatrix:
            J = self.getJ(triangle)
            invTransJ = np.linalg.inv(J).T
            detJ = abs(np.linalg.det(J))
            KLocal = self.getLocalStiffnessMatrix(invTransJ, detJ, sigma)
            FLocal = self.getLocalLoadVector(detJ, f)
            self.K[np.array([triangle]).T, triangle] += KLocal
            self.F[triangle] += FLocal
        self.applyBoundary()
        return np.linalg.solve(self.K, self.F).T[0]

    def getJ(self, triangle):
        [[x0, y0], [x1, y1], [x2, y2]] = self.verticesMatrix[triangle, :]
        return np.array([[x1 - x0, x2 - x0], [y1 - y0, y2 - y0]])

    def getLocalStiffnessMatrix(self, invTransJ, detJ, sigma):
        KLocal = np.zeros((3, 3))
        for i in range(3):
            for j in range(i+1):
                nPsi1 = self.getShapeFunctionDerivative(i)
                nPsi2 = self.getShapeFunctionDerivative(j)
                integralConstant = invTransJ.dot(nPsi1).dot(
                                   invTransJ.dot(nPsi2))*detJ
                integral = integralConstant*integrate.dblquad(
                           sigma, 0, 1, lambda x: 0, lambda x: 1-x)[0]
                KLocal[i][j] = integral
                KLocal[j][i] = integral
        return KLocal

    def getLocalLoadVector(self, detJ, f):
        FLocal = np.zeros((3, 1))
        for i in range(3):
            Psi = self.getShapeFunction(i)
            FLocal[i] = detJ*integrate.dblquad(
                lambda x, y: f(x, y)*Psi(x, y), 0, 1,
                lambda x: 0, lambda x: 1-x)[0]
        return FLocal

    def getShapeFunction(self, node):
        # z0 =(0,0) z1 =(1,0) z2 =(0,1)
        shapeFunctionArray = np.array(
            [lambda x, y: 1-x-y, lambda x, y: x, lambda x, y: y])
        return shapeFunctionArray[node]

    def getShapeFunctionDerivative(self, node):
        # z0 =(0,0) z1 =(1,0) z2 =(0,1)
        shapeFunctionMatrix = np.array([[-1, -1], [1, 0], [0, 1]])
        return shapeFunctionMatrix[node]

    def applyBoundary(self):
        self.K[self.boundaryArray, :] = 0
        self.K[:, self.boundaryArray] = 0
        self.K[np.array([self.boundaryArray]).T, self.boundaryArray] \
            = np.identity(len(self.boundaryArray))

        self.F[self.boundaryArray] = 0
