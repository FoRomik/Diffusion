import numpy as np
from source.data import Data
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D

#def solve(vtk_file_name, f, sigma, integration_order=4):
#data = Data("example.vtk")
#verticesMatrix = data.getVerticesMatrix()
#connectivityMatrix = data.getConnectivityMatrix()

#numberOfVertices = len(verticesMatrix)
#numberOfTriangles = len(connectivityMatrix)

def solve(m,n):
    p = np.array([[(j%m)/(m - 1) if i==0 else (j//m)/(m - 1) for i in range(2)]for j in range(m*n)])
    t = [[1,2,m+2], [1,m+2,m+1]]
    t = np.kron(t, np.ones((m-1,1))) + np.kron(np.ones([len(t),len(t[0])]),np.transpose([range(0,m-1)]))
    t = np.kron(t,np.ones((n-1,1))) + np.kron(np.ones([len(t),len(t[0])]),np.dot(np.transpose([range(0,n-1)]),m))

    b = np.hstack((np.arange(1,m+1),np.arange(m+1,m*n+1,m),np.arange(2*m,m*n+1,m),np.arange(m*n-m+2,m*n))) - 1 # bottom, left, right, top

    N = len(p)
    T = len(t)

    K = np.zeros((N,N))
    F = np.zeros((N,1))

    for e in range(T):
        nodes = t[e] - 1
        nodes = np.int_(nodes)
        Pe = np.hstack((np.ones((3,1)),p[nodes,:]))
        Area = abs(np.linalg.det(Pe))/2
        C = np.linalg.inv(Pe)
        grad = [C[1],C[2]]
        Ke = Area*np.dot(np.transpose(grad),grad)
        Fe = Area/3
        K[np.transpose([nodes]),nodes] += Ke
        F[nodes] += Fe

    K[b,:] = 0
    K[:,b] = 0
    F[b] = 0;
    K[np.transpose([b]),b] = np.identity(len(b))

    U = np.linalg.solve(K, F)
    U = np.transpose(U)[0]

    triangles = np.asarray(np.int_(t)-1)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(p[:,0], p[:,1],U, triangles = triangles)
    plt.show()
