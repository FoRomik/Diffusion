from source.data import Data
def solve(vtk_file_name, f, sigma, integration_order=4):
    data = Data("example.vtk")
    verticesMatrix = data.getVerticesMatrix()
    connectivityMatrix = data.getConnectivityMatrix()
