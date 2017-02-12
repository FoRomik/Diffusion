import numpy as np


class Grid(object):
    """
    This is a legacy vtk file reader.
    """
    def __init__(self, vtk_file_name="example.vtk"):
        self.name = vtk_file_name
        self.verticesMatrix = self.getVerticesMatrix()
        self.connectivityMatrix = self.getConnectivityMatrix()
        self.boundaryArray = self.getBoundaryArray()

    def getVerticesMatrix(self):
        numberOfVertices = None
        verticesMatrix = []
        with open("{0}".format(self.name), "r") as file:
            for line in file:
                words = line.split()
                if words:  # not an empty line
                    if words[0] == "POINTS":
                        numberOfVertices = int(words[1])
                        continue  # skip next if-statement and go to next line
                if numberOfVertices != 0 and numberOfVertices is not None:
                    verticesMatrix.append(list(map(int, words))[:-1])
                    numberOfVertices = numberOfVertices - 1
                elif numberOfVertices == 0:
                    break  # search is done
        return np.array(verticesMatrix)

    def getConnectivityMatrix(self):
        numberOfCells = None
        connectivityMatrix = []
        with open("{0}".format(self.name), "r") as file:
            for line in file:
                words = line.split()
                if words:
                    if words[0] == "CELLS":
                        numberOfCells = int(words[1])
                        continue
                if numberOfCells != 0 and numberOfCells is not None:
                    connectivityMatrix.append(list(map(int, words))[1:])
                    numberOfCells = numberOfCells - 1
                elif numberOfCells == 0:
                    break
        return np.array(connectivityMatrix)

    def getBoundaryArray(self):
        pass
