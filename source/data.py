class Data(object):
    def __init__(self, vtk_file_name):
        self.name = vtk_file_name

    def getVerticesMatrix(self):
        numberOfVertices = 0
        verticesMatrix = []
        with open("{0}".format(self.name), "r") as file:
            for line in file:
                words = line.split()
                if words:
                    if words[0] == "POINTS":
                        numberOfVertices = int(words[1])
                        continue
                if numberOfVertices != 0:
                    verticesMatrix.append(list(map(int, words)))
                    numberOfVertices = numberOfVertices - 1
        return verticesMatrix

    def getConnectivityMatrix(self):
        numberOfCells = 0
        connectivityMatrix = []
        with open("{0}".format(self.name), "r") as file:
            for line in file:
                words = line.split()
                if words:
                    if words[0] == "CELLS":
                        numberOfCells = int(words[1])
                        continue
                if numberOfCells != 0:
                    connectivityMatrix.append(list(map(int, words)))
                    numberOfCells = numberOfCells - 1
        return connectivityMatrix
