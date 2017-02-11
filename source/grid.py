class Grid(object):
    def __init__(self, vtk_file_name):
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
                if words: # not an empty line
                    if words[0] == "POINTS":
                        numberOfVertices = int(words[1])
                        continue # skip next if-statement and go to next line
                if numberOfVertices != 0 and numberOfVertices != None:
                    verticesMatrix.append(list(map(int, words))[:-1])
                    numberOfVertices = numberOfVertices - 1
                elif numberOfVertices == 0: break # search is done
        return verticesMatrix

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
                if numberOfCells != 0 and numberOfCells != None:
                    connectivityMatrix.append(list(map(int, words))[1:])
                    numberOfCells = numberOfCells - 1
                elif numberOfCells == 0: break
        return connectivityMatrix

    def getBoundaryArray(self):
        pass
