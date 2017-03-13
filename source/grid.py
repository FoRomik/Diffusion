"""
This module imports information from a vtk file format.
"""

import numpy as np



class Grid(object):
    """
    This class of methods exports vertices matrix and connectivity matrix from
    a vtk file format.
    """

    def __init__(self, vtk_file_name, output_file_name="result.vtk"):
        self.name = vtk_file_name
        self.output = output_file_name

    def get_vertices_matrix(self):
        """
        This method imports vertices matrix after "POINTS" keyword and breaks
        after finding the matrix.
        """
        number_of_vertices = None
        vertices_matrix = []
        with open("{0}".format(self.name), "r") as file:
            for line in file:
                words = line.split()
                if words:  # not an empty line
                    if words[0] == "POINTS":
                        number_of_vertices = int(words[1])
                        continue  # skip next if-statement and go to next line
                if number_of_vertices != 0 and number_of_vertices is not None:
                    vertices_matrix.append(
                        [float(word) for word in words[:-1]])
                    number_of_vertices = number_of_vertices - 1
                elif number_of_vertices == 0:
                    break  # search is done
        return np.array(vertices_matrix)

    def get_connectivity_matrix(self):
        """
        This method imports connectivity matrix after "CELLS" keyword and
        breaks after finding the matrix.
        """
        number_of_cells = None
        connectivity_matrix = []
        with open("{0}".format(self.name), "r") as file:
            for line in file:
                words = line.split()
                if words:
                    if words[0] == "CELLS":
                        number_of_cells = int(words[1])
                        continue
                if number_of_cells != 0 and number_of_cells is not None:
                    if int(words[0]) == 3: # If triangle
                        connectivity_matrix.append(
                            [int(word) for word in words[1:]])
                    number_of_cells = number_of_cells - 1
                elif number_of_cells == 0:
                    break
        return np.array(connectivity_matrix)

    def export(self, solution):
        """
        This method exports a vtk file with "solution" on z-axis.
        """
        number_of_vertices = None
        iterator = 0
        with open(self.output, "w") as new_file:
            with open(self.name, "r") as old_file:
                for line in old_file:
                    words = line.split()
                    if words:
                        if words[0] == "POINTS":
                            number_of_vertices = int(words[1])
                            new_file.write(line)
                            continue
                    if number_of_vertices != 0 and number_of_vertices is not None:
                        new_file.write("{0} {1} {2}\n".format(
                            words[0], words[1], solution[iterator]))
                        iterator += 1
                        number_of_vertices -= 1
                    else:
                        new_file.write("{0}".format(line))
