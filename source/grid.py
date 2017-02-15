"""
This module imports information from a vtk file format.
"""

import numpy as np


class Grid(object):
    """
    This class of methods exports vertices matrix and connectivity matrix from
    a vtk file format.
    """
    def __init__(self, vtk_file_name="example.vtk"):
        self.name = vtk_file_name
        self.connectivity_matrix = [[]]

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
                    connectivity_matrix.append(
                        [int(word) for word in words[1:]])
                    number_of_cells = number_of_cells - 1
                elif number_of_cells == 0:
                    break
        return np.array(connectivity_matrix)
