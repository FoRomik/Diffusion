"""
This module imports information from a vtk file format.
"""

import numpy as np
import os


class Grid(object):
    """
    This class of methods exports vertices matrix and connectivity matrix from
    a vtk file format.
    """

    def __init__(self, vtk_file_name):
        self.name = vtk_file_name

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

    def export(self, vertices_matrix, connectivity_matrix):
        filename = os.path.basename(self.name)
        number_of_vertices = len(vertices_matrix)
        number_of_triangles = len(connectivity_matrix)
        with open(self.name, "w") as file:
            file.write("# vtk DataFile Version 2.0\n")
            file.write("{0}, Created by Gmsh\n".format(filename))
            file.write("ASCII\n")
            file.write("DATASET UNSTRUCTURED_GRID\n")
            file.write("POINTS {0} double\n".format(number_of_vertices))
            for i in range(number_of_vertices):
                file.write("{0} {1} 0\n".format(
                    vertices_matrix[i][0],
                    vertices_matrix[i][1]))
            file.write("\nCELLS {0} 0\n".format(number_of_triangles))
            for i in range(number_of_triangles):
                file.write("3 {0} {1} {2}\n".format(
                    connectivity_matrix[i, 0],
                    connectivity_matrix[i, 1],
                    connectivity_matrix[i, 2]))
            file.write("\nCELL_TYPES {0}\n".format(number_of_triangles))
            for i in range(number_of_triangles):
                file.write("5\n")
