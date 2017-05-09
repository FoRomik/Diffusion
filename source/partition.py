import numpy as np
import networkx as nx
import pymetis

class Partition(object):
    """docstring for Partition."""
    def __init__(self, vertices_matrix, connectivity_matrix):
        self.vertices_matrix = np.matrix(vertices_matrix)
        self.connectivity_matrix = np.matrix(connectivity_matrix)

    def compute_adjacency(self):
        number_of_nodes = len(self.vertices_matrix)
        number_of_elements = len(self.connectivity_matrix)
        adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes))
        for i in range(number_of_elements):
            elements = np.array(self.connectivity_matrix[i]).flatten()
            elements.sort()
            adjacency_matrix[elements[0]][elements[1]] = 1
            adjacency_matrix[elements[1]][elements[0]] = 1
            adjacency_matrix[elements[0]][elements[2]] = 1
            adjacency_matrix[elements[2]][elements[0]] = 1
            adjacency_matrix[elements[1]][elements[2]] = 1
            adjacency_matrix[elements[2]][elements[1]] = 1
        return adjacency_matrix

    def split_graph(self, parts):
        adjacency_matrix = self.compute_adjacency()
        G = nx.from_numpy_matrix(adjacency_matrix)
        cuts, cell_tasks = pymetis.part_graph(parts, adjacency=G)
        cell_tasks = np.array(cell_tasks, dtype=np.int32)
        return cell_tasks

    def compute_connectivity(self, part, cell_tasks):
        included = []
        i = 0
        for element in cell_tasks:
            if element == part:
                included.append(i)
            i = i + 1
        connections = []
        for row in self.connectivity_matrix.tolist():
            if row[0] in included or row[1] in included or row[2] in included:
                connections.append(row)
        return np.matrix(connections)
