from source.grid import Grid
from source.partition import Partition

grid = Grid("bend.vtk")
connectivity_matrix = grid.get_connectivity_matrix()
vertices_matrix = grid.get_vertices_matrix()
partition = Partition(vertices_matrix, connectivity_matrix)
parts = partition.split_graph(2)
connectivity0 = partition.compute_connectivity(0, parts)
connectivity1 = partition.compute_connectivity(1, parts)

grid0 = Grid("export0.vtk")
grid1 = Grid("export1.vtk")

grid0.export(vertices_matrix, connectivity0)
grid1.export(vertices_matrix, connectivity1)
