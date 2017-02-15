"""
This module solves diffusion equation using FEM by reading vtk file and
exporting a vtk file (result.vtk).
"""
from source.grid import Grid
from source.fem import Fem


def solve(vtk_file_name, sigma, function, integration_order=4):
    """
    This function
    """
    grid = Grid(vtk_file_name)
    solution = Fem(
        grid.get_vertices_matrix(),
        grid.get_connectivity_matrix()
        ).solve(sigma, function, integration_order)
    grid.export(solution)
