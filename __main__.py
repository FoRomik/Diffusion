from source.grid import Grid
from source.fem import Fem
from source.fdm import Fdm
from source.solve import Solve
from source.plot import Plot

# Example FEM solver and plotter:

femGrid = Grid("../vtk examples/airfoil_exterior.vtk")
fem = Fem(femGrid.get_vertices_matrix(), femGrid.get_connectivity_matrix())
fem.solve(lambda x, y: 1, lambda x, y: 1)
solver = Solve(fem.get_matrix_a(), fem.get_vector_b())
solution = solver.parallel()
solution = fem.modify_solution(solution)
plotter = Plot(solution)
plotter.FEM(femGrid.get_vertices_matrix(), femGrid.get_connectivity_matrix())

# Example FDM solver and plotter:

# fdm = Fdm(51, 51, 0.02, 0.02)
# (X, Y) = fdm.grid()
# A = fdm.get_matrix_a(lambda x, y: 1, lambda x, y: 0, lambda x, y: 0)
# b = fdm.get_vector_b(lambda x, y: 1)
# solver = Solve(A, b)
# solution = solver.direct()
# solution = fdm.modify_solution(solution)
# plotter = Plot(solution)
# plotter.FDM(X, Y)
