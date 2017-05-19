import time
from argparse import ArgumentParser
from source.grid import Grid
from source.fem import Fem
from source.solve import Solve
from source.plot import Plot

def test(vtk_file_name, subtype):
    assert subtype in ["direct", "gmres", "minres", "cg"]
    # Create grid
    grid = Grid(vtk_file_name)
    vertices_matrix = grid.get_vertices_matrix()
    connectivity_matrix = grid.get_connectivity_matrix()

    # Obtain stiffness/load
    fem = Fem(vertices_matrix, connectivity_matrix)
    fem.solve(lambda x, y: 1, lambda x, y: 1)
    A = fem.get_matrix_a()
    b = fem.get_vector_b()

    t = time.time()
    # Solve
    solver = Solve(A, b)
    if subtype=="direct":
        x = solver.direct()
    else:
        x = solver.iterative(subtype)
    x = fem.modify_solution(x)

    elapsed=time.time()-t
    print(elapsed)

    # Plot
    plotter = Plot(x)
    plotter.FEM(vertices_matrix, connectivity_matrix)


def parallelTest(vtk_file_name, pc, krylov):
    # Create grid
    grid = Grid(vtk_file_name)
    vertices_matrix = grid.get_vertices_matrix()
    connectivity_matrix = grid.get_connectivity_matrix()

    # Obtain stiffness/load
    fem = Fem(vertices_matrix, connectivity_matrix)
    fem.solve(lambda x, y: 1, lambda x, y: 1)
    A = fem.get_matrix_a()
    b = fem.get_vector_b()
    t = time.time()
    # Solve
    solver = Solve(A, b)
    x = solver.parallel(pc, krylov)
    x = fem.modify_solution(x)

    elapsed = time.time()-t
    print(elapsed)

    # Plot
    plotter = Plot(x)
    plotter.FEM(vertices_matrix, connectivity_matrix)


parser = ArgumentParser(description = "Perform FEM on vtk grid and mesaure time.")
parser.add_argument('--vtk', '-v', help='name of vtk file', type=str, dest='vtk')
parser.add_argument('--subtype', '-s', help='subtype of the direct/iterative method', type=str, dest='subtype')
parser.add_argument('--preconditioner', '-p', help='preconditioner for parallel method', type=str, dest='pc')
parser.add_argument('--krylov', '-k', help="Krylov solver for parallel method", type=str, dest='krylov')

arguments= parser.parse_args()
if arguments.subtype:
    test(arguments.vtk, arguments.subtype)
elif arguments.pc and arguments.krylov:
    parallelTest(arguments.vtk, arguments.pc, arguments.krylov)
