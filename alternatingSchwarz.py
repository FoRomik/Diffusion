from argparse import ArgumentParser
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from mpi4py import MPI


from source.grid import Grid
from source.partition import Partition
from source.fem import Fem
#from source.solve import Solve (we're implementing solver!)
from source.plot import Plot

def Schwarz(vtk_filename, iterations):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(rank)

    if rank == 0:
        # obtain grid from vtk file
        grid = Grid(vtk_filename)
        vertices_matrix = grid.get_vertices_matrix()
        connectivity_matrix = grid.get_connectivity_matrix()

        # send vertices_matrix and connectivity matrix to process 1
        comm.send(vertices_matrix, dest=1, tag=1)
        comm.send(connectivity_matrix, dest=1, tag=2)

        # partition grid into two
        partitioner = Partition(vertices_matrix, connectivity_matrix)
        cell_tasks = np.array(partitioner.split_graph(2))

        # perform FEM on data
        fem = Fem(vertices_matrix, connectivity_matrix)
        fem.solve(lambda x, y: 1, lambda x, y: 1)
        A = fem.get_matrix_a()
        b = fem.get_vector_b()
        boundary = fem.get_boundary_array()
        cell_tasks = np.delete(cell_tasks, boundary)

        # send cell_tasks
        comm.send(cell_tasks, dest=1, tag=3)

        # Dimensions
        higher = len(cell_tasks)

        # Initial guess
        x = np.zeros(higher)

        # Compute restriction matrix
        identity = np.identity(higher)
        modified_cell_tasks = [True if task==0 else False for task in cell_tasks]
        R0 = identity[modified_cell_tasks]
        R0T = R0.transpose()

        # get subgrid stiffness matrix
        A0 = sp.csc_matrix(R0.dot(A).dot(R0T))

        # invert matrix
        inv0 = linalg.inv(A0)

        for i in range(iterations):
            comm.send(x, dest=1, tag=4)
            residual = b-A.dot(x)
            delta0 = R0T.dot(inv0.todense()).dot(R0).dot(residual)
            delta1 = comm.recv(source=1, tag=5)
            x = x + delta0 + delta1
            print(x)

        #x = fem.modify_solution(x)
        #plot = Plot(x)
        #plot.FEM(vertices_matrix, connectivity_matrix)


    elif rank == 1:
        # receive vertices matrix and connectivity matrix from process 0
        vertices_matrix = comm.recv(source=0, tag=1)
        connectivity_matrix = comm.recv(source=0, tag=2)

        # receive cell_tasks
        cell_tasks = comm.recv(source=0, tag=3)

        # perform FEM on data
        fem = Fem(vertices_matrix, connectivity_matrix)
        fem.solve(lambda x, y: 1, lambda x, y: 1)
        A = fem.get_matrix_a()
        b = fem.get_vector_b()

        # Dimensions
        higher = len(cell_tasks)

        # Compute restriction matrix
        identity = np.identity(higher)
        modified_cell_tasks = [True if task==1 else False for task in cell_tasks]
        R1 = identity[modified_cell_tasks]
        R1T = R1.transpose()

        # get subgrid stiffness matrix
        A1 = sp.csc_matrix(R1.dot(A).dot(R1T))

        # invert matrix
        inv1 = linalg.inv(A1)

        for i in range(iterations):
            x = comm.recv(source=0, tag=4)
            delta1 = R1T.dot(inv1.todense()).dot(R1).dot(b-A.dot(x))
            comm.send(delta1, dest=0, tag=5)

def intersection(vertices_matrix, connectivity0, connectivity1):
    intersect = []
    for i in range(len(vertices_matrix)):
        if i in connectivity0 and i in connectivity1:
            intersect.append(i)
    return intersect

parser = ArgumentParser(description = "Perform direct Schwarz method on vtk grid.")
parser.add_argument('--vtk', '-v', help='name of vtk file', type=str, dest='vtk')
parser.add_argument('--iterations', '-i', help='number of iterations', type=int, dest='iter')

arguments= parser.parse_args()
Schwarz(arguments.vtk, arguments.iter)
