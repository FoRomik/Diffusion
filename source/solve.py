import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from petsc4py import PETSc


class Solve(object):
    """docstring for Solve."""
    def __init__(self, matrix_a, vector_b):
        self.matrix_a = matrix_a
        self.vector_b = vector_b

    def default(self):
        solution = np.linalg.solve(self.matrix_a, self.vector_b)
        return solution.flatten()

    def direct(self):
        A = sparse.csr_matrix(self.matrix_a)
        b = self.vector_b
        solution = linalg.spsolve(A, b)
        return solution

    def iterative(self, subtype):
        assert subtype in ['cg', 'gmres', 'minres']
        A = sparse.csr_matrix(self.matrix_a)
        b = self.vector_b
        counter = method_counter()
        if subtype == 'gmres':
            solution = linalg.gmres(A, b, callback=counter)
        elif subtype == 'minres':
            solution = linalg.minres(A, b, callback=counter)
        elif subtype == 'cg':
            solution = linalg.cg(A, b, callback=counter)
        print(counter.iter)
        return solution

    def AMG(self):
        pass

    def parallel(self, subtype):
        pass


class method_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
        self.residuals = []
    def __call__(self, rk):
        self.niter += 1
        self.residuals.append(rk)
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
