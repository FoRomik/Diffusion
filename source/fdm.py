import numpy as np

class FDM(object):
    """docstring for FDM."""
    def __init__(self, m, n, hx, hy):
        self.m = m
        self.n = n
        self.hx = hx
        self.hy = hy

    def compute_a(self, sigma, dx_sigma, dy_sigma):
        rank = (self.m-2)*(self.n-2)
        matrix_a = np.zeros((rank, rank))
        for i in range(rank):
            for j in range(rank):
                x = (i%(self.m-2)+1)*self.hx
                y = (i//(self.m-2)+1)*self.hy
                if j == i:
                    matrix_a[i][j] = -2*sigma(x,y) * (1/self.hx**2 + 1/self.hy**2)
                elif j == i+1:
                    matrix_a[i][j] = sigma(x,y)/self.hx**2 + dx_sigma(x,y)/(2*self.hx)
                elif j == i-1:
                    matrix_a[i][j] = sigma(x,y)/self.hx**2 - dx_sigma(x,y)/(2*self.hx)
                elif j == i+self.m-2:
                    matrix_a[i][j] = sigma(x,y)/self.hy**2 - dy_sigma(x,y)/(2*self.hy)
                elif j == i-self.m-2:
                    matrix_a[i][j] = sigma(x,y)/self.hy**2 - dy_sigma(x,y)/(2*self.hy)
        return matrix_a

    def compute_b(self, f):
        rank = (self.m-2)*(self.n-2)
        vector_b = np.zeros(rank)
        for i in range(rank):
            x = (i%(self.m-2)+1)*self.hx
            y = (i//(self.m-2)+1)*self.hy
            vector_b[i] = f(x,y)
        return vector_b
