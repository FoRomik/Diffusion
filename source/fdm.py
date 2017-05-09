import numpy as np

class Fdm(object):
    """docstring for FDM."""
    def __init__(self, m, n, hx, hy):
        self.m = m
        self.n = n
        self.hx = hx
        self.hy = hy

    def get_matrix_a(self, sigma, dx_sigma, dy_sigma):
        rank = (self.m-2)*(self.n-2)
        matrix_a = np.zeros((rank, rank))
        for i in range(rank):
            for j in range(rank):
                x = (i%(self.m-2)+1)*self.hx
                y = (i//(self.m-2)+1)*self.hy
                if j == i:
                    matrix_a[i][j] = 2*sigma(x,y) * (1/self.hx**2 + 1/self.hy**2)
                elif j == i+1 and j%(self.m-2)!=0:
                    matrix_a[i][j] = -sigma(x,y)/self.hx**2 - dx_sigma(x,y)/(2*self.hx)
                elif j == i-1 and i%(self.m-2)!=0:
                    matrix_a[i][j] = -sigma(x,y)/self.hx**2 + dx_sigma(x,y)/(2*self.hx)
                elif j == i+(self.m-2):
                    matrix_a[i][j] = -sigma(x,y)/self.hy**2 + dy_sigma(x,y)/(2*self.hy)
                elif j == i-(self.m-2):
                    matrix_a[i][j] = -sigma(x,y)/self.hy**2 + dy_sigma(x,y)/(2*self.hy)
        return matrix_a

    def grid(self):
        X = [i*self.hx for i in range(self.m)]
        Y = [i*self.hy for i in range(self.n)]
        return (X, Y)

    def get_vector_b(self, function):
        rank = (self.m-2)*(self.n-2)
        vector_b = np.zeros(rank)
        for i in range(rank):
            x = (i%(self.m-2)+1)*self.hx
            y = (i//(self.m-2)+1)*self.hy
            vector_b[i] = function(x,y)
        return vector_b

    def modify_solution(self, solution):
        new_solution = np.zeros((self.m, self.n))
        counter = 0
        for i in range(1, self.m-1):
            for j in range(1,self.n-1):
                new_solution[i][j] = solution[counter]
                counter = counter + 1
        return new_solution
