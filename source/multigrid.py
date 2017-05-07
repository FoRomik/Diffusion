import numpy as np

class Multigrid(object):
    """docstring for Grid."""
    def __init__(self, matrix_a, vector_b):
        self.matrix_a = matrix_a
        self.vector_b = vector_b
