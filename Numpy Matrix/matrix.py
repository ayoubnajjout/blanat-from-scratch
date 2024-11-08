import numpy as np

class Matrix:
    def __init__(self,n,m):
        self.n = n
        self.m = m
        self.M = np.ones((n, m), dtype=int)
        
    def show_matrix(self):
        print(self.M)

    # Exercice 1

    def identity_matrix(self):
        identity = np.empty((self.n,self.n),dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    identity[i,j] = 1
                else:
                    identity[i,j] = 0
        return identity
    
    def diagonal_matrix(self):
        diagonal = np.zeros((self.n, self.n), dtype=int) 
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    diagonal[i, j] = self.M[i, j]
        return diagonal
    
    def triangular_supp_matrix(self):
        triangular = np.empty((self.n,self.n),dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                if i <= j:
                    triangular[i,j] = self.M[i,j]
                else:    
                    triangular[i,j] = 0
        return triangular
    
    def triangular_inf_matrix(self):
        triangular = np.empty((self.n,self.n),dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                if i >= j:
                    triangular[i,j] = self.M[i,j]
                else:    
                    triangular[i,j] = 0
        return triangular
    
    # Exercice 2

    def transpose_matrix(self):
        transpose = np.empty((self.m,self.n),dtype=int)
        for i in range(self.n):
            for j in range(self.m):
                transpose[j,i] = self.M[i,j]
        return transpose

    # Exercice 3

    def sum_matrix(self):
        sum = np.empty((self.n),dtype=int)
        temp = 0
        for i in range(self.n):
            for j in range(self.m):
                temp += self.M[i,j]
            sum[i] = temp
            temp = 0
        return sum
    
    def multip_matrix(self):
        sum = np.empty((self.m),dtype=int)
        temp = 1
        for i in range(self.m):
            for j in range(self.n):
                temp *= self.M[j,i]
            sum[i] = temp
            temp = 1
        return sum
    
    def max_rows(self):
        max_vals = np.empty(self.n, dtype=int)
        for i in range(self.n):
            temp = self.M[i, 0]  
            for j in range(1, self.m):  
                if self.M[i, j] > temp:
                    temp = self.M[i, j]
            max_vals[i] = temp
        return max_vals

    def max_col(self):
        max_vals = np.empty(self.m, dtype=int)
        for i in range(self.m):
            temp = self.M[0, i]  
            for j in range(1, self.n):
                if self.M[j, i] > temp:
                    temp = self.M[j, i]
            max_vals[i] = temp
        return max_vals
    
    def min_rows(self):
        min_vals = np.empty(self.n, dtype=int)
        for i in range(self.n):
            temp = self.M[i, 0]  
            for j in range(1, self.m): 
                if self.M[i, j] < temp:
                    temp = self.M[i, j]
            min_vals[i] = temp
        return min_vals

    def min_col(self):
        min_vals = np.empty(self.m, dtype=int)
        for i in range(self.m):
            temp = self.M[0, i]
            for j in range(1, self.n):  
                if self.M[j, i] < temp:
                    temp = self.M[j, i]
            min_vals[i] = temp
        return min_vals
    
    # Exercice 4

    def mean_matrix(self):
        mean = 0
        for i in range(self.n):
            for j in range(self.m):
                mean += self.M[i,j]
        return mean/(self.n*self.m)
    
    def var_matrix(self):
        mean = self.mean_matrix()
        var = 0
        for i in range(self.n):
            for j in range(self.m):
                var += (self.M[i,j] - mean)**2
        return var/(self.n*self.m)

    def std_matrix(self):
        return np.sqrt(self.var_matrix())
    
    # Exercice 5

    def dot_matrix(self, matrix):
        dot = np.empty((self.n, matrix.m), dtype=int)
        for i in range(self.n):
            for j in range(matrix.m):
                dot[i, j] = 0
                for k in range(self.m): 
                    dot[i, j] += self.M[i, k] * matrix.M[k, j]
        return dot

    # Exercice 6

    def det_matrix(self):
        if self.n != self.m:
            raise ValueError("Determinant is only defined for square matrices.")
        
        def recursive_determinant(matrix):
            if matrix.shape == (2, 2):
                return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
            
            det = 0
            for col in range(matrix.shape[1]):
                sub_matrix = np.delete(np.delete(matrix, 0, axis=0), col, axis=1)
                cofactor = ((-1) ** col) * matrix[0, col] * recursive_determinant(sub_matrix)
                det += cofactor
            return det
        
        return recursive_determinant(self.M)