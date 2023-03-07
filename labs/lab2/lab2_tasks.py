import numpy as np


# TASK 1
# function that takes two matrices A and B as arguments
def multiply_matrices(a: np.array, b: np.array):
    # number of columns of matrix A
    col_a = a.shape[1]
    # number of columns of matrix B
    col_b = b.shape[1]
    # number of rows of matrix A
    rows_a = a.shape[0]
    # number of rows of matrix B
    rows_b = b.shape[0]
    # condition that checks if the number of columns is equal to the number of rows
    if col_a != rows_b:
        # if not, a message is returned
        return "Number of columns of A must be equal to the number of rows B"

    # if yes, matrix C is created (with the same number of rows as matrix A
    # and the same number of columns as matrix B)
    c = np.zeros(([rows_a, col_b]))
    # Iterating over all rows of matrix a
    for i in range(rows_a):
        # Iterating over all columns of matrix b
        for j in range(col_b):
            # Iterating over all columns of matrix a (or rows of matrix b, since col_a and rows_b are equal)
            for k in range(col_a):
                # For each iteration, c[i][j] is updated by adding the product of a[i][k] and b[k][j]
                c[i][j] += a[i][k] * b[k][j]
    # matrix c is returned as the result
    return c


A = np.array([[2, 1, 1],
              [1, 3, 6],
              [4, 5, 5]])

B = np.array([[1, 0, 5],
              [2, 1, 6],
              [0, 3, 0]])

print("Multiplying using np.matmul() \n", np.matmul(A, B))
print()
print("Multiplying using my function \n", multiply_matrices(A, B))


# TASK 2
# function that takes matrix 3x3 as argument
def det_3x3(matrix: np.array):
    # condition that checks if the shape is 3x3
    if matrix.shape != (3, 3):
        # if not, a message is returned
        return "Matrix must be 3x3"

    # if yes, variables are created to apply the Sarruss method
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    # Calculating and returning the determinant using the Sarrus method
    return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (a * f * h) - (b * d * i)


C = np.array([[1, 4, 5],
              [2, 1, 6],
              [0, 3, 2]])

print("\nCalculation of the determinant using np.linalg.det(): ", np.linalg.det(C))
print("Calculation of the determinant using my function:", det_3x3(C))


# TASK 3
# function that takes matrix as argument
def transpose(matrix: np.array):
    # number of rows of matrix
    rows = matrix.shape[0]
    # number of columns of matrix
    cols = matrix.shape[1]
    # creating a new resulting matrix with dimensions inverted from the input matrix
    transposed = np.zeros((cols, rows))
    # Iterating over all rows of matrix
    for i in range(rows):
        # Iterating over all columns of matrix
        for j in range(cols):
            # For each iteration, transposed[j][i] is updated by setting the value of matrix[i][j]
            transposed[j][i] = matrix[i][j]
    # matrix transposed is returned as the result
    return transposed


D = np.array([[3, 2, 4, 1, 8, 0],
              [2, 3, 5, 6, 0, 3],
              [7, 7, 2, 1, 4, 5]])

print("\nTransposition of matrix using np.transpose() \n", np.transpose(D))
print("\nTransposition of matrix using my function \n", transpose(D))
