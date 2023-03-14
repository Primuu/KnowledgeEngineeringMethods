import numpy as np

# Functions are based on lab2_tasks.py file


def transpose(matrix: np.array):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    transposed = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed


def det_2x2(matrix: np.array):
    if matrix.shape != (2, 2):
        return "Matrix must be 2x2"
    a, b = matrix[0]
    c, d = matrix[1]
    return (a * d) - (b * c)


def det_3x3(matrix: np.array):
    if matrix.shape != (3, 3):
        return "Matrix must be 3x3"
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (a * f * h) - (b * d * i)


# TASK 1


def inverse_2x2(matrix: np.array):
    if matrix.shape != (2, 2):
        return "Matrix must be 2x2"

    det = det_2x2(matrix)
    if det == 0:
        return "Determinant = 0 - matrix cannot be inverted"

    dim = matrix.shape[0]
    complements = np.zeros((2, 2))
    for i in range(dim):
        for j in range(dim):
            complements[i][j] = (-1) ** ((i + 1) + (j + 1)) * matrix[dim - 1 - i][dim - 1 - j]

    transposed = transpose(complements)
    inverted = np.zeros((2, 2))
    for i in range(dim):
        for j in range(dim):
            inverted[i][j] = transposed[i][j] / det

    return inverted


A1 = np.array([[3, 6],
               [0, 9]])

B1 = np.array([[-3, -9],
               [1, 3]])

print("\nInversion of matrix A1 using my function \n", inverse_2x2(A1))
print("\nInversion of matrix A1 using np.linalg.inv() \n", np.linalg.inv(A1))

print("\nInversion of matrix B1 using my function \n", inverse_2x2(B1))
# print("\nInversion of matrix B1 using np.linalg.inv() \n", np.linalg.inv(B1))  # Det is 0


# TASK 2


def inverse_3x3(a: np.array):
    if a.shape != (3, 3):
        return "Matrix must be 3x3"

    det = det_3x3(a)
    if det == 0:
        return "Determinant = 0 - matrix cannot be inverted"

    inverted = np.array([
        [a[1][1] * a[2][2] - a[1][2] * a[2][1], a[0][2] * a[2][1] - a[0][1] * a[2][2], a[0][1] * a[1][2] - a[0][2] * a[1][1]],
        [a[1][2] * a[2][0] - a[1][0] * a[2][2], a[0][0] * a[2][2] - a[0][2] * a[2][0], a[0][2] * a[1][0] - a[0][0] * a[1][2]],
        [a[1][0] * a[2][1] - a[1][1] * a[2][0], a[0][1] * a[2][0] - a[0][0] * a[2][1], a[0][0] * a[1][1] - a[0][1] * a[1][0]]
    ])

    return inverted / det


A2 = np.array([[1, 0, 5],
               [2, 7, 6],
               [8, 3, 2]])

B2 = np.array([[3, -1, 1],
               [5, 1, 4],
               [-1, 3, 2]])

print("\nInversion of matrix A2 using my function \n", inverse_3x3(A2))
print("\nInversion of matrix A2 using np.linalg.inv() \n", np.linalg.inv(A2))

print("\nInversion of matrix B2 using my function \n", inverse_3x3(B2))
# print("\nInversion of matrix B2 using np.linalg.inv() \n", np.linalg.inv(B2))  # Det is 0


# TASK 3


def inverse(matrix: np.array):
    if matrix.shape == (2, 2):
        return inverse_2x2(matrix)
    if matrix.shape == (3, 3):
        return inverse_3x3(matrix)
    return "Matrix must be 2x2 or 3x3"


print("\n\n\n\nFinal check:")
print("\nInversion of matrix A1 using my function \n", inverse(A1))
print("\nInversion of matrix A1 using np.linalg.inv() \n", np.linalg.inv(A1))

print("\nInversion of matrix B1 using my function \n", inverse(B1))
# print("\nInversion of matrix B1 using np.linalg.inv() \n", np.linalg.inv(B1))  # Det is 0

print("\nInversion of matrix A2 using my function \n", inverse(A2))
print("\nInversion of matrix A2 using np.linalg.inv() \n", np.linalg.inv(A2))

print("\nInversion of matrix B2 using my function \n", inverse(B2))
# print("\nInversion of matrix B2 using np.linalg.inv() \n", np.linalg.inv(B2))  # Det is 0
