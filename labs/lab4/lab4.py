import numpy as np

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


# Cramer's rule
# 3x + 5y = -7
#  x + 4y = -14

# 1. Matrix
a = np.array([[3, 5],
              [1, 4]])

# 2. Determinant
det = det_2x2(a)
print("Det =", det)

# 3. Substitution under x column and det calc
b = np.array([[-7, 5],
              [-14, 4]])

det_x = det_2x2(b)
print("Det x =", det_x)

# 3. Substitution under y column and det calc
c = np.array([[3, -7],
              [1, -14]])

det_y = det_2x2(c)
print("Det y =", det_y)

x = det_x / det
y = det_y / det
print("x = " + str(x) + ", y = " + str(y))


# TASK 1
def equations_solver(matrix: np.array):
    if matrix.shape == (2, 3):
        first_column = matrix[:, 0]
        second_column = matrix[:, 1]
        third_column = matrix[:, -1]

        det_matrix = np.column_stack((first_column, second_column))
        det = det_2x2(det_matrix)

        matrix_x = np.column_stack((third_column, second_column))
        det_x = det_2x2(matrix_x)

        matrix_y = np.column_stack((first_column, third_column))
        det_y = det_2x2(matrix_y)

        x = det_x / det
        y = det_y / det
        return [x, y]

    if matrix.shape == (3, 4):
        first_column = matrix[:, 0]
        second_column = matrix[:, 1]
        third_column = matrix[:, 2]
        fourth_column = matrix[:, -1]

        det_matrix = np.column_stack((first_column, second_column, third_column))
        det = det_3x3(det_matrix)

        matrix_x = np.column_stack((fourth_column, second_column, third_column))
        det_x = det_3x3(matrix_x)

        matrix_y = np.column_stack((first_column, fourth_column, third_column))
        det_y = det_3x3(matrix_y)

        matrix_z = np.column_stack((first_column, second_column, fourth_column))
        det_z = det_3x3(matrix_z)

        x = det_x / det
        y = det_y / det
        z = det_z / det
        return [x, y, z]


# 3x + 5y = -7
#  x + 4y = -14
a = np.array([[3, 5, -7],
              [1, 4, -14]])
print("\nx and y have values:", equations_solver(a))

#   x + 2y + 3z = -5
#  3x +  y - 3z =  4
# -3x + 4y + 7z = -7
b = np.array([[1, 2, 3, -5],
              [3, 1, -3, 4],
              [-3, 4, 7, -7]])
print("\nx, y and z have values:", equations_solver(b))
