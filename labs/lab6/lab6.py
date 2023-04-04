import numpy as np


def matrix_rank(matrix: np.array):
    # Loop applied to check every possible sub-matrix size (but also the original matrix),
    # e.g. for 4x4 it will be 4x4 first, then 3x3 etc.
    for k in range(matrix.shape[0], 0, -1):
        # range in parentheses means, the loop goes from 0 to number of rows of sub-matrix
        for i in range(matrix.shape[0] - k + 1):
            # range in parentheses means, the loop goes from 0 to number of columns of sub-matrix
            for j in range(matrix.shape[1] - k + 1):
                # In each iteration new sub-matrix is created
                # by selecting rows and columns from the original matrix

                # Selecting rows with indexes i to i+k-1
                sub_matrix = matrix[i:i+k, :]
                # Selecting columns with indexes j to j+k-1
                sub_matrix = sub_matrix[:, j:j+k]

                # Calculating the determinant of a sub-matrix
                sub_det = np.linalg.det(sub_matrix).round(10)
                # Rounded applied to get rid of near-zero values

                # If result is number other than zero,
                # rank of a matrix is number of matrix rows
                if sub_det != 0:
                    # sub_matrix.shape[0] is the number of rows in the matrix
                    return sub_matrix.shape[0]
                # If result equals zero,
                # all sub-matrices are selected one by one (by crossing out next columns and rows),
                # e.g. for 4x4 starting with 3x3 (then 2x2 if loop inside loop ends, and if so again - 1x1)
                # until sub-matrix with non-zero determinant will be found

    # If none of the sub-matrices has a determinant other than zero,
    # then the value 0 is returned, which means that the rank of the matrix is 0
    return 0


a = np.array([[1, 1, 5],
              [2, 0, 6],
              [8, 3, 2]])

b = np.array([[3, -1, 1],
              [5, 1, 4],
              [-1, 3, 2]])

c = np.array([[1, 3, -2, 4],
              [1, -1, 3, 5],
              [0, 1, 4, -2],
              [10, -2, 5, 1]])

d = np.array([[2, 8, 3, -4],
              [1, 4, 1, -2],
              [5, 20, 0, -10],
              [-3, -12, -2, 6]])

print("\nRank of matrix A using my function: ", matrix_rank(a))
print("Rank of matrix A using numpy function: ", np.linalg.matrix_rank(a))

print("\nRank of matrix B using my function: ", matrix_rank(b))
print("Rank of matrix B using numpy function: ", np.linalg.matrix_rank(b))

print("\nRank of matrix C using my function: ", matrix_rank(c))
print("Rank of matrix C using numpy function: ", np.linalg.matrix_rank(c))

print("\nRank of matrix D using my function: ", matrix_rank(d))
print("Rank of matrix D using numpy function: ", np.linalg.matrix_rank(d))
