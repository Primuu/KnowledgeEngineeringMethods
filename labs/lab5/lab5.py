import numpy as np


def invert_matrix(matrix):
    # Matrix shape X x Y
    x = matrix.shape[0]
    y = matrix.shape[1]
    if x != y:
        return "Matrix must be square"
    # Create identity matrix with the same dimensions as the main matrix
    identity_matrix = np.identity(matrix.shape[0])

    # Create copy of both matrices
    matrix_copy = np.copy(matrix).astype(float)
    identity_copy = np.copy(identity_matrix).astype(float)

    # Run a loop through the length of the matrix (rows)
    for i in range(x):
        # Calculate the diagonal values of the matrix
        diagonal_value = 1 / matrix_copy[i][i]

        # Run a loop through the length of the matrix (columns)
        for j in range(x):
            # Multiply successive row values by diagonal_value
            matrix_copy[i][j] = matrix_copy[i][j] * diagonal_value
            identity_copy[i][j] = identity_copy[i][j] * diagonal_value

        # Run a loop through the length of the matrix (rows)
        for k in range(x):
            # Skipping values placed on the diagonal
            if i != k:
                # Create variable that stores the values of the main matrix currently considered in the iteration
                coefficient = matrix_copy[k][i]

                # Run a loop through the length of the matrix (columns)
                for m in range(x):
                    # Calculate values in matrices in addition to the diagonal
                    matrix_copy[k][m] = matrix_copy[k][m] - matrix_copy[i][m] * coefficient
                    identity_copy[k][m] = identity_copy[k][m] - identity_copy[i][m] * coefficient

    # Show that the matrix_copy is an identity matrix now
    print("Matrix copy: \n", matrix_copy.astype(int))
    # Return the modified identity matrix (our inverted matrix)
    return identity_copy


x = np.array([[8, 1, 4, 2, 1],
              [8, 6, 4, 2, 1],
              [1, 2, 3, 4, 1],
              [8, 0, 6, 2, 5],
              [1, 9, 6, 4, 1]])

print("\nInversion of matrix using my function \n", invert_matrix(x))
print("\nInversion of matrix using numpy function \n", np.linalg.inv(x))
