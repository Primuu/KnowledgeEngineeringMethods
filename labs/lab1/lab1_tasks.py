import numpy as np
import tensorflow as tf

# TASK 1
vector_t1 = np.array([1, 4, 5, 6, 2, 1, 5, 6, 7, 0])
print(vector_t1)

# a
a = np.delete(vector_t1, 5)
print(a)

# b
b = np.append(vector_t1, 8)
print(b)

# c
c = vector_t1.copy()
odd_indices = np.arange(1, len(vector_t1), 2).astype('i')
c[odd_indices] += 2
print(c)

# d
d = vector_t1[::-1]
print(d)

# TASK 2

vector_t2 = np.array([0, 1, 2, 3, 4, 5])
list_t2 = [0, 1, 2, 3, 4, 5]

scalar = 3

multiplying_vector_np = np.multiply(vector_t2, scalar)
multiplying_list = list_t2 * scalar

print("Multiplying numpy vector: ", multiplying_vector_np)
print("Multiplying vector: ", multiplying_list)

# Jak widać na przykładzie, mnożenie wektora stworzonego przy funkcji np.array
# zostało przeprowadzone poprawnie, natomiast mnożenie listy przez skalar
# zaskutkowało powieleniem (rówmemu skalarowi) listy

# TASK 3

matrix_t3 = np.array([3, 5, 0, 2, 6, 1, 3, 8, 9]).reshape((3, 3))
print(matrix_t3)

# a, b, c
matrix_t3[0][0] = -2
matrix_t3[1][1] = 44
matrix_t3[2][2] = 0
print(matrix_t3)

# TASK 4

for i in range(matrix_t3.shape[0]):
    for j in range(matrix_t3.shape[1]):
        if i % 2 == 0 and j % 2 == 0:
            matrix_t3[i, j] = 0

print(matrix_t3)

# TASK 5

arr1 = np.array([2, 1, 1, 1, 3, 6, 4, 5, 5]).reshape((3, 3))
arr2 = np.array([1, 0, 5, 2, 1, 6, 0, 3, 0]).reshape((3, 3))

arr_sum = np.zeros((3, 3)).astype('i')

for i in range(arr1.shape[0]):
    for j in range(arr2.shape[1]):
        arr_sum[i][j] = arr1[i][j] + arr2[i][j]

print(arr_sum)

# TASK 6

tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
print(tensor)

# Wyświetlony zostaje tensor wraz z jego wymiarami i typem danych.
# Tensor ma wymiary 4x4 (shape=(4, 4)) i zawiera liczby całkowite typu int32
# (dtype=int32 - typ danych całkowitych o rozmiarze 32 bitów (4 bajtów)),
# co można wyczytać w ostatniej linijce wyświetlonego wyniku.
