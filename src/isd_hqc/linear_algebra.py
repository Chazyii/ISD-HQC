from typing import Iterable

Vector = list[int]


def gf2_add_vectors(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")

    return [(x + y) % 2 for x, y in zip(a, b)]


def hamming_weight(v: Iterable[int]) -> int:
    return sum(v)

Matrix = list[Vector]
def gf2_matrix_vector_mul(matrix: Matrix, vector: Vector) -> Vector:
     if not matrix:
        raise ValueError("Matrix must not be empty.")
    
     if len(matrix[0]) != len(vector):
        raise ValueError("Number of matrix columns must match vector length.")

     result = []

     for row in matrix:
        value = sum(x * y for x, y in zip(row, vector)) % 2
        result.append(value)

     return result


def transpose_matrix(matrix: Matrix) -> Matrix:
    if not matrix:
        raise ValueError("Matrix must not be empty.")

    number_of_columns = len(matrix[0])

    if any(len(row) != number_of_columns for row in matrix):
        raise ValueError("All matrix rows must have the same length.")

    transposed = []

    for column in range(number_of_columns):
        new_row = []

        for row in matrix:
            new_row.append(row[column])

        transposed.append(new_row)

    return transposed


def identity_matrix(size: int) -> Matrix:

    if size <= 0:
        raise ValueError("Size must be positive.")

    matrix = []

    for row in range(size):
        new_row = []

        for column in range(size):
            if row == column:
                new_row.append(1)
            else:
                new_row.append(0)

        matrix.append(new_row)

    return matrix
   