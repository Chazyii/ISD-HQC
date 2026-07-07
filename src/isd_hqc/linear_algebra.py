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



def gf2_matrix_matrix_mul(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    
    if not matrix_a or not matrix_b:
        raise ValueError("Matrices must not be empty.")

    columns_a = len(matrix_a[0])

    if any(len(row) != columns_a for row in matrix_a):
        raise ValueError("All rows in the first matrix must have the same length.")

    columns_b = len(matrix_b[0])

    if any(len(row) != columns_b for row in matrix_b):
        raise ValueError("All rows in the second matrix must have the same length.")

    if columns_a != len(matrix_b):
        raise ValueError("Matrix dimensions are incompatible for multiplication.")

    result = []

    for row in matrix_a:
        result_row = []

        for column in transpose_matrix(matrix_b):
            value = 0

            for a, b in zip(row, column):
                value ^= a & b

            result_row.append(value)

        result.append(result_row)

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


def gf2_row_echelon_form(matrix: Matrix) -> Matrix:

    if not matrix:
        raise ValueError("Matrix must not be empty.")

    number_of_columns = len(matrix[0])

    if any(len(row) != number_of_columns for row in matrix):
        raise ValueError("All matrix rows must have the same length.")

    result = [row.copy() for row in matrix]

    pivot_row = 0

    for column in range(number_of_columns):
        pivot = None

        for row in range(pivot_row, len(result)):
            if result[row][column] == 1:
                pivot = row
                break

        if pivot is None:
            continue

        result[pivot_row], result[pivot] = result[pivot], result[pivot_row]

        for row in range(pivot_row + 1, len(result)):
            if result[row][column] == 1:
                result[row] = gf2_add_vectors(result[row], result[pivot_row])

        pivot_row += 1

        if pivot_row == len(result):
            break

    return result


def gf2_rank(matrix: Matrix) -> int:
    
    echelon_form = gf2_row_echelon_form(matrix)

    rank = 0

    for row in echelon_form:
        if any(value == 1 for value in row):
            rank += 1

    return rank