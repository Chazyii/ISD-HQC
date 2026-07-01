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

   