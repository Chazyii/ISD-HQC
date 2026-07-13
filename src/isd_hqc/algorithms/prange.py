"""
Implementation of the Prange ISD algorithm.
"""

import random
from isd_hqc.linear_algebra import (
    Matrix,
    Vector,
    gf2_solve_linear_system,
)


def select_information_set(length: int, dimension: int) -> list[int]:

    if length <= 0:
        raise ValueError("Length must be positive.")

    if dimension <= 0:
        raise ValueError("Dimension must be positive.")

    if dimension > length:
        raise ValueError("Dimension cannot exceed code length.")

    return sorted(random.sample(range(length), dimension))



def construct_induced_system(
    parity_check_matrix: Matrix,
    syndrome: Vector,
    information_set: list[int],
) -> tuple[Matrix, Vector, list[int]]:

    if not parity_check_matrix:
        raise ValueError("Parity-check matrix must not be empty.")

    number_of_rows = len(parity_check_matrix)
    number_of_columns = len(parity_check_matrix[0])

    if any(len(row) != number_of_columns for row in parity_check_matrix):
        raise ValueError("All matrix rows must have the same length.")

    if len(syndrome) != number_of_rows:
        raise ValueError("Syndrome length must match the number of matrix rows.")

    if len(information_set) != number_of_columns - number_of_rows:
        raise ValueError("Information set has an invalid size.")

    if len(information_set) != len(set(information_set)):
        raise ValueError("Information set indices must be unique.")

    if any(index < 0 or index >= number_of_columns for index in information_set):
        raise ValueError("Information set index is out of range.")

    information_set_values = set(information_set)

    complement = [
        index
        for index in range(number_of_columns)
        if index not in information_set_values
    ]

    induced_matrix = [
        [row[index] for index in complement]
        for row in parity_check_matrix
    ]

    return induced_matrix, syndrome.copy(), complement



def solve_induced_system(
    induced_matrix: Matrix,
    syndrome: Vector,
) -> Vector | None:

    return gf2_solve_linear_system(
        induced_matrix,
        syndrome,
    )


def reconstruct_candidate_error(
    length: int,
    partial_error: Vector,
    complement: list[int],
) -> Vector:

    if length <= 0:
        raise ValueError("Length must be positive.")

    if len(partial_error) != len(complement):
        raise ValueError(
            "Partial error length must match the complement size."
        )

    if len(complement) != len(set(complement)):
        raise ValueError("Complement indices must be unique.")

    if any(index < 0 or index >= length for index in complement):
        raise ValueError("Complement index is out of range.")

    candidate_error = [0] * length

    for position in range(len(complement)):
        index = complement[position]
        candidate_error[index] = partial_error[position]

    return candidate_error