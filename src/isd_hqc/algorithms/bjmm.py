"""
Implementation of the BJMM ISD algorithm.
"""

from isd_hqc.linear_algebra import (
    gf2_add_vectors,
    hamming_weight,
)

from itertools import combinations
from isd_hqc.algorithms.stern import project_syndrome
from isd_hqc.linear_algebra import gf2_matrix_vector_mul

def is_valid_representation(
    target_vector: list[int],
    left_vector: list[int],
    right_vector: list[int],
    component_weight: int,
) -> bool:
    """
    Check whether two binary vectors form a valid BJMM representation
    of a target vector.

    """

    if len(left_vector) != len(target_vector):
        raise ValueError(
            "Left vector length must match target vector length."
        )

    if len(right_vector) != len(target_vector):
        raise ValueError(
            "Right vector length must match target vector length."
        )

    if component_weight < 0:
        raise ValueError(
            "Component weight must not be negative."
        )

    if hamming_weight(left_vector) != component_weight:
        return False

    if hamming_weight(right_vector) != component_weight:
        return False

    represented_vector = gf2_add_vectors(
        left_vector,
        right_vector,
    )

    return represented_vector == target_vector




def generate_representations(
    target_vector: list[int],
    component_weight: int,
) -> list[tuple[list[int], list[int]]]:
    """
    Generate all BJMM representations of a target binary vector.

    A representation consists of two binary vectors of equal
    Hamming weight such that.
    """

    length = len(target_vector)

    if component_weight < 0:
        raise ValueError(
            "Component weight must not be negative."
        )

    if component_weight > length:
        raise ValueError(
            "Component weight must not exceed vector length."
        )

    if any(value not in {0, 1} for value in target_vector):
        raise ValueError(
            "Target vector must contain only binary values."
        )

    representations: list[
        tuple[list[int], list[int]]
    ] = []

    for positions in combinations(
        range(length),
        component_weight,
    ):
        left_vector = [0] * length

        for position in positions:
            left_vector[position] = 1

        right_vector = gf2_add_vectors(
            left_vector,
            target_vector,
        )

        if hamming_weight(right_vector) != component_weight:
            continue

        representations.append(
            (
                left_vector,
                right_vector,
            )
        )

    return representations




def build_bjmm_syndrome_list(
    parity_check_matrix: list[list[int]],
    positions: list[int],
    vectors: list[list[int]],
    merge_rows: list[int],
) -> list[tuple[list[int], list[int]]]:
    """
    Build a BJMM list containing projected syndrome contributions.

    Each input vector is defined on the selected matrix positions.
    Its syndrome contribution is computed and projected onto merge_rows.

    """

    if not parity_check_matrix:
        raise ValueError(
            "Parity-check matrix must not be empty."
        )

    number_of_columns = len(parity_check_matrix[0])

    if any(
        len(row) != number_of_columns
        for row in parity_check_matrix
    ):
        raise ValueError(
            "All parity-check matrix rows must have the same length."
        )

    if any(
        position < 0 or position >= number_of_columns
        for position in positions
    ):
        raise IndexError(
            "Position is outside the matrix column range."
        )

    syndrome_list: list[
        tuple[list[int], list[int]]
    ] = []

    partial_matrix = [
        [row[position] for position in positions]
        for row in parity_check_matrix
    ]

    for vector in vectors:
        if len(vector) != len(positions):
            raise ValueError(
                "Vector length must match the number of positions."
            )

        full_syndrome = gf2_matrix_vector_mul(
            partial_matrix,
            vector,
        )

        projected_syndrome = project_syndrome(
            syndrome=full_syndrome,
            collision_rows=merge_rows,
        )

        syndrome_list.append(
            (
                projected_syndrome,
                vector,
            )
        )

    return syndrome_list