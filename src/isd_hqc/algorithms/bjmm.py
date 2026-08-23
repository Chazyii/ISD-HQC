"""
Implementation of the BJMM ISD algorithm.
"""

from isd_hqc.linear_algebra import (
    gf2_add_vectors,
    hamming_weight,
)

from itertools import combinations

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