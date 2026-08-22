"""
Implementation of the BJMM ISD algorithm.
"""

from isd_hqc.linear_algebra import (
    gf2_add_vectors,
    hamming_weight,
)


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