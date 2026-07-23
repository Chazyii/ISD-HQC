"""
Implementation of the Stern ISD algorithm.
"""
from itertools import combinations
from isd_hqc.linear_algebra import gf2_matrix_vector_mul

def generate_weight_vectors(
    length: int,
    weight: int,
) -> list[list[int]]:
    """
    Generate all binary vectors of a given length and Hamming weight.

    Args:
        length:
            Length of each generated binary vector.
        weight:
            Required Hamming weight of each vector.

    Returns:
        A list containing all binary vectors of the specified length
        and Hamming weight.

    Raises:
        ValueError:
            If length is negative, weight is negative, or weight
            is greater than length.
    """

    if length < 0:
        raise ValueError("Length must be non-negative.")

    if weight < 0:
        raise ValueError("Weight must be non-negative.")

    if weight > length:
        raise ValueError("Weight cannot be greater than vector length.")

    vectors: list[list[int]] = []

    for positions in combinations(range(length), weight):
        vector = [0] * length

        for position in positions:
            vector[position] = 1

        vectors.append(vector)

    return vectors


def compute_partial_syndrome(
    parity_check_matrix: list[list[int]],
    positions: list[int],
    partial_error: list[int],
) -> list[int]:
    """
    Compute the syndrome contribution of a partial error vector.
    The partial error is defined only on the selected column positions
    of the parity-check matrix.

    """

    if len(positions) != len(partial_error):
        raise ValueError(
            "Number of positions must match partial error length."
        )

    if not parity_check_matrix:
        return []

    number_of_columns = len(parity_check_matrix[0])

    for position in positions:
        if position < 0 or position >= number_of_columns:
            raise IndexError(
                "Partial error position is outside the matrix range."
            )

    partial_matrix = [
        [row[position] for position in positions]
        for row in parity_check_matrix
    ]

    return gf2_matrix_vector_mul(
        partial_matrix,
        partial_error,
    )



def build_partial_syndrome_list(
    parity_check_matrix: list[list[int]],
    positions: list[int],
    weight: int,
) -> list[tuple[list[int], list[int]]]:
    """
    Build a list of partial syndromes and their corresponding
    fixed-weight partial error vectors.
    """

    partial_errors = generate_weight_vectors(
        length=len(positions),
        weight=weight,
    )

    syndrome_list: list[tuple[list[int], list[int]]] = []

    for partial_error in partial_errors:
        partial_syndrome = compute_partial_syndrome(
            parity_check_matrix=parity_check_matrix,
            positions=positions,
            partial_error=partial_error,
        )

        syndrome_list.append(
            (
                partial_syndrome,
                partial_error,
            )
        )

    return syndrome_list