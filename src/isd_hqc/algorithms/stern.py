"""
Implementation of the Stern ISD algorithm.
"""
from itertools import combinations
from isd_hqc.linear_algebra import gf2_matrix_vector_mul
from isd_hqc.syndrome import verify_solution

def generate_weight_vectors(
    length: int,
    weight: int,
) -> list[list[int]]:
    """
    Generate all binary vectors of a given length and Hamming weight.

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


def find_syndrome_collisions(
    left_list: list[tuple[list[int], list[int]]],
    right_list: list[tuple[list[int], list[int]]],
    target_syndrome: list[int],
) -> list[tuple[list[int], list[int]]]:
    """
    Find pairs of partial errors whose syndrome contributions
    combine to the target syndrome over GF(2).
    """

    collisions: list[tuple[list[int], list[int]]] = []

    for left_syndrome, left_error in left_list:
        if len(left_syndrome) != len(target_syndrome):
            raise ValueError(
                "Left partial syndrome length must match target syndrome length."
            )

        for right_syndrome, right_error in right_list:
            if len(right_syndrome) != len(target_syndrome):
                raise ValueError(
                    "Right partial syndrome length must match target syndrome length."
                )

            combined_syndrome = [
                left_bit ^ right_bit
                for left_bit, right_bit in zip(
                    left_syndrome,
                    right_syndrome,
                )
            ]

            if combined_syndrome == target_syndrome:
                collisions.append(
                    (
                        left_error,
                        right_error,
                    )
                )

    return collisions



def reconstruct_candidate_error(
    left_positions: list[int],
    left_error: list[int],
    right_positions: list[int],
    right_error: list[int],
    length: int,
) -> list[int]:
    """
    Reconstruct a full candidate error vector from two partial errors.
    """

    if len(left_positions) != len(left_error):
        raise ValueError(
            "Left positions must match left partial error length."
        )

    if len(right_positions) != len(right_error):
        raise ValueError(
            "Right positions must match right partial error length."
        )

    candidate_error = [0] * length

    for position, value in zip(left_positions, left_error):
        if position < 0 or position >= length:
            raise IndexError(
                "Left position is outside the candidate error vector."
            )

        candidate_error[position] = value

    for position, value in zip(right_positions, right_error):
        if position < 0 or position >= length:
            raise IndexError(
                "Right position is outside the candidate error vector."
            )

        candidate_error[position] = value

    return candidate_error


def stern_decode(
    parity_check_matrix: list[list[int]],
    syndrome: list[int],
    left_positions: list[int],
    right_positions: list[int],
    left_weight: int,
    right_weight: int,
) -> list[int] | None:
    """
    Decode a syndrome using a simplified educational Stern procedure.

    """

    if not parity_check_matrix:
        return None

    left_list = build_partial_syndrome_list(
        parity_check_matrix=parity_check_matrix,
        positions=left_positions,
        weight=left_weight,
    )

    right_list = build_partial_syndrome_list(
        parity_check_matrix=parity_check_matrix,
        positions=right_positions,
        weight=right_weight,
    )

    collisions = find_syndrome_collisions(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=syndrome,
    )

    total_length = len(parity_check_matrix[0])

    for left_error, right_error in collisions:
        candidate_error = reconstruct_candidate_error(
            left_positions=left_positions,
            left_error=left_error,
            right_positions=right_positions,
            right_error=right_error,
            length=total_length,
        )

        if verify_solution(
            parity_check_matrix,
            candidate_error,
            syndrome,
        ):
            return candidate_error

    return None