"""
Implementation of the BJMM ISD algorithm.
"""

from isd_hqc.linear_algebra import (
    gf2_add_vectors,
    hamming_weight,
    gf2_matrix_vector_mul,
)



from isd_hqc.algorithms.stern import (
    generate_weight_vectors,
    project_syndrome,
)

from itertools import combinations
from isd_hqc.syndrome import verify_solution

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




def build_bjmm_base_list(
    parity_check_matrix: list[list[int]],
    positions: list[int],
    component_weight: int,
    merge_rows: list[int],
) -> list[tuple[list[int], list[int]]]:
    """
    Build one base list for the BJMM merge tree.

    """

    if component_weight < 0:
        raise ValueError(
            "Component weight must not be negative."
        )

    if component_weight > len(positions):
        raise ValueError(
            "Component weight must not exceed the number of positions."
        )

    vectors = generate_weight_vectors(
        length=len(positions),
        weight=component_weight,
    )

    return build_bjmm_syndrome_list(
        parity_check_matrix=parity_check_matrix,
        positions=positions,
        vectors=vectors,
        merge_rows=merge_rows,
    )




def split_bjmm_information_positions(
    information_positions: list[int],
) -> tuple[list[int], list[int]]:
    """
    Split BJMM information positions into two equal halves.

    """

    if not information_positions:
        raise ValueError(
            "Information positions must not be empty."
        )

    if len(set(information_positions)) != len(
        information_positions
    ):
        raise ValueError(
            "Information positions must not contain duplicates."
        )

    if len(information_positions) % 2 != 0:
        raise ValueError(
            "Number of information positions must be even."
        )

    middle = len(information_positions) // 2

    first_half = information_positions[:middle]
    second_half = information_positions[middle:]

    return first_half, second_half




def merge_bjmm_lists(
    left_list: list[tuple[list[int], list[int]]],
    right_list: list[tuple[list[int], list[int]]],
    target_syndrome: list[int],
    target_weight: int | None = None,
) -> list[list[int]]:
    """
    Merge two BJMM lists using a projected syndrome condition.

    If target_weight is provided, only merged vectors with exactly
    that Hamming weight are retained.
    """

    if target_weight is not None and target_weight < 0:
        raise ValueError(
            "Target weight must not be negative."
        )

    merged_vectors: list[list[int]] = []

    target_length = len(target_syndrome)

    for left_syndrome, left_vector in left_list:
        if len(left_syndrome) != target_length:
            raise ValueError(
                "Left projected syndrome length must match target syndrome length."
            )

        for right_syndrome, right_vector in right_list:
            if len(right_syndrome) != target_length:
                raise ValueError(
                    "Right projected syndrome length must match target syndrome length."
                )

            if len(left_vector) != len(right_vector):
                raise ValueError(
                    "Vectors selected for merging must have the same length."
                )

            combined_syndrome = gf2_add_vectors(
                left_syndrome,
                right_syndrome,
            )

            if combined_syndrome != target_syndrome:
                continue

            merged_vector = gf2_add_vectors(
                left_vector,
                right_vector,
            )

            if (
                target_weight is not None
                and hamming_weight(merged_vector) != target_weight
            ):
                continue

            merged_vectors.append(
                merged_vector
            )

    return merged_vectors



def merge_bjmm_level(
    parity_check_matrix: list[list[int]],
    positions: list[int],
    left_list: list[tuple[list[int], list[int]]],
    right_list: list[tuple[list[int], list[int]]],
    merge_target: list[int],
    next_merge_rows: list[int],
    merged_weight: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """
    Perform one BJMM list-merging level.

    Optionally retain only merged vectors of a specified Hamming weight.
    """

    merged_vectors = merge_bjmm_lists(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=merge_target,
        target_weight=merged_weight,
    )

    if not merged_vectors:
        return []

    return build_bjmm_syndrome_list(
        parity_check_matrix=parity_check_matrix,
        positions=positions,
        vectors=merged_vectors,
        merge_rows=next_merge_rows,
    )




def merge_bjmm_tree(
    parity_check_matrix: list[list[int]],
    positions: list[int],
    first_left_list: list[tuple[list[int], list[int]]],
    first_right_list: list[tuple[list[int], list[int]]],
    second_left_list: list[tuple[list[int], list[int]]],
    second_right_list: list[tuple[list[int], list[int]]],
    first_merge_target: list[int],
    second_merge_target: list[int],
    next_merge_rows: list[int],
    final_merge_target: list[int],
) -> list[list[int]]:
    """
    Perform a two-level BJMM merge tree.

    """

    first_intermediate = merge_bjmm_level(
        parity_check_matrix=parity_check_matrix,
        positions=positions,
        left_list=first_left_list,
        right_list=first_right_list,
        merge_target=first_merge_target,
        next_merge_rows=next_merge_rows,
    )

    if not first_intermediate:
        return []

    second_intermediate = merge_bjmm_level(
        parity_check_matrix=parity_check_matrix,
        positions=positions,
        left_list=second_left_list,
        right_list=second_right_list,
        merge_target=second_merge_target,
        next_merge_rows=next_merge_rows,
    )

    if not second_intermediate:
        return []

    return merge_bjmm_lists(
        left_list=first_intermediate,
        right_list=second_intermediate,
        target_syndrome=final_merge_target,
    )






def reconstruct_bjmm_candidate(
    systematic_matrix: list[list[int]],
    transformed_syndrome: list[int],
    pivot_positions: list[int],
    information_positions: list[int],
    information_error: list[int],
) -> list[int]:
    """
    Reconstruct a complete error vector from a BJMM information-set error.

    The parity-check matrix is assumed to be in systematic form with
    identity columns at pivot_positions.

    """

    if not systematic_matrix:
        raise ValueError(
            "Systematic matrix must not be empty."
        )

    number_of_rows = len(systematic_matrix)
    number_of_columns = len(systematic_matrix[0])

    if any(
        len(row) != number_of_columns
        for row in systematic_matrix
    ):
        raise ValueError(
            "All systematic matrix rows must have the same length."
        )

    if len(transformed_syndrome) != number_of_rows:
        raise ValueError(
            "Transformed syndrome length must match matrix rows."
        )

    if len(pivot_positions) != number_of_rows:
        raise ValueError(
            "Number of pivot positions must match matrix rows."
        )

    if len(information_positions) != len(information_error):
        raise ValueError(
            "Information positions must match information error length."
        )

    if len(set(pivot_positions)) != len(pivot_positions):
        raise ValueError(
            "Pivot positions must not contain duplicates."
        )

    if len(set(information_positions)) != len(information_positions):
        raise ValueError(
            "Information positions must not contain duplicates."
        )

    if set(pivot_positions) & set(information_positions):
        raise ValueError(
            "Pivot and information positions must be disjoint."
        )

    all_positions = (
        pivot_positions
        + information_positions
    )

    if any(
        position < 0 or position >= number_of_columns
        for position in all_positions
    ):
        raise IndexError(
            "Error position is outside the matrix column range."
        )

    full_information_error = [
        0
    ] * number_of_columns

    for position, value in zip(
        information_positions,
        information_error,
    ):
        full_information_error[position] = value

    information_syndrome = gf2_matrix_vector_mul(
        systematic_matrix,
        full_information_error,
    )

    pivot_error = [
        syndrome_bit ^ contribution_bit
        for syndrome_bit, contribution_bit in zip(
            transformed_syndrome,
            information_syndrome,
        )
    ]

    candidate_error = full_information_error.copy()

    for position, value in zip(
        pivot_positions,
        pivot_error,
    ):
        candidate_error[position] = value

    return candidate_error






def find_valid_bjmm_candidate(
    systematic_matrix: list[list[int]],
    transformed_syndrome: list[int],
    pivot_positions: list[int],
    information_positions: list[int],
    information_candidates: list[list[int]],
    target_weight: int,
) -> list[int] | None:
    """
    Find a valid complete error vector among BJMM information-set candidates.

    Each candidate in information_candidates represents an error vector
    defined only on information_positions.

    """

    if not systematic_matrix:
        raise ValueError(
            "Systematic matrix must not be empty."
        )

    number_of_columns = len(systematic_matrix[0])

    if target_weight < 0:
        raise ValueError(
            "Target weight must not be negative."
        )

    if target_weight > number_of_columns:
        raise ValueError(
            "Target weight must not exceed the code length."
        )

    for information_error in information_candidates:
        candidate_error = reconstruct_bjmm_candidate(
            systematic_matrix=systematic_matrix,
            transformed_syndrome=transformed_syndrome,
            pivot_positions=pivot_positions,
            information_positions=information_positions,
            information_error=information_error,
        )

        if hamming_weight(candidate_error) != target_weight:
            continue

        if verify_solution(
            parity_check_matrix=systematic_matrix,
            error=candidate_error,
            syndrome=transformed_syndrome,
            weight=target_weight,
        ):
            return candidate_error

    return None