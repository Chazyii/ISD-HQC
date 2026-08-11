"""
Implementation of the Stern ISD algorithm.
"""
from itertools import combinations
from isd_hqc.linear_algebra import (
    gf2_inverse_matrix,
    gf2_matrix_matrix_mul,
    gf2_matrix_vector_mul,
)
from isd_hqc.syndrome import verify_solution
import random

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



def validate_stern_positions(
    left_positions: list[int],
    right_positions: list[int],
    number_of_columns: int,
) -> None:
    """
    Validate the column positions used by the simplified Stern decoder.

    """

    if number_of_columns < 0:
        raise ValueError(
            "Number of columns must be non-negative."
        )

    if len(set(left_positions)) != len(left_positions):
        raise ValueError(
            "Left positions must not contain duplicates."
        )

    if len(set(right_positions)) != len(right_positions):
        raise ValueError(
            "Right positions must not contain duplicates."
        )

    if set(left_positions) & set(right_positions):
        raise ValueError(
            "Left and right positions must be disjoint."
        )

    for position in left_positions:
        if position < 0 or position >= number_of_columns:
            raise IndexError(
                "Left position is outside the matrix column range."
            )

    for position in right_positions:
        if position < 0 or position >= number_of_columns:
            raise IndexError(
                "Right position is outside the matrix column range."
            )




def select_pivot_positions(
    number_of_rows: int,
    number_of_columns: int,
    rng: random.Random | None = None,
) -> list[int]:
    """
    Randomly select column positions for a square pivot submatrix.

    """

    if number_of_rows <= 0:
        raise ValueError(
            "Number of rows must be positive."
        )

    if number_of_columns <= 0:
        raise ValueError(
            "Number of columns must be positive."
        )

    if number_of_rows > number_of_columns:
        raise ValueError(
            "Number of rows must not exceed the number of columns."
        )

    random_generator = rng if rng is not None else random

    return sorted(
        random_generator.sample(
            range(number_of_columns),
            number_of_rows,
        )
    )



def construct_systematic_form(
    parity_check_matrix: list[list[int]],
    syndrome: list[int],
    pivot_positions: list[int],
) -> tuple[list[list[int]], list[int]] | None:
    """
    Transform a parity-check matrix and syndrome into an equivalent
    systematic form over GF(2).

    """

    if not parity_check_matrix:
        raise ValueError(
            "Parity-check matrix must not be empty."
        )

    number_of_rows = len(parity_check_matrix)
    number_of_columns = len(parity_check_matrix[0])

    if any(
        len(row) != number_of_columns
        for row in parity_check_matrix
    ):
        raise ValueError(
            "All parity-check matrix rows must have the same length."
        )

    if len(syndrome) != number_of_rows:
        raise ValueError(
            "Syndrome length must match the number of matrix rows."
        )

    if len(pivot_positions) != number_of_rows:
        raise ValueError(
            "Number of pivot positions must match the number of matrix rows."
        )

    if len(set(pivot_positions)) != len(pivot_positions):
        raise ValueError(
            "Pivot positions must not contain duplicates."
        )

    if any(
        position < 0 or position >= number_of_columns
        for position in pivot_positions
    ):
        raise IndexError(
            "Pivot position is outside the matrix column range."
        )

    pivot_matrix = [
        [
            row[position]
            for position in pivot_positions
        ]
        for row in parity_check_matrix
    ]

    inverse_pivot_matrix = gf2_inverse_matrix(
        pivot_matrix
    )

    if inverse_pivot_matrix is None:
        return None

    transformed_matrix = gf2_matrix_matrix_mul(
        inverse_pivot_matrix,
        parity_check_matrix,
    )

    transformed_syndrome = gf2_matrix_vector_mul(
        inverse_pivot_matrix,
        syndrome,
    )

    return (
        transformed_matrix,
        transformed_syndrome,
    )



def find_systematic_form(
    parity_check_matrix: list[list[int]],
    syndrome: list[int],
    max_attempts: int,
    rng: random.Random | None = None,
) -> tuple[list[list[int]], list[int], list[int]] | None:
    """
    Find an invertible pivot submatrix and construct systematic form.

    """

    if max_attempts <= 0:
        raise ValueError(
            "Maximum number of attempts must be positive."
        )

    if not parity_check_matrix:
        raise ValueError(
            "Parity-check matrix must not be empty."
        )

    number_of_rows = len(parity_check_matrix)
    number_of_columns = len(parity_check_matrix[0])

    if any(
        len(row) != number_of_columns
        for row in parity_check_matrix
    ):
        raise ValueError(
            "All parity-check matrix rows must have the same length."
        )

    if len(syndrome) != number_of_rows:
        raise ValueError(
            "Syndrome length must match the number of matrix rows."
        )

    random_generator = (
        rng
        if rng is not None
        else random
    )

    for _ in range(max_attempts):
        pivot_positions = select_pivot_positions(
            number_of_rows=number_of_rows,
            number_of_columns=number_of_columns,
            rng=random_generator,
        )

        systematic_form = construct_systematic_form(
            parity_check_matrix=parity_check_matrix,
            syndrome=syndrome,
            pivot_positions=pivot_positions,
        )

        if systematic_form is None:
            continue

        transformed_matrix, transformed_syndrome = systematic_form

        return (
            transformed_matrix,
            transformed_syndrome,
            pivot_positions,
        )

    return None

        
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

    total_length = len(parity_check_matrix[0])

    validate_stern_positions(
        left_positions=left_positions,
        right_positions=right_positions,
        number_of_columns=total_length,
    )

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


def select_stern_partition(
    positions: list[int],
    seed: int | None = None,
) -> tuple[list[int], list[int]]:
    """
    Randomly split positions into left and right halves.
    
    """

    if len(positions) < 2:
        raise ValueError(
            "At least two positions are required."
        )

    rng = random.Random(seed)

    shuffled_positions = positions.copy()
    rng.shuffle(shuffled_positions)

    middle = len(shuffled_positions) // 2

    left_positions = sorted(
        shuffled_positions[:middle]
    )

    right_positions = sorted(
        shuffled_positions[middle:]
    )

    return (
        left_positions,
        right_positions,
    )



def stern_decode_with_random_partition(
    parity_check_matrix: list[list[int]],
    syndrome: list[int],
    left_weight: int,
    right_weight: int,
    seed: int | None = None,
) -> list[int] | None:
    """
    Decode a syndrome using the simplified Stern procedure with an
    automatically generated random partition of matrix columns.

    """

    if not parity_check_matrix:
        return None

    number_of_columns = len(parity_check_matrix[0])
    positions = list(range(number_of_columns))

    left_positions, right_positions = select_stern_partition(
        positions=positions,
        seed=seed,
    )

    return stern_decode(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        left_positions=left_positions,
        right_positions=right_positions,
        left_weight=left_weight,
        right_weight=right_weight,
    )



def build_stern_information_partition(
    number_of_columns: int,
    pivot_positions: list[int],
    rng: random.Random | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Construct and randomly split the information-set positions for Stern.

    """

    if number_of_columns <= 0:
        raise ValueError(
            "Number of columns must be positive."
        )

    if len(set(pivot_positions)) != len(pivot_positions):
        raise ValueError(
            "Pivot positions must not contain duplicates."
        )

    if any(
        position < 0 or position >= number_of_columns
        for position in pivot_positions
    ):
        raise IndexError(
            "Pivot position is outside the matrix column range."
        )

    pivot_set = set(pivot_positions)

    information_set = [
        position
        for position in range(number_of_columns)
        if position not in pivot_set
    ]

    if len(information_set) < 2:
        raise ValueError(
            "Information set must contain at least two positions."
        )

    random_generator = (
        rng
        if rng is not None
        else random
    )

    shuffled_positions = information_set.copy()
    random_generator.shuffle(shuffled_positions)

    middle = len(shuffled_positions) // 2

    left_positions = sorted(
        shuffled_positions[:middle]
    )

    right_positions = sorted(
        shuffled_positions[middle:]
    )

    return (
        information_set,
        left_positions,
        right_positions,
    )



def select_collision_rows(
    number_of_rows: int,
    ell: int,
    rng: random.Random | None = None,
) -> list[int]:
    """
    Randomly select syndrome rows used for Stern collision matching.
   
    """

    if number_of_rows <= 0:
        raise ValueError(
            "Number of rows must be positive."
        )

    if ell <= 0:
        raise ValueError(
            "Collision parameter ell must be positive."
        )

    if ell > number_of_rows:
        raise ValueError(
            "Collision parameter ell must not exceed the number of rows."
        )

    random_generator = (
        rng
        if rng is not None
        else random
    )

    return sorted(
        random_generator.sample(
            range(number_of_rows),
            ell,
        )
    )





def project_syndrome(
    syndrome: list[int],
    collision_rows: list[int],
) -> list[int]:
    """
    Project a syndrome onto the rows selected for Stern collision matching.

    """

    if len(set(collision_rows)) != len(collision_rows):
        raise ValueError(
            "Collision rows must not contain duplicates."
        )

    if any(
        row < 0 or row >= len(syndrome)
        for row in collision_rows
    ):
        raise IndexError(
            "Collision row is outside the syndrome range."
        )

    return [
        syndrome[row]
        for row in collision_rows
    ]