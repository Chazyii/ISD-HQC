"""
Common structures and helpers for syndrome-decoding instances.
"""

from dataclasses import dataclass

from isd_hqc.linear_algebra import (
    Matrix,
    Vector,
    gf2_matrix_vector_mul,
)


@dataclass
class SyndromeDecodingInstance:
    """
    A syndrome-decoding problem instance.

    """

    parity_check_matrix: Matrix
    syndrome: Vector
    error: Vector
    weight: int

    @property
    def rows(self) -> int:
        return len(self.parity_check_matrix)

    @property
    def columns(self) -> int:
        if not self.parity_check_matrix:
            return 0

        return len(self.parity_check_matrix[0])


def generate_error_vector(
    length: int,
    weight: int,
    rng,
) -> Vector:
    """
    Generate a random binary vector with exactly the requested
    Hamming weight.
    """

    if length <= 0:
        raise ValueError(
            "Length must be positive."
        )

    if weight < 0:
        raise ValueError(
            "Weight must not be negative."
        )

    if weight > length:
        raise ValueError(
            "Weight must not exceed vector length."
        )

    error = [0] * length

    error_positions = rng.sample(
        range(length),
        weight,
    )

    for position in error_positions:
        error[position] = 1

    return error


def build_decoding_instance(
    parity_check_matrix: Matrix,
    error: Vector,
) -> SyndromeDecodingInstance:
    """
    Build a syndrome-decoding instance from a parity-check
    matrix and an error vector.
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

    if len(error) != number_of_columns:
        raise ValueError(
            "Error length must match the number of matrix columns."
        )

    if any(value not in (0, 1) for value in error):
        raise ValueError(
            "Error vector must be binary."
        )

    syndrome = gf2_matrix_vector_mul(
        parity_check_matrix,
        error,
    )

    return SyndromeDecodingInstance(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        error=error.copy(),
        weight=sum(error),
    )