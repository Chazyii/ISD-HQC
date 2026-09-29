"""
Generation of quasi-cyclic binary syndrome-decoding instances.
"""

import random

from isd_hqc.instances.common import (
    SyndromeDecodingInstance,
    build_decoding_instance,
    generate_error_vector,
)
from isd_hqc.linear_algebra import Matrix, Vector


def cyclic_shift_right(
    vector: Vector,
    shift: int = 1,
) -> Vector:
    """
    Cyclically shift a binary vector to the right.
    """

    if not vector:
        raise ValueError(
            "Vector must not be empty."
        )

    if shift < 0:
        raise ValueError(
            "Shift must not be negative."
        )

    shift %= len(vector)

    if shift == 0:
        return vector.copy()

    return (
        vector[-shift:]
        + vector[:-shift]
    )


def build_circulant_matrix(
    first_row: Vector,
) -> Matrix:
    """
    Build a binary circulant matrix from its first row.

    Each following row is obtained by a one-position
    cyclic right shift of the previous row.
    """

    if not first_row:
        raise ValueError(
            "First row must not be empty."
        )

    if any(value not in (0, 1) for value in first_row):
        raise ValueError(
            "First row must be binary."
        )

    size = len(first_row)

    return [
        cyclic_shift_right(
            first_row,
            shift=row_index,
        )
        for row_index in range(size)
    ]





def generate_quasi_cyclic_parity_check_matrix(
    block_size: int,
    rng=None,
) -> Matrix:
    """
    Generate a binary index-2 quasi-cyclic parity-check matrix

    """

    if block_size <= 0:
        raise ValueError(
            "Block size must be positive."
        )

    if rng is None:
        rng = random

    first_row_0 = [
        rng.randint(0, 1)
        for _ in range(block_size)
    ]

    first_row_1 = [
        rng.randint(0, 1)
        for _ in range(block_size)
    ]

    block_0 = build_circulant_matrix(
        first_row_0
    )

    block_1 = build_circulant_matrix(
        first_row_1
    )

    return [
        block_0[row_index]
        + block_1[row_index]
        for row_index in range(block_size)
    ]






def generate_quasi_cyclic_code_instance(
    block_size: int,
    weight: int,
    rng=None,
) -> SyndromeDecodingInstance:
    """
    Generate an index-2 quasi-cyclic syndrome-decoding instance.
    """

    if rng is None:
        rng = random

    parity_check_matrix = (
        generate_quasi_cyclic_parity_check_matrix(
            block_size=block_size,
            rng=rng,
        )
    )

    columns = 2 * block_size

    error = generate_error_vector(
        length=columns,
        weight=weight,
        rng=rng,
    )

    return build_decoding_instance(
        parity_check_matrix=parity_check_matrix,
        error=error,
    )