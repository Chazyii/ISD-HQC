"""
Generation of random binary syndrome-decoding instances.
"""

import random

from isd_hqc.instances.common import (
    SyndromeDecodingInstance,
    build_decoding_instance,
    generate_error_vector,
)
from isd_hqc.linear_algebra import Matrix


def generate_random_parity_check_matrix(
    rows: int,
    columns: int,
    rng=None,
) -> Matrix:
    """
    Generate a random binary parity-check matrix.
    """

    if rows <= 0:
        raise ValueError(
            "Number of rows must be positive."
        )

    if columns <= 0:
        raise ValueError(
            "Number of columns must be positive."
        )

    if rows >= columns:
        raise ValueError(
            "Number of rows must be smaller than number of columns."
        )

    if rng is None:
        rng = random

    return [
        [
            rng.randint(0, 1)
            for _ in range(columns)
        ]
        for _ in range(rows)
    ]


def generate_random_code_instance(
    rows: int,
    columns: int,
    weight: int,
    rng=None,
) -> SyndromeDecodingInstance:
    """
    Generate a random binary syndrome-decoding instance.
    """

    if rng is None:
        rng = random

    parity_check_matrix = (
        generate_random_parity_check_matrix(
            rows=rows,
            columns=columns,
            rng=rng,
        )
    )

    error = generate_error_vector(
        length=columns,
        weight=weight,
        rng=rng,
    )

    return build_decoding_instance(
        parity_check_matrix=parity_check_matrix,
        error=error,
    )