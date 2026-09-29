import random

import pytest

from isd_hqc.instances.common import (
    build_decoding_instance,
    generate_error_vector,
)


def test_generate_error_vector_has_correct_weight():
    error = generate_error_vector(
        length=10,
        weight=3,
        rng=random.Random(42),
    )

    assert len(error) == 10
    assert sum(error) == 3
    assert all(value in (0, 1) for value in error)


def test_generate_error_vector_is_reproducible():
    error1 = generate_error_vector(
        length=10,
        weight=3,
        rng=random.Random(42),
    )

    error2 = generate_error_vector(
        length=10,
        weight=3,
        rng=random.Random(42),
    )

    assert error1 == error2


def test_generate_error_vector_rejects_negative_weight():
    with pytest.raises(
        ValueError,
        match="Weight must not be negative.",
    ):
        generate_error_vector(
            length=10,
            weight=-1,
            rng=random.Random(42),
        )


def test_generate_error_vector_rejects_excessive_weight():
    with pytest.raises(
        ValueError,
        match="Weight must not exceed vector length.",
    ):
        generate_error_vector(
            length=5,
            weight=6,
            rng=random.Random(42),
        )


def test_build_decoding_instance_computes_syndrome():
    matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    error = [1, 0, 1, 0]

    instance = build_decoding_instance(
        parity_check_matrix=matrix,
        error=error,
    )

    assert instance.syndrome == [0, 0]
    assert instance.weight == 2
    assert instance.rows == 2
    assert instance.columns == 4