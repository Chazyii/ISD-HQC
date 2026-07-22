import pytest

from isd_hqc.algorithms.stern import (
    compute_partial_syndrome,
    generate_weight_vectors,
)


def test_generate_weight_vectors():
    vectors = generate_weight_vectors(
        length=4,
        weight=2,
    )

    assert vectors == [
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ]


def test_generate_weight_vectors_have_correct_weight():
    vectors = generate_weight_vectors(
        length=5,
        weight=2,
    )

    for vector in vectors:
        assert sum(vector) == 2


def test_generate_weight_vectors_have_correct_length():
    vectors = generate_weight_vectors(
        length=5,
        weight=2,
    )

    for vector in vectors:
        assert len(vector) == 5


def test_generate_weight_vectors_count():
    vectors = generate_weight_vectors(
        length=4,
        weight=2,
    )

    assert len(vectors) == 6


def test_generate_zero_weight_vector():
    vectors = generate_weight_vectors(
        length=4,
        weight=0,
    )

    assert vectors == [
        [0, 0, 0, 0],
    ]


def test_generate_full_weight_vector():
    vectors = generate_weight_vectors(
        length=4,
        weight=4,
    )

    assert vectors == [
        [1, 1, 1, 1],
    ]


def test_generate_weight_vectors_rejects_negative_length():
    with pytest.raises(
        ValueError,
        match="Length must be non-negative.",
    ):
        generate_weight_vectors(
            length=-1,
            weight=0,
        )


def test_generate_weight_vectors_rejects_negative_weight():
    with pytest.raises(
        ValueError,
        match="Weight must be non-negative.",
    ):
        generate_weight_vectors(
            length=4,
            weight=-1,
        )


def test_generate_weight_vectors_rejects_weight_greater_than_length():
    with pytest.raises(
        ValueError,
        match="Weight cannot be greater than vector length.",
    ):
        generate_weight_vectors(
            length=4,
            weight=5,
        )


def test_compute_partial_syndrome():
    parity_check_matrix = [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
    ]

    result = compute_partial_syndrome(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 2],
        partial_error=[1, 1],
    )

    assert result == [0, 1]


def test_compute_partial_syndrome_with_zero_error():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    result = compute_partial_syndrome(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1],
        partial_error=[0, 0],
    )

    assert result == [0, 0]


def test_compute_partial_syndrome_single_position():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    result = compute_partial_syndrome(
        parity_check_matrix=parity_check_matrix,
        positions=[2],
        partial_error=[1],
    )

    assert result == [1, 1]


def test_compute_partial_syndrome_rejects_length_mismatch():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    with pytest.raises(
        ValueError,
        match="Number of positions must match partial error length.",
    ):
        compute_partial_syndrome(
            parity_check_matrix=parity_check_matrix,
            positions=[0, 1],
            partial_error=[1],
        )


def test_compute_partial_syndrome_rejects_invalid_position():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    with pytest.raises(
        IndexError,
        match="Partial error position is outside the matrix range.",
    ):
        compute_partial_syndrome(
            parity_check_matrix=parity_check_matrix,
            positions=[3],
            partial_error=[1],
        )