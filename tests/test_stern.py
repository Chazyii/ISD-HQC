import pytest

from isd_hqc.algorithms.stern import generate_weight_vectors


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