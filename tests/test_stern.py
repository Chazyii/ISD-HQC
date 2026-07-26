import pytest

from isd_hqc.algorithms.stern import (
    build_partial_syndrome_list,
    compute_partial_syndrome,
    generate_weight_vectors,
    find_syndrome_collisions,
    reconstruct_candidate_error,
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


def test_build_partial_syndrome_list():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    result = build_partial_syndrome_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2],
        weight=1,
    )

    assert result == [
        ([1, 0], [1, 0, 0]),
        ([0, 1], [0, 1, 0]),
        ([1, 1], [0, 0, 1]),
    ]


def test_build_partial_syndrome_list_count():
    parity_check_matrix = [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
    ]

    result = build_partial_syndrome_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2, 3],
        weight=2,
    )

    assert len(result) == 6


def test_build_partial_syndrome_list_preserves_weight():
    parity_check_matrix = [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
    ]

    result = build_partial_syndrome_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2, 3],
        weight=2,
    )

    for _, partial_error in result:
        assert sum(partial_error) == 2


def test_build_partial_syndrome_list_zero_weight():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]

    result = build_partial_syndrome_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1],
        weight=0,
    )

    assert result == [
        ([0, 0], [0, 0]),
    ]


def test_find_syndrome_collisions():
    left_list = [
        ([1, 0], [1, 0]),
        ([0, 1], [0, 1]),
    ]

    right_list = [
        ([0, 1], [1, 0]),
        ([1, 1], [0, 1]),
    ]

    collisions = find_syndrome_collisions(
        left_list=left_list,
        right_list=right_list,
    )

    assert collisions == [
        (
            [0, 1],
            [1, 0],
        ),
    ]

def test_find_syndrome_collisions_with_multiple_matches():
    left_list = [
        ([1, 0], [1, 0]),
        ([1, 0], [0, 1]),
    ]

    right_list = [
        ([1, 0], [1, 1]),
    ]

    collisions = find_syndrome_collisions(
        left_list=left_list,
        right_list=right_list,
    )

    assert collisions == [
        (
            [1, 0],
            [1, 1],
        ),
        (
            [0, 1],
            [1, 1],
        ),
    ]

def test_find_syndrome_collisions_without_matches():
    left_list = [
        ([1, 0], [1, 0]),
    ]

    right_list = [
        ([0, 1], [0, 1]),
    ]

    collisions = find_syndrome_collisions(
        left_list=left_list,
        right_list=right_list,
    )

    assert collisions == []

def test_find_syndrome_collisions_with_empty_lists():
    collisions = find_syndrome_collisions(
        left_list=[],
        right_list=[],
    )

    assert collisions == []



def test_reconstruct_candidate_error():
    result = reconstruct_candidate_error(
        left_positions=[0, 2],
        left_error=[1, 1],
        right_positions=[4],
        right_error=[1],
        length=6,
    )

    assert result == [
        1,
        0,
        1,
        0,
        1,
        0,
    ]

def test_reconstruct_candidate_error_with_zero_values():
    result = reconstruct_candidate_error(
        left_positions=[0, 2],
        left_error=[0, 0],
        right_positions=[3],
        right_error=[1],
        length=5,
    )

    assert result == [
        0,
        0,
        0,
        1,
        0,
    ]

def test_reconstruct_candidate_error_rejects_left_length_mismatch():
    with pytest.raises(
        ValueError,
        match="Left positions must match left partial error length.",
    ):
        reconstruct_candidate_error(
            left_positions=[0, 1],
            left_error=[1],
            right_positions=[],
            right_error=[],
            length=4,
        )

def test_reconstruct_candidate_error_rejects_invalid_left_position():
    with pytest.raises(
        IndexError,
        match="Left position is outside the candidate error vector.",
    ):
        reconstruct_candidate_error(
            left_positions=[5],
            left_error=[1],
            right_positions=[],
            right_error=[],
            length=5,
        )

def test_reconstruct_candidate_error_rejects_invalid_right_position():
    with pytest.raises(
        IndexError,
        match="Right position is outside the candidate error vector.",
    ):
        reconstruct_candidate_error(
            left_positions=[],
            left_error=[],
            right_positions=[6],
            right_error=[1],
            length=5,
        )