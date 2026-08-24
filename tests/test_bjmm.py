import pytest

from isd_hqc.algorithms.bjmm import (
    build_bjmm_syndrome_list,
    generate_representations,
    is_valid_representation,
)


def test_is_valid_representation():
    target_vector = [
        1, 1, 0, 0,
    ]

    left_vector = [
        1, 0, 1, 0,
    ]

    right_vector = [
        0, 1, 1, 0,
    ]

    result = is_valid_representation(
        target_vector=target_vector,
        left_vector=left_vector,
        right_vector=right_vector,
        component_weight=2,
    )

    assert result is True


def test_is_valid_representation_uses_cancellation():
    target_vector = [
        1, 1, 0, 0,
    ]

    left_vector = [
        1, 0, 1, 0,
    ]

    right_vector = [
        0, 1, 1, 0,
    ]

    assert left_vector[2] == 1
    assert right_vector[2] == 1

    assert is_valid_representation(
        target_vector=target_vector,
        left_vector=left_vector,
        right_vector=right_vector,
        component_weight=2,
    )


def test_is_valid_representation_rejects_wrong_xor():
    result = is_valid_representation(
        target_vector=[1, 1, 0, 0],
        left_vector=[1, 0, 1, 0],
        right_vector=[0, 0, 1, 1],
        component_weight=2,
    )

    assert result is False


def test_is_valid_representation_rejects_wrong_component_weight():
    result = is_valid_representation(
        target_vector=[1, 1, 0, 0],
        left_vector=[1, 0, 0, 0],
        right_vector=[0, 1, 0, 0],
        component_weight=2,
    )

    assert result is False


def test_is_valid_representation_rejects_left_length_mismatch():
    with pytest.raises(
        ValueError,
        match="Left vector length must match target vector length.",
    ):
        is_valid_representation(
            target_vector=[1, 1, 0],
            left_vector=[1, 0],
            right_vector=[0, 1, 0],
            component_weight=1,
        )


def test_is_valid_representation_rejects_right_length_mismatch():
    with pytest.raises(
        ValueError,
        match="Right vector length must match target vector length.",
    ):
        is_valid_representation(
            target_vector=[1, 1, 0],
            left_vector=[1, 0, 0],
            right_vector=[0, 1],
            component_weight=1,
        )


def test_is_valid_representation_rejects_negative_component_weight():
    with pytest.raises(
        ValueError,
        match="Component weight must not be negative.",
    ):
        is_valid_representation(
            target_vector=[1, 1],
            left_vector=[1, 0],
            right_vector=[0, 1],
            component_weight=-1,
        )




def test_generate_representations():
    target_vector = [
        1, 1, 0, 0,
    ]

    representations = generate_representations(
        target_vector=target_vector,
        component_weight=2,
    )

    assert representations == [
        (
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ),
        (
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ),
        (
            [0, 1, 1, 0],
            [1, 0, 1, 0],
        ),
        (
            [0, 1, 0, 1],
            [1, 0, 0, 1],
        ),
    ]


def test_generate_representations_are_valid():
    target_vector = [
        1, 1, 0, 0,
    ]

    representations = generate_representations(
        target_vector=target_vector,
        component_weight=2,
    )

    for left_vector, right_vector in representations:
        assert is_valid_representation(
            target_vector=target_vector,
            left_vector=left_vector,
            right_vector=right_vector,
            component_weight=2,
        )


def test_generate_representations_can_return_empty_list():
    representations = generate_representations(
        target_vector=[1, 0, 0],
        component_weight=1,
    )

    assert representations == []


def test_generate_representations_zero_vector():
    representations = generate_representations(
        target_vector=[0, 0],
        component_weight=1,
    )

    assert representations == [
        (
            [1, 0],
            [1, 0],
        ),
        (
            [0, 1],
            [0, 1],
        ),
    ]


def test_generate_representations_rejects_negative_weight():
    with pytest.raises(
        ValueError,
        match="Component weight must not be negative.",
    ):
        generate_representations(
            target_vector=[1, 0],
            component_weight=-1,
        )


def test_generate_representations_rejects_excessive_weight():
    with pytest.raises(
        ValueError,
        match="Component weight must not exceed vector length.",
    ):
        generate_representations(
            target_vector=[1, 0],
            component_weight=3,
        )


def test_generate_representations_rejects_non_binary_target():
    with pytest.raises(
        ValueError,
        match="Target vector must contain only binary values.",
    ):
        generate_representations(
            target_vector=[1, 2, 0],
            component_weight=1,
        )





def test_build_bjmm_syndrome_list():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]

    vectors = [
        [1, 0, 0],
        [0, 1, 0],
    ]

    result = build_bjmm_syndrome_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2],
        vectors=vectors,
        merge_rows=[0, 2],
    )

    assert result == [
        ([1, 1], [1, 0, 0]),
        ([0, 1], [0, 1, 0]),
    ]


def test_build_bjmm_syndrome_list_preserves_all_vectors():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]

    vectors = [
        [1, 0],
        [0, 1],
        [1, 1],
    ]

    result = build_bjmm_syndrome_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1],
        vectors=vectors,
        merge_rows=[0],
    )

    assert len(result) == 3

    assert [vector for _, vector in result] == vectors


def test_build_bjmm_syndrome_list_rejects_invalid_vector_length():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]

    with pytest.raises(
        ValueError,
        match="Vector length must match the number of positions.",
    ):
        build_bjmm_syndrome_list(
            parity_check_matrix=parity_check_matrix,
            positions=[0, 1],
            vectors=[[1]],
            merge_rows=[0],
        )


def test_build_bjmm_syndrome_list_rejects_invalid_position():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]

    with pytest.raises(
        IndexError,
        match="Position is outside the matrix column range.",
    ):
        build_bjmm_syndrome_list(
            parity_check_matrix=parity_check_matrix,
            positions=[0, 2],
            vectors=[[1, 0]],
            merge_rows=[0],
        )