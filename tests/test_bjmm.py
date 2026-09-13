import pytest
import random
from isd_hqc.algorithms.bjmm import (
    build_bjmm_base_list,
    build_bjmm_base_lists,
    build_bjmm_merge_targets,
    build_bjmm_syndrome_list,
    find_valid_bjmm_candidate,
    generate_representations,
    is_valid_representation,
    merge_bjmm_level,
    merge_bjmm_lists,
    merge_bjmm_tree,
    reconstruct_bjmm_candidate,
    split_bjmm_information_positions,
    split_bjmm_merge_rows,
    validate_bjmm_parameters,
)

from isd_hqc.syndrome import (
    compute_syndrome,
    verify_solution,
)

from isd_hqc.linear_algebra import (
    gf2_add_vectors,
    hamming_weight,
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





def test_merge_bjmm_lists():
    left_list = [
        (
            [1, 0],
            [1, 0, 1, 0],
        ),
    ]

    right_list = [
        (
            [0, 1],
            [0, 1, 1, 0],
        ),
    ]

    result = merge_bjmm_lists(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=[1, 1],
    )

    assert result == [
        [1, 1, 0, 0],
    ]


def test_merge_bjmm_lists_finds_multiple_merges():
    left_list = [
        (
            [1, 0],
            [1, 0, 1, 0],
        ),
        (
            [0, 1],
            [0, 1, 0, 1],
        ),
    ]

    right_list = [
        (
            [0, 1],
            [0, 1, 1, 0],
        ),
        (
            [1, 0],
            [1, 0, 0, 1],
        ),
    ]

    result = merge_bjmm_lists(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=[1, 1],
    )

    assert result == [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ]


def test_merge_bjmm_lists_returns_empty_list_without_match():
    left_list = [
        (
            [1, 0],
            [1, 0],
        ),
    ]

    right_list = [
        (
            [1, 0],
            [0, 1],
        ),
    ]

    result = merge_bjmm_lists(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=[1, 1],
    )

    assert result == []


def test_merge_bjmm_lists_rejects_invalid_left_syndrome_length():
    left_list = [
        (
            [1],
            [1, 0],
        ),
    ]

    right_list = [
        (
            [0, 1],
            [0, 1],
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Left projected syndrome length must match target syndrome length.",
    ):
        merge_bjmm_lists(
            left_list=left_list,
            right_list=right_list,
            target_syndrome=[1, 1],
        )


def test_merge_bjmm_lists_rejects_invalid_right_syndrome_length():
    left_list = [
        (
            [1, 0],
            [1, 0],
        ),
    ]

    right_list = [
        (
            [1],
            [0, 1],
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Right projected syndrome length must match target syndrome length.",
    ):
        merge_bjmm_lists(
            left_list=left_list,
            right_list=right_list,
            target_syndrome=[1, 1],
        )


def test_merge_bjmm_lists_rejects_different_vector_lengths():
    left_list = [
        (
            [1, 0],
            [1, 0, 0],
        ),
    ]

    right_list = [
        (
            [0, 1],
            [0, 1],
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Vectors selected for merging must have the same length.",
    ):
        merge_bjmm_lists(
            left_list=left_list,
            right_list=right_list,
            target_syndrome=[1, 1],
        )





def test_merge_bjmm_level():
    parity_check_matrix = [
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
    ]

    left_list = [
        (
            [1],
            [1, 0, 1, 0],
        ),
    ]

    right_list = [
        (
            [0],
            [0, 1, 1, 0],
        ),
    ]

    result = merge_bjmm_level(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2, 3],
        left_list=left_list,
        right_list=right_list,
        merge_target=[1],
        next_merge_rows=[1, 2],
    )

    assert result == [
        (
            [1, 0],
            [1, 1, 0, 0],
        ),
    ]


def test_merge_bjmm_level_returns_empty_list_without_collision():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]

    left_list = [
        (
            [0],
            [1, 0],
        ),
    ]

    right_list = [
        (
            [0],
            [0, 1],
        ),
    ]

    result = merge_bjmm_level(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1],
        left_list=left_list,
        right_list=right_list,
        merge_target=[1],
        next_merge_rows=[1],
    )

    assert result == []


def test_merge_bjmm_level_preserves_merged_vector():
    parity_check_matrix = [
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
    ]

    result = merge_bjmm_level(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2, 3],
        left_list=[
            (
                [1],
                [1, 0, 1, 0],
            ),
        ],
        right_list=[
            (
                [0],
                [0, 1, 1, 0],
            ),
        ],
        merge_target=[1],
        next_merge_rows=[1],
    )

    assert len(result) == 1

    projected_syndrome, merged_vector = result[0]

    assert merged_vector == [
        1, 1, 0, 0,
    ]

    assert projected_syndrome == [1]





def test_merge_bjmm_tree_finds_candidate():
    parity_check_matrix = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]

    result = merge_bjmm_tree(
        parity_check_matrix=parity_check_matrix,
        information_positions=[0, 1, 2, 3],
        syndrome=[0, 1],
        p=4,
        p1=2,
        ell1=1,
        ell2=1,
        merge_rows=[0, 1],
    )

    assert result

    for candidate in result:
        assert candidate == [1, 1, 1, 1]
        assert hamming_weight(candidate) == 4


def test_merge_bjmm_tree_returns_empty_when_final_target_does_not_match():
    parity_check_matrix = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]

    result = merge_bjmm_tree(
        parity_check_matrix=parity_check_matrix,
        information_positions=[0, 1, 2, 3],
        syndrome=[0, 0],
        p=4,
        p1=2,
        ell1=1,
        ell2=1,
        merge_rows=[0, 1],
    )

    assert result == []


def test_merge_bjmm_tree_candidates_have_target_weight():
    parity_check_matrix = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]

    result = merge_bjmm_tree(
        parity_check_matrix=parity_check_matrix,
        information_positions=[0, 1, 2, 3],
        syndrome=[0, 1],
        p=4,
        p1=2,
        ell1=1,
        ell2=1,
        merge_rows=[0, 1],
    )

    assert result

    for candidate in result:
        assert hamming_weight(candidate) == 4





def test_reconstruct_bjmm_candidate():
    systematic_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    transformed_syndrome = [
        0,
        1,
    ]

    result = reconstruct_bjmm_candidate(
        systematic_matrix=systematic_matrix,
        transformed_syndrome=transformed_syndrome,
        pivot_positions=[0, 1],
        information_positions=[2, 3],
        information_error=[1, 0],
    )

    assert result == [
        1, 1, 1, 0,
    ]


def test_reconstruct_bjmm_candidate_satisfies_syndrome():
    systematic_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    transformed_syndrome = [
        0,
        1,
    ]

    candidate = reconstruct_bjmm_candidate(
        systematic_matrix=systematic_matrix,
        transformed_syndrome=transformed_syndrome,
        pivot_positions=[0, 1],
        information_positions=[2, 3],
        information_error=[1, 0],
    )

    assert compute_syndrome(
        systematic_matrix,
        candidate,
    ) == transformed_syndrome


def test_reconstruct_bjmm_candidate_with_multiple_information_errors():
    systematic_matrix = [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
    ]

    transformed_syndrome = [
        1,
        0,
    ]

    candidate = reconstruct_bjmm_candidate(
        systematic_matrix=systematic_matrix,
        transformed_syndrome=transformed_syndrome,
        pivot_positions=[0, 1],
        information_positions=[2, 3],
        information_error=[1, 1],
    )

    assert candidate == [
        1, 1, 1, 1,
    ]

    assert compute_syndrome(
        systematic_matrix,
        candidate,
    ) == transformed_syndrome


def test_reconstruct_bjmm_candidate_rejects_information_length_mismatch():
    with pytest.raises(
        ValueError,
        match="Information positions must match information error length.",
    ):
        reconstruct_bjmm_candidate(
            systematic_matrix=[
                [1, 0, 1],
                [0, 1, 1],
            ],
            transformed_syndrome=[0, 1],
            pivot_positions=[0, 1],
            information_positions=[2],
            information_error=[1, 0],
        )


def test_reconstruct_bjmm_candidate_rejects_overlapping_positions():
    with pytest.raises(
        ValueError,
        match="Pivot and information positions must be disjoint.",
    ):
        reconstruct_bjmm_candidate(
            systematic_matrix=[
                [1, 0, 1],
                [0, 1, 1],
            ],
            transformed_syndrome=[0, 1],
            pivot_positions=[0, 1],
            information_positions=[1],
            information_error=[1],
        )


def test_reconstruct_bjmm_candidate_rejects_duplicate_pivots():
    with pytest.raises(
        ValueError,
        match="Pivot positions must not contain duplicates.",
    ):
        reconstruct_bjmm_candidate(
            systematic_matrix=[
                [1, 0, 1],
                [0, 1, 1],
            ],
            transformed_syndrome=[0, 1],
            pivot_positions=[0, 0],
            information_positions=[2],
            information_error=[1],
        )


def test_reconstruct_bjmm_candidate_rejects_duplicate_information_positions():
    with pytest.raises(
        ValueError,
        match="Information positions must not contain duplicates.",
    ):
        reconstruct_bjmm_candidate(
            systematic_matrix=[
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            transformed_syndrome=[0, 1],
            pivot_positions=[0, 1],
            information_positions=[2, 2],
            information_error=[1, 1],
        )


def test_reconstruct_bjmm_candidate_rejects_position_outside_matrix():
    with pytest.raises(
        IndexError,
        match="Error position is outside the matrix column range.",
    ):
        reconstruct_bjmm_candidate(
            systematic_matrix=[
                [1, 0, 1],
                [0, 1, 1],
            ],
            transformed_syndrome=[0, 1],
            pivot_positions=[0, 1],
            information_positions=[3],
            information_error=[1],
        )





def test_find_valid_bjmm_candidate():
    systematic_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    result = find_valid_bjmm_candidate(
        systematic_matrix=systematic_matrix,
        transformed_syndrome=[0, 1],
        pivot_positions=[0, 1],
        information_positions=[2, 3],
        information_candidates=[
            [1, 0],
        ],
        target_weight=3,
    )

    assert result is not None

    assert hamming_weight(result) == 3

    assert verify_solution(
        parity_check_matrix=systematic_matrix,
        error=result,
        syndrome=[0, 1],
        weight=3,
    )


def test_find_valid_bjmm_candidate_skips_wrong_weight():
    systematic_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    result = find_valid_bjmm_candidate(
        systematic_matrix=systematic_matrix,
        transformed_syndrome=[0, 1],
        pivot_positions=[0, 1],
        information_positions=[2, 3],
        information_candidates=[
            [0, 0],
            [1, 0],
        ],
        target_weight=3,
    )

    assert result is not None

    assert hamming_weight(result) == 3

    assert verify_solution(
        parity_check_matrix=systematic_matrix,
        error=result,
        syndrome=[0, 1],
        weight=3,
    )


def test_find_valid_bjmm_candidate_returns_none_without_valid_candidate():
    systematic_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    result = find_valid_bjmm_candidate(
        systematic_matrix=systematic_matrix,
        transformed_syndrome=[0, 1],
        pivot_positions=[0, 1],
        information_positions=[2, 3],
        information_candidates=[
            [0, 0],
        ],
        target_weight=3,
    )

    assert result is None


def test_find_valid_bjmm_candidate_returns_none_for_empty_candidates():
    result = find_valid_bjmm_candidate(
        systematic_matrix=[
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        transformed_syndrome=[0, 1],
        pivot_positions=[0, 1],
        information_positions=[2, 3],
        information_candidates=[],
        target_weight=3,
    )

    assert result is None


def test_find_valid_bjmm_candidate_rejects_negative_target_weight():
    with pytest.raises(
        ValueError,
        match="Target weight must not be negative.",
    ):
        find_valid_bjmm_candidate(
            systematic_matrix=[
                [1, 0, 1],
                [0, 1, 1],
            ],
            transformed_syndrome=[0, 1],
            pivot_positions=[0, 1],
            information_positions=[2],
            information_candidates=[
                [1],
            ],
            target_weight=-1,
        )


def test_find_valid_bjmm_candidate_rejects_excessive_target_weight():
    with pytest.raises(
        ValueError,
        match="Target weight must not exceed the code length.",
    ):
        find_valid_bjmm_candidate(
            systematic_matrix=[
                [1, 0, 1],
                [0, 1, 1],
            ],
            transformed_syndrome=[0, 1],
            pivot_positions=[0, 1],
            information_positions=[2],
            information_candidates=[
                [1],
            ],
            target_weight=4,
        )






def test_merge_bjmm_lists_filters_by_target_weight():
    left_list = [
        ([0], [1, 1, 0, 0]),
    ]

    right_list = [
        ([0], [1, 0, 1, 0]),
        ([0], [0, 0, 1, 1]),
    ]

    result = merge_bjmm_lists(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=[0],
        target_weight=2,
    )

    assert result == [
        [0, 1, 1, 0],
    ]


def test_merge_bjmm_lists_without_target_weight_keeps_all_matches():
    left_list = [
        ([0], [1, 1, 0, 0]),
    ]

    right_list = [
        ([0], [1, 0, 1, 0]),
        ([0], [0, 0, 1, 1]),
    ]

    result = merge_bjmm_lists(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=[0],
    )

    assert result == [
        [0, 1, 1, 0],
        [1, 1, 1, 1],
    ]


def test_merge_bjmm_lists_returns_empty_when_weight_does_not_match():
    left_list = [
        ([0], [1, 1, 0, 0]),
    ]

    right_list = [
        ([0], [0, 0, 1, 1]),
    ]

    result = merge_bjmm_lists(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=[0],
        target_weight=2,
    )

    assert result == []


def test_merge_bjmm_lists_rejects_negative_target_weight():
    with pytest.raises(
        ValueError,
        match="Target weight must not be negative.",
    ):
        merge_bjmm_lists(
            left_list=[],
            right_list=[],
            target_syndrome=[0],
            target_weight=-1,
        )




def test_merge_bjmm_level_filters_intermediate_weight():
    parity_check_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    positions = [0, 1, 2, 3]

    left_list = [
        ([0], [1, 1, 0, 0]),
    ]

    right_list = [
        ([0], [1, 0, 1, 0]),
        ([0], [0, 0, 1, 1]),
    ]

    result = merge_bjmm_level(
        parity_check_matrix=parity_check_matrix,
        positions=positions,
        left_list=left_list,
        right_list=right_list,
        merge_target=[0],
        next_merge_rows=[1, 2],
        merged_weight=2,
    )

    assert result == [
        ([1, 1], [0, 1, 1, 0]),
    ]





def test_build_bjmm_base_list():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    result = build_bjmm_base_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2],
        component_weight=1,
        merge_rows=[0],
    )

    assert result == [
        ([1], [1, 0, 0]),
        ([0], [0, 1, 0]),
        ([1], [0, 0, 1]),
    ]


def test_build_bjmm_base_list_generates_correct_weight():
    parity_check_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    result = build_bjmm_base_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2, 3],
        component_weight=2,
        merge_rows=[0],
    )

    assert len(result) == 6

    for _, vector in result:
        assert hamming_weight(vector) == 2


def test_build_bjmm_base_list_zero_weight():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]

    result = build_bjmm_base_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1],
        component_weight=0,
        merge_rows=[0, 1],
    )

    assert result == [
        ([0, 0], [0, 0]),
    ]


def test_build_bjmm_base_list_rejects_negative_weight():
    with pytest.raises(
        ValueError,
        match="Component weight must not be negative.",
    ):
        build_bjmm_base_list(
            parity_check_matrix=[
                [1, 0],
                [0, 1],
            ],
            positions=[0, 1],
            component_weight=-1,
            merge_rows=[0],
        )


def test_build_bjmm_base_list_rejects_excessive_weight():
    with pytest.raises(
        ValueError,
        match="Component weight must not exceed the number of positions.",
    ):
        build_bjmm_base_list(
            parity_check_matrix=[
                [1, 0],
                [0, 1],
            ],
            positions=[0, 1],
            component_weight=3,
            merge_rows=[0],
        )






def test_split_bjmm_information_positions():
    first_half, second_half = split_bjmm_information_positions(
        information_positions=[2, 3, 4, 5, 6, 7],
    )

    assert first_half == [2, 3, 4]
    assert second_half == [5, 6, 7]


def test_split_bjmm_information_positions_preserves_all_positions():
    information_positions = [
        4, 7, 2, 9,
    ]

    first_half, second_half = split_bjmm_information_positions(
        information_positions=information_positions,
    )

    assert first_half + second_half == information_positions


def test_split_bjmm_information_positions_creates_equal_halves():
    first_half, second_half = split_bjmm_information_positions(
        information_positions=[0, 2, 4, 6, 8, 10],
    )

    assert len(first_half) == 3
    assert len(second_half) == 3


def test_split_bjmm_information_positions_rejects_empty_positions():
    with pytest.raises(
        ValueError,
        match="Information positions must not be empty.",
    ):
        split_bjmm_information_positions(
            information_positions=[],
        )


def test_split_bjmm_information_positions_rejects_duplicates():
    with pytest.raises(
        ValueError,
        match="Information positions must not contain duplicates.",
    ):
        split_bjmm_information_positions(
            information_positions=[2, 3, 3, 4],
        )


def test_split_bjmm_information_positions_rejects_odd_number_of_positions():
    with pytest.raises(
        ValueError,
        match="Number of information positions must be even.",
    ):
        split_bjmm_information_positions(
            information_positions=[2, 3, 4],
        )





def test_build_bjmm_base_lists():
    parity_check_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    l1, l2, l3, l4 = build_bjmm_base_lists(
        parity_check_matrix=parity_check_matrix,
        information_positions=[0, 1, 2, 3],
        p1=2,
        merge_rows=[0, 1],
    )

    expected_first_half_list = [
        ([1, 0], [1, 0, 0, 0]),
        ([0, 1], [0, 1, 0, 0]),
    ]

    expected_second_half_list = [
        ([1, 0], [0, 0, 1, 0]),
        ([0, 1], [0, 0, 0, 1]),
    ]

    assert l1 == expected_first_half_list
    assert l2 == expected_first_half_list
    assert l3 == expected_second_half_list
    assert l4 == expected_second_half_list


def test_build_bjmm_base_lists_have_weight_p1_over_two():
    parity_check_matrix = [
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ]

    l1, l2, l3, l4 = build_bjmm_base_lists(
        parity_check_matrix=parity_check_matrix,
        information_positions=[0, 1, 2, 3, 4, 5],
        p1=4,
        merge_rows=[0],
    )

    for bjmm_list in [l1, l2, l3, l4]:
        for _, vector in bjmm_list:
            assert hamming_weight(vector) == 2


def test_build_bjmm_base_lists_use_full_information_vector_length():
    parity_check_matrix = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ]

    l1, l2, l3, l4 = build_bjmm_base_lists(
        parity_check_matrix=parity_check_matrix,
        information_positions=[0, 1, 2, 3, 4, 5],
        p1=2,
        merge_rows=[0],
    )

    for bjmm_list in [l1, l2, l3, l4]:
        for _, vector in bjmm_list:
            assert len(vector) == 6


def test_build_bjmm_base_lists_use_correct_halves():
    parity_check_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    l1, l2, l3, l4 = build_bjmm_base_lists(
        parity_check_matrix=parity_check_matrix,
        information_positions=[0, 1, 2, 3],
        p1=2,
        merge_rows=[0],
    )

    for _, vector in l1 + l3:
        assert vector[2:] == [0, 0]

    for _, vector in l2 + l4:
        assert vector[:2] == [0, 0]


def test_build_bjmm_base_lists_rejects_negative_p1():
    with pytest.raises(
        ValueError,
        match="p1 must not be negative.",
    ):
        build_bjmm_base_lists(
            parity_check_matrix=[
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            information_positions=[0, 1, 2, 3],
            p1=-2,
            merge_rows=[0],
        )


def test_build_bjmm_base_lists_rejects_odd_p1():
    with pytest.raises(
        ValueError,
        match="p1 must be even.",
    ):
        build_bjmm_base_lists(
            parity_check_matrix=[
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            information_positions=[0, 1, 2, 3],
            p1=3,
            merge_rows=[0],
        )


def test_build_bjmm_base_lists_rejects_excessive_p1():
    with pytest.raises(
        ValueError,
        match="p1 / 2 must not exceed half of the information positions.",
    ):
        build_bjmm_base_lists(
            parity_check_matrix=[
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            information_positions=[0, 1, 2, 3],
            p1=6,
            merge_rows=[0],
        )








def test_validate_bjmm_parameters_accepts_valid_parameters():
    validate_bjmm_parameters(
        information_length=8,
        p=4,
        p1=4,
        ell1=2,
        ell2=2,
    )


def test_validate_bjmm_parameters_rejects_non_positive_information_length():
    with pytest.raises(
        ValueError,
        match="Information length must be positive.",
    ):
        validate_bjmm_parameters(
            information_length=0,
            p=2,
            p1=2,
            ell1=1,
            ell2=1,
        )


def test_validate_bjmm_parameters_rejects_odd_information_length():
    with pytest.raises(
        ValueError,
        match="Information length must be even.",
    ):
        validate_bjmm_parameters(
            information_length=7,
            p=2,
            p1=2,
            ell1=1,
            ell2=1,
        )


def test_validate_bjmm_parameters_rejects_negative_p():
    with pytest.raises(
        ValueError,
        match="p must not be negative.",
    ):
        validate_bjmm_parameters(
            information_length=8,
            p=-2,
            p1=2,
            ell1=1,
            ell2=1,
        )


def test_validate_bjmm_parameters_rejects_negative_p1():
    with pytest.raises(
        ValueError,
        match="p1 must not be negative.",
    ):
        validate_bjmm_parameters(
            information_length=8,
            p=2,
            p1=-2,
            ell1=1,
            ell2=1,
        )


def test_validate_bjmm_parameters_rejects_odd_p():
    with pytest.raises(
        ValueError,
        match="p must be even.",
    ):
        validate_bjmm_parameters(
            information_length=8,
            p=3,
            p1=2,
            ell1=1,
            ell2=1,
        )


def test_validate_bjmm_parameters_rejects_odd_p1():
    with pytest.raises(
        ValueError,
        match="p1 must be even.",
    ):
        validate_bjmm_parameters(
            information_length=8,
            p=2,
            p1=3,
            ell1=1,
            ell2=1,
        )


def test_validate_bjmm_parameters_rejects_excessive_p():
    with pytest.raises(
        ValueError,
        match="p must not exceed the information length.",
    ):
        validate_bjmm_parameters(
            information_length=8,
            p=10,
            p1=2,
            ell1=1,
            ell2=1,
        )


def test_validate_bjmm_parameters_rejects_excessive_p1():
    with pytest.raises(
        ValueError,
        match="p1 / 2 must not exceed half of the information length.",
    ):
        validate_bjmm_parameters(
            information_length=8,
            p=2,
            p1=10,
            ell1=1,
            ell2=1,
        )


def test_validate_bjmm_parameters_rejects_negative_ell1():
    with pytest.raises(
        ValueError,
        match="ell1 must not be negative.",
    ):
        validate_bjmm_parameters(
            information_length=8,
            p=2,
            p1=2,
            ell1=-1,
            ell2=1,
        )


def test_validate_bjmm_parameters_rejects_negative_ell2():
    with pytest.raises(
        ValueError,
        match="ell2 must not be negative.",
    ):
        validate_bjmm_parameters(
            information_length=8,
            p=2,
            p1=2,
            ell1=1,
            ell2=-1,
        )





def test_split_bjmm_merge_rows():
    first_level_rows, final_level_rows = split_bjmm_merge_rows(
        merge_rows=[0, 2, 4, 5],
        ell1=2,
        ell2=2,
    )

    assert first_level_rows == [0, 2]
    assert final_level_rows == [4, 5]


def test_split_bjmm_merge_rows_with_zero_ell1():
    first_level_rows, final_level_rows = split_bjmm_merge_rows(
        merge_rows=[1, 3],
        ell1=0,
        ell2=2,
    )

    assert first_level_rows == []
    assert final_level_rows == [1, 3]


def test_split_bjmm_merge_rows_with_zero_ell2():
    first_level_rows, final_level_rows = split_bjmm_merge_rows(
        merge_rows=[1, 3],
        ell1=2,
        ell2=0,
    )

    assert first_level_rows == [1, 3]
    assert final_level_rows == []


def test_split_bjmm_merge_rows_rejects_negative_ell1():
    with pytest.raises(
        ValueError,
        match="ell1 must not be negative.",
    ):
        split_bjmm_merge_rows(
            merge_rows=[0, 1],
            ell1=-1,
            ell2=2,
        )


def test_split_bjmm_merge_rows_rejects_negative_ell2():
    with pytest.raises(
        ValueError,
        match="ell2 must not be negative.",
    ):
        split_bjmm_merge_rows(
            merge_rows=[0, 1],
            ell1=2,
            ell2=-1,
        )


def test_split_bjmm_merge_rows_rejects_duplicates():
    with pytest.raises(
        ValueError,
        match="Merge rows must not contain duplicates.",
    ):
        split_bjmm_merge_rows(
            merge_rows=[0, 0, 1, 2],
            ell1=2,
            ell2=2,
        )


def test_split_bjmm_merge_rows_rejects_wrong_number_of_rows():
    with pytest.raises(
        ValueError,
        match="Number of merge rows must equal ell1 \\+ ell2.",
    ):
        split_bjmm_merge_rows(
            merge_rows=[0, 1, 2],
            ell1=2,
            ell2=2,
        )










def test_build_bjmm_merge_targets():
    syndrome = [1, 0, 1, 1]

    left_target, right_target, final_target = (
        build_bjmm_merge_targets(
            syndrome=syndrome,
            first_level_rows=[0, 2],
            final_level_rows=[1, 3],
        )
    )

    assert left_target == [0, 0]
    assert right_target == [1, 1]
    assert final_target == [0, 1]


def test_build_bjmm_merge_targets_with_empty_first_level_rows():
    left_target, right_target, final_target = (
        build_bjmm_merge_targets(
            syndrome=[1, 0],
            first_level_rows=[],
            final_level_rows=[0, 1],
        )
    )

    assert left_target == []
    assert right_target == []
    assert final_target == [1, 0]


def test_build_bjmm_merge_targets_with_empty_final_level_rows():
    left_target, right_target, final_target = (
        build_bjmm_merge_targets(
            syndrome=[1, 0],
            first_level_rows=[0, 1],
            final_level_rows=[],
        )
    )

    assert left_target == [0, 0]
    assert right_target == [1, 0]
    assert final_target == []