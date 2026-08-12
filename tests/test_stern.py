import pytest
import random

from isd_hqc.algorithms.stern import (
    build_partial_syndrome_list,
    compute_partial_syndrome,
    generate_weight_vectors,
    find_syndrome_collisions,
    reconstruct_candidate_error,
    stern_decode,
    validate_stern_positions,
    select_stern_partition,
    stern_decode_with_random_partition,
    construct_systematic_form,
    select_pivot_positions,
    construct_systematic_form,
    find_systematic_form,
    build_stern_information_partition,
    select_collision_rows,
    project_syndrome,
    build_stern_collision_list,
)
from isd_hqc.syndrome import verify_solution, compute_syndrome

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



def test_find_syndrome_collisions_finds_matching_pair():
    left_list = [
        ([1, 0], [1, 0]),
        ([0, 1], [0, 1]),
    ]

    right_list = [
        ([0, 1], [1, 0]),
        ([0, 0], [0, 1]),
    ]

    result = find_syndrome_collisions(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=[1, 1],
    )

    assert result == [
        (
            [1, 0],
            [1, 0],
        ),
    ]


def test_find_syndrome_collisions_finds_multiple_pairs():
    left_list = [
        ([1, 0], [1, 0]),
        ([0, 1], [0, 1]),
    ]

    right_list = [
        ([0, 1], [1, 1]),
        ([1, 0], [0, 0]),
    ]

    result = find_syndrome_collisions(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=[1, 1],
    )

    assert result == [
        (
            [1, 0],
            [1, 1],
        ),
        (
            [0, 1],
            [0, 0],
        ),
    ]


def test_find_syndrome_collisions_returns_empty_list_without_matches():
    left_list = [
        ([1, 0], [1, 0]),
    ]

    right_list = [
        ([0, 1], [0, 1]),
    ]

    result = find_syndrome_collisions(
        left_list=left_list,
        right_list=right_list,
        target_syndrome=[0, 0],
    )

    assert result == []




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



def test_stern_decode_finds_solution():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]

    syndrome = [1, 1]

    result = stern_decode(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        left_positions=[0],
        right_positions=[1],
        left_weight=1,
        right_weight=1,
    )

    assert result == [1, 1]


def test_stern_decode_returns_none_when_weights_exclude_solution():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]

    syndrome = [1, 1]

    result = stern_decode(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        left_positions=[0],
        right_positions=[1],
        left_weight=0,
        right_weight=0,
    )

    assert result is None


def test_stern_decode_returns_complete_valid_error_vector():
    parity_check_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]

    syndrome = [1, 1]

    result = stern_decode(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        left_positions=[0, 1],
        right_positions=[2, 3],
        left_weight=1,
        right_weight=1,
    )

    assert result is not None
    assert len(result) == 4
    assert sum(result) == 2
    assert verify_solution(
        parity_check_matrix,
        result,
        syndrome,
    )



def test_validate_stern_positions_accepts_valid_partition():
    validate_stern_positions(
        left_positions=[0, 2],
        right_positions=[1, 3],
        number_of_columns=4,
    )


def test_validate_stern_positions_rejects_duplicate_left_positions():
    with pytest.raises(
        ValueError,
        match="Left positions must not contain duplicates.",
    ):
        validate_stern_positions(
            left_positions=[0, 0],
            right_positions=[1, 2],
            number_of_columns=4,
        )


def test_validate_stern_positions_rejects_overlapping_positions():
    with pytest.raises(
        ValueError,
        match="Left and right positions must be disjoint.",
    ):
        validate_stern_positions(
            left_positions=[0, 1],
            right_positions=[1, 2],
            number_of_columns=4,
        )


def test_validate_stern_positions_rejects_out_of_range_position():
    with pytest.raises(
        IndexError,
        match="Right position is outside the matrix column range.",
    ):
        validate_stern_positions(
            left_positions=[0, 1],
            right_positions=[2, 4],
            number_of_columns=4,
        )



def test_select_stern_partition_even_number_of_positions():
    left_positions, right_positions = select_stern_partition(
        positions=[0, 1, 2, 3],
        seed=42,
    )

    assert len(left_positions) == 2
    assert len(right_positions) == 2

    assert sorted(
        left_positions + right_positions
    ) == [0, 1, 2, 3]


def test_select_stern_partition_odd_number_of_positions():
    left_positions, right_positions = select_stern_partition(
        positions=[0, 1, 2, 3, 4],
        seed=42,
    )

    assert len(left_positions) == 2
    assert len(right_positions) == 3

    assert sorted(
        left_positions + right_positions
    ) == [0, 1, 2, 3, 4]


def test_select_stern_partition_rejects_small_input():
    with pytest.raises(
        ValueError,
        match="At least two positions are required.",
    ):
        select_stern_partition(
            positions=[0]
        )


def test_select_stern_partition_is_reproducible():
    first = select_stern_partition(
        positions=[0, 1, 2, 3, 4, 5],
        seed=123,
    )

    second = select_stern_partition(
        positions=[0, 1, 2, 3, 4, 5],
        seed=123,
    )

    assert first == second



def test_stern_decode_with_random_partition_finds_solution():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]
    syndrome = [1, 1]

    result = stern_decode_with_random_partition(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        left_weight=1,
        right_weight=1,
        seed=42,
    )

    assert result == [1, 1]


def test_stern_decode_with_random_partition_returns_none():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]
    syndrome = [1, 1]

    result = stern_decode_with_random_partition(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        left_weight=0,
        right_weight=0,
        seed=42,
    )

    assert result is None


def test_stern_decode_with_random_partition_is_reproducible():
    parity_check_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    syndrome = [1, 1]

    first_result = stern_decode_with_random_partition(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        left_weight=1,
        right_weight=1,
        seed=123,
    )

    second_result = stern_decode_with_random_partition(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        left_weight=1,
        right_weight=1,
        seed=123,
    )

    assert first_result == second_result

    assert first_result is not None

    assert verify_solution(
        parity_check_matrix=parity_check_matrix,
        error=first_result,
        syndrome=syndrome,
        weight=2,
    )


def test_stern_decode_with_random_partition_empty_matrix():
    result = stern_decode_with_random_partition(
        parity_check_matrix=[],
        syndrome=[],
        left_weight=0,
        right_weight=0,
        seed=42,
    )

    assert result is None



def test_construct_systematic_form():
    parity_check_matrix = [
        [1, 1, 0, 1],
        [0, 1, 1, 1],
    ]
    syndrome = [1, 1]

    result = construct_systematic_form(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        pivot_positions=[0, 1],
    )

    assert result is not None

    transformed_matrix, transformed_syndrome = result

    assert transformed_matrix == [
        [1, 0, 1, 0],
        [0, 1, 1, 1],
    ]

    assert transformed_syndrome == [0, 1]


def test_construct_systematic_form_selected_columns_are_identity():
    parity_check_matrix = [
        [1, 0, 1, 1],
        [1, 1, 0, 1],
    ]
    syndrome = [1, 0]

    result = construct_systematic_form(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        pivot_positions=[2, 3],
    )

    assert result is not None

    transformed_matrix, _ = result

    selected_columns = [
        [
            row[2],
            row[3],
        ]
        for row in transformed_matrix
    ]

    assert selected_columns == [
        [1, 0],
        [0, 1],
    ]


def test_construct_systematic_form_preserves_syndrome_equation():
    parity_check_matrix = [
        [1, 1, 0, 1],
        [0, 1, 1, 1],
    ]
    error = [1, 0, 1, 0]
    syndrome = compute_syndrome(
        parity_check_matrix,
        error,
    )

    result = construct_systematic_form(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        pivot_positions=[0, 1],
    )

    assert result is not None

    transformed_matrix, transformed_syndrome = result

    assert compute_syndrome(
        transformed_matrix,
        error,
    ) == transformed_syndrome


def test_construct_systematic_form_returns_none_for_singular_submatrix():
    parity_check_matrix = [
        [1, 1, 0],
        [1, 1, 1],
    ]
    syndrome = [0, 1]

    result = construct_systematic_form(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        pivot_positions=[0, 1],
    )

    assert result is None


def test_construct_systematic_form_rejects_empty_matrix():
    with pytest.raises(
        ValueError,
        match="Parity-check matrix must not be empty.",
    ):
        construct_systematic_form(
            parity_check_matrix=[],
            syndrome=[],
            pivot_positions=[],
        )


def test_construct_systematic_form_rejects_invalid_matrix():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1],
    ]

    with pytest.raises(
        ValueError,
        match="All parity-check matrix rows must have the same length.",
    ):
        construct_systematic_form(
            parity_check_matrix=parity_check_matrix,
            syndrome=[1, 0],
            pivot_positions=[0, 1],
        )


def test_construct_systematic_form_rejects_invalid_syndrome_length():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    with pytest.raises(
        ValueError,
        match="Syndrome length must match the number of matrix rows.",
    ):
        construct_systematic_form(
            parity_check_matrix=parity_check_matrix,
            syndrome=[1],
            pivot_positions=[0, 1],
        )


def test_construct_systematic_form_rejects_invalid_pivot_count():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    with pytest.raises(
        ValueError,
        match="Number of pivot positions must match the number of matrix rows.",
    ):
        construct_systematic_form(
            parity_check_matrix=parity_check_matrix,
            syndrome=[1, 0],
            pivot_positions=[0],
        )


def test_construct_systematic_form_rejects_duplicate_pivots():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    with pytest.raises(
        ValueError,
        match="Pivot positions must not contain duplicates.",
    ):
        construct_systematic_form(
            parity_check_matrix=parity_check_matrix,
            syndrome=[1, 0],
            pivot_positions=[0, 0],
        )


def test_construct_systematic_form_rejects_out_of_range_pivot():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    with pytest.raises(
        IndexError,
        match="Pivot position is outside the matrix column range.",
    ):
        construct_systematic_form(
            parity_check_matrix=parity_check_matrix,
            syndrome=[1, 0],
            pivot_positions=[0, 3],
        )


def test_select_pivot_positions_has_correct_size():
    positions = select_pivot_positions(
        number_of_rows=3,
        number_of_columns=7,
        rng=random.Random(42),
    )

    assert len(positions) == 3


def test_select_pivot_positions_are_unique_and_sorted():
    positions = select_pivot_positions(
        number_of_rows=4,
        number_of_columns=8,
        rng=random.Random(42),
    )

    assert len(positions) == len(set(positions))
    assert positions == sorted(positions)


def test_select_pivot_positions_are_within_matrix_range():
    number_of_columns = 8

    positions = select_pivot_positions(
        number_of_rows=4,
        number_of_columns=number_of_columns,
        rng=random.Random(42),
    )

    assert all(
        0 <= position < number_of_columns
        for position in positions
    )


def test_select_pivot_positions_can_select_all_columns():
    positions = select_pivot_positions(
        number_of_rows=4,
        number_of_columns=4,
        rng=random.Random(42),
    )

    assert positions == [0, 1, 2, 3]


def test_select_pivot_positions_is_reproducible():
    first_positions = select_pivot_positions(
        number_of_rows=3,
        number_of_columns=7,
        rng=random.Random(123),
    )

    second_positions = select_pivot_positions(
        number_of_rows=3,
        number_of_columns=7,
        rng=random.Random(123),
    )

    assert first_positions == second_positions


def test_select_pivot_positions_changes_across_generator_calls():
    rng = random.Random(42)

    first_positions = select_pivot_positions(
        number_of_rows=3,
        number_of_columns=10,
        rng=rng,
    )

    second_positions = select_pivot_positions(
        number_of_rows=3,
        number_of_columns=10,
        rng=rng,
    )

    assert first_positions != second_positions


def test_select_pivot_positions_rejects_non_positive_rows():
    with pytest.raises(
        ValueError,
        match="Number of rows must be positive.",
    ):
        select_pivot_positions(
            number_of_rows=0,
            number_of_columns=5,
        )


def test_select_pivot_positions_rejects_non_positive_columns():
    with pytest.raises(
        ValueError,
        match="Number of columns must be positive.",
    ):
        select_pivot_positions(
            number_of_rows=2,
            number_of_columns=0,
        )


def test_select_pivot_positions_rejects_more_rows_than_columns():
    with pytest.raises(
        ValueError,
        match="Number of rows must not exceed the number of columns.",
    ):
        select_pivot_positions(
            number_of_rows=6,
            number_of_columns=5,
        )



def test_find_systematic_form_returns_valid_result():
    parity_check_matrix = [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
    ]
    syndrome = [1, 0]

    result = find_systematic_form(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        max_attempts=10,
        rng=random.Random(42),
    )

    assert result is not None

    transformed_matrix, transformed_syndrome, pivot_positions = result

    assert len(transformed_matrix) == 2
    assert all(len(row) == 4 for row in transformed_matrix)
    assert len(transformed_syndrome) == 2
    assert len(pivot_positions) == 2

    selected_columns = [
        [
            row[position]
            for position in pivot_positions
        ]
        for row in transformed_matrix
    ]

    assert selected_columns == [
        [1, 0],
        [0, 1],
    ]


def test_find_systematic_form_retries_after_singular_selection(
    monkeypatch,
):
    parity_check_matrix = [
        [1, 1, 0],
        [1, 1, 1],
    ]
    syndrome = [0, 1]

    selected_positions = [
        [0, 1],
        [0, 2],
    ]

    def controlled_select_pivot_positions(
        number_of_rows,
        number_of_columns,
        rng,
    ):
        return selected_positions.pop(0)

    monkeypatch.setattr(
        "isd_hqc.algorithms.stern.select_pivot_positions",
        controlled_select_pivot_positions,
    )

    result = find_systematic_form(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        max_attempts=2,
        rng=random.Random(42),
    )

    assert result is not None

    transformed_matrix, transformed_syndrome, pivot_positions = result

    assert pivot_positions == [0, 2]

    assert transformed_matrix == [
        [1, 0, 0],
        [0, 0, 1],
    ]

    assert transformed_syndrome == [1, 1]


def test_find_systematic_form_returns_none_after_all_attempts(
    monkeypatch,
):
    parity_check_matrix = [
        [1, 1, 0],
        [1, 1, 1],
    ]
    syndrome = [0, 1]

    def singular_pivot_selection(
        number_of_rows,
        number_of_columns,
        rng,
    ):
        return [0, 1]

    monkeypatch.setattr(
        "isd_hqc.algorithms.stern.select_pivot_positions",
        singular_pivot_selection,
    )

    result = find_systematic_form(
        parity_check_matrix=parity_check_matrix,
        syndrome=syndrome,
        max_attempts=3,
        rng=random.Random(42),
    )

    assert result is None


def test_find_systematic_form_rejects_non_positive_attempt_count():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]

    with pytest.raises(
        ValueError,
        match="Maximum number of attempts must be positive.",
    ):
        find_systematic_form(
            parity_check_matrix=parity_check_matrix,
            syndrome=[1, 0],
            max_attempts=0,
        )


def test_find_systematic_form_rejects_empty_matrix():
    with pytest.raises(
        ValueError,
        match="Parity-check matrix must not be empty.",
    ):
        find_systematic_form(
            parity_check_matrix=[],
            syndrome=[],
            max_attempts=10,
        )


def test_find_systematic_form_rejects_invalid_matrix():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1],
    ]

    with pytest.raises(
        ValueError,
        match="All parity-check matrix rows must have the same length.",
    ):
        find_systematic_form(
            parity_check_matrix=parity_check_matrix,
            syndrome=[1, 0],
            max_attempts=10,
        )


def test_find_systematic_form_rejects_invalid_syndrome_length():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    with pytest.raises(
        ValueError,
        match="Syndrome length must match the number of matrix rows.",
    ):
        find_systematic_form(
            parity_check_matrix=parity_check_matrix,
            syndrome=[1],
            max_attempts=10,
        )




def test_build_stern_information_partition_constructs_complement():
    information_set, left_positions, right_positions = (
        build_stern_information_partition(
            number_of_columns=7,
            pivot_positions=[1, 4, 6],
            rng=random.Random(42),
        )
    )

    assert information_set == [0, 2, 3, 5]

    assert sorted(
        left_positions + right_positions
    ) == information_set


def test_build_stern_information_partition_is_disjoint_from_pivots():
    pivot_positions = [0, 3]

    information_set, left_positions, right_positions = (
        build_stern_information_partition(
            number_of_columns=6,
            pivot_positions=pivot_positions,
            rng=random.Random(42),
        )
    )

    assert set(information_set).isdisjoint(
        pivot_positions
    )

    assert set(left_positions).isdisjoint(
        pivot_positions
    )

    assert set(right_positions).isdisjoint(
        pivot_positions
    )


def test_build_stern_information_partition_splits_information_set():
    information_set, left_positions, right_positions = (
        build_stern_information_partition(
            number_of_columns=8,
            pivot_positions=[0, 1, 2],
            rng=random.Random(42),
        )
    )

    assert len(information_set) == 5
    assert len(left_positions) == 2
    assert len(right_positions) == 3

    assert set(left_positions).isdisjoint(
        right_positions
    )


def test_build_stern_information_partition_is_reproducible():
    first = build_stern_information_partition(
        number_of_columns=8,
        pivot_positions=[0, 1, 2],
        rng=random.Random(123),
    )

    second = build_stern_information_partition(
        number_of_columns=8,
        pivot_positions=[0, 1, 2],
        rng=random.Random(123),
    )

    assert first == second


def test_build_stern_information_partition_rejects_duplicate_pivots():
    with pytest.raises(
        ValueError,
        match="Pivot positions must not contain duplicates.",
    ):
        build_stern_information_partition(
            number_of_columns=6,
            pivot_positions=[0, 0],
        )


def test_build_stern_information_partition_rejects_invalid_pivot():
    with pytest.raises(
        IndexError,
        match="Pivot position is outside the matrix column range.",
    ):
        build_stern_information_partition(
            number_of_columns=5,
            pivot_positions=[0, 5],
        )


def test_build_stern_information_partition_requires_two_information_positions():
    with pytest.raises(
        ValueError,
        match="Information set must contain at least two positions.",
    ):
        build_stern_information_partition(
            number_of_columns=4,
            pivot_positions=[0, 1, 2],
        )




def test_select_collision_rows_has_correct_size():
    rows = select_collision_rows(
        number_of_rows=6,
        ell=3,
        rng=random.Random(42),
    )

    assert len(rows) == 3


def test_select_collision_rows_are_unique_and_sorted():
    rows = select_collision_rows(
        number_of_rows=8,
        ell=4,
        rng=random.Random(42),
    )

    assert len(rows) == len(set(rows))
    assert rows == sorted(rows)


def test_select_collision_rows_are_within_range():
    number_of_rows = 7

    rows = select_collision_rows(
        number_of_rows=number_of_rows,
        ell=3,
        rng=random.Random(42),
    )

    assert all(
        0 <= row < number_of_rows
        for row in rows
    )


def test_select_collision_rows_can_select_all_rows():
    rows = select_collision_rows(
        number_of_rows=4,
        ell=4,
        rng=random.Random(42),
    )

    assert rows == [0, 1, 2, 3]


def test_select_collision_rows_is_reproducible():
    first = select_collision_rows(
        number_of_rows=8,
        ell=3,
        rng=random.Random(123),
    )

    second = select_collision_rows(
        number_of_rows=8,
        ell=3,
        rng=random.Random(123),
    )

    assert first == second


def test_select_collision_rows_rejects_non_positive_ell():
    with pytest.raises(
        ValueError,
        match="Collision parameter ell must be positive.",
    ):
        select_collision_rows(
            number_of_rows=5,
            ell=0,
        )


def test_select_collision_rows_rejects_ell_greater_than_rows():
    with pytest.raises(
        ValueError,
        match="Collision parameter ell must not exceed the number of rows.",
    ):
        select_collision_rows(
            number_of_rows=4,
            ell=5,
        )


def test_select_collision_rows_rejects_non_positive_row_count():
    with pytest.raises(
        ValueError,
        match="Number of rows must be positive.",
    ):
        select_collision_rows(
            number_of_rows=0,
            ell=1,
        )





def test_project_syndrome():
    syndrome = [1, 0, 1, 1, 0]

    result = project_syndrome(
        syndrome=syndrome,
        collision_rows=[1, 3],
    )

    assert result == [0, 1]


def test_project_syndrome_preserves_requested_order():
    syndrome = [1, 0, 1, 1]

    result = project_syndrome(
        syndrome=syndrome,
        collision_rows=[3, 0, 2],
    )

    assert result == [1, 1, 1]


def test_project_syndrome_single_row():
    result = project_syndrome(
        syndrome=[0, 1, 0],
        collision_rows=[1],
    )

    assert result == [1]


def test_project_syndrome_empty_selection():
    result = project_syndrome(
        syndrome=[1, 0, 1],
        collision_rows=[],
    )

    assert result == []


def test_project_syndrome_rejects_duplicate_rows():
    with pytest.raises(
        ValueError,
        match="Collision rows must not contain duplicates.",
    ):
        project_syndrome(
            syndrome=[1, 0, 1],
            collision_rows=[0, 0],
        )


def test_project_syndrome_rejects_out_of_range_row():
    with pytest.raises(
        IndexError,
        match="Collision row is outside the syndrome range.",
    ):
        project_syndrome(
            syndrome=[1, 0, 1],
            collision_rows=[3],
        )


def test_project_syndrome_rejects_negative_row():
    with pytest.raises(
        IndexError,
        match="Collision row is outside the syndrome range.",
    ):
        project_syndrome(
            syndrome=[1, 0, 1],
            collision_rows=[-1],
        )




def test_build_stern_collision_list():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]

    result = build_stern_collision_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2],
        weight=1,
        collision_rows=[0, 2],
    )

    assert result == [
        ([1, 1], [1, 0, 0]),
        ([0, 1], [0, 1, 0]),
        ([1, 0], [0, 0, 1]),
    ]


def test_build_stern_collision_list_has_correct_candidate_count():
    parity_check_matrix = [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
    ]

    result = build_stern_collision_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2, 3],
        weight=2,
        collision_rows=[0, 1],
    )

    assert len(result) == 6


def test_build_stern_collision_list_preserves_error_weight():
    parity_check_matrix = [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
    ]

    result = build_stern_collision_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1, 2, 3],
        weight=2,
        collision_rows=[0, 2],
    )

    for projected_syndrome, partial_error in result:
        assert len(projected_syndrome) == 2
        assert sum(partial_error) == 2


def test_build_stern_collision_list_zero_weight():
    parity_check_matrix = [
        [1, 0],
        [0, 1],
    ]

    result = build_stern_collision_list(
        parity_check_matrix=parity_check_matrix,
        positions=[0, 1],
        weight=0,
        collision_rows=[0],
    )

    assert result == [
        ([0], [0, 0]),
    ]