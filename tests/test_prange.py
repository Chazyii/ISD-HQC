import pytest

from isd_hqc.algorithms.prange import (
    construct_induced_system,
    select_information_set,
    solve_induced_system,
    reconstruct_candidate_error,
    verify_candidate,
)

def test_information_set_has_correct_size():
    information_set = select_information_set(10, 4)

    assert len(information_set) == 4


def test_information_set_is_sorted():
    information_set = select_information_set(20, 7)

    assert information_set == sorted(information_set)


def test_information_set_contains_unique_indices():
    information_set = select_information_set(20, 7)

    assert len(information_set) == len(set(information_set))


def test_information_set_indices_are_in_range():
    information_set = select_information_set(20, 7)

    assert all(0 <= index < 20 for index in information_set)


def test_information_set_invalid_dimension():
    with pytest.raises(ValueError):
        select_information_set(5, 6)


def test_information_set_invalid_length():
    with pytest.raises(ValueError):
        select_information_set(0, 1)


def test_information_set_negative_dimension():
    with pytest.raises(ValueError):
        select_information_set(10, -1)





def test_construct_induced_system():
    parity_check_matrix = [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
    ]
    syndrome = [1, 0]
    information_set = [0, 1]

    induced_matrix, induced_syndrome, complement = construct_induced_system(
        parity_check_matrix,
        syndrome,
        information_set,
    )

    assert induced_matrix == [
        [1, 1],
        [1, 0],
    ]
    assert induced_syndrome == [1, 0]
    assert complement == [2, 3]


def test_construct_induced_system_invalid_information_set_size():
    parity_check_matrix = [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
    ]

    with pytest.raises(ValueError):
        construct_induced_system(
            parity_check_matrix,
            [1, 0],
            [0],
        )


def test_construct_induced_system_invalid_syndrome_size():
    parity_check_matrix = [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
    ]

    with pytest.raises(ValueError):
        construct_induced_system(
            parity_check_matrix,
            [1],
            [0, 1],
        )



#solve induced system

def test_solve_induced_system():
    induced_matrix = [
        [1, 1],
        [1, 0],
    ]
    syndrome = [1, 0]

    solution = solve_induced_system(
        induced_matrix,
        syndrome,
    )

    assert solution == [0, 1]


def test_solve_induced_system_requires_row_swap():
    induced_matrix = [
        [0, 1],
        [1, 1],
    ]
    syndrome = [1, 0]

    solution = solve_induced_system(
        induced_matrix,
        syndrome,
    )

    assert solution == [1, 1]


def test_solve_induced_system_singular_matrix():
    induced_matrix = [
        [1, 1],
        [1, 1],
    ]
    syndrome = [0, 0]

    solution = solve_induced_system(
        induced_matrix,
        syndrome,
    )

    assert solution is None



#reconstruct candidate error

def test_reconstruct_candidate_error():
    partial_error = [1, 0, 1]
    complement = [0, 2, 4]

    candidate_error = reconstruct_candidate_error(
        length=5,
        partial_error=partial_error,
        complement=complement,
    )

    assert candidate_error == [1, 0, 0, 0, 1]


def test_reconstruct_candidate_error_unsorted_complement():
    partial_error = [1, 0, 1]
    complement = [4, 0, 2]

    candidate_error = reconstruct_candidate_error(
        length=5,
        partial_error=partial_error,
        complement=complement,
    )

    assert candidate_error == [0, 0, 1, 0, 1]


def test_reconstruct_candidate_error_invalid_length():
    with pytest.raises(ValueError):
        reconstruct_candidate_error(
            length=0,
            partial_error=[],
            complement=[],
        )


def test_reconstruct_candidate_error_invalid_partial_error_length():
    partial_error = [1, 0]
    complement = [0, 2, 4]

    with pytest.raises(ValueError):
        reconstruct_candidate_error(
            length=5,
            partial_error=partial_error,
            complement=complement,
        )


def test_reconstruct_candidate_error_duplicate_indices():
    partial_error = [1, 0, 1]
    complement = [0, 2, 2]

    with pytest.raises(ValueError):
        reconstruct_candidate_error(
            length=5,
            partial_error=partial_error,
            complement=complement,
        )


def test_reconstruct_candidate_error_index_out_of_range():
    partial_error = [1, 0, 1]
    complement = [0, 2, 5]

    with pytest.raises(ValueError):
        reconstruct_candidate_error(
            length=5,
            partial_error=partial_error,
            complement=complement,
        )


#verify candidate
def test_verify_candidate_valid():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    candidate_error = [1, 0, 1]
    syndrome = [0, 1]

    result = verify_candidate(
        parity_check_matrix=parity_check_matrix,
        candidate_error=candidate_error,
        syndrome=syndrome,
        weight=2,
    )

    assert result


def test_verify_candidate_invalid_syndrome():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    candidate_error = [1, 0, 1]
    syndrome = [1, 1]

    result = verify_candidate(
        parity_check_matrix=parity_check_matrix,
        candidate_error=candidate_error,
        syndrome=syndrome,
        weight=2,
    )

    assert not result


def test_verify_candidate_invalid_weight():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    candidate_error = [1, 0, 1]
    syndrome = [0, 1]

    result = verify_candidate(
        parity_check_matrix=parity_check_matrix,
        candidate_error=candidate_error,
        syndrome=syndrome,
        weight=1,
    )

    assert not result