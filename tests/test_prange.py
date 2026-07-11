import pytest

from isd_hqc.algorithms.prange import (
    construct_induced_system,
    select_information_set,
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