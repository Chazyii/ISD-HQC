import pytest

from isd_hqc.linear_algebra import hamming_weight
from isd_hqc.syndrome import (
    compute_syndrome,
    generate_random_error,
    generate_random_parity_check_matrix,
    verify_solution,
    generate_sd_instance,
)

def test_generate_random_error_has_correct_length():
    error = generate_random_error(length=10, weight=3)

    assert len(error) == 10


def test_generate_random_error_has_correct_weight():
    error = generate_random_error(length=10, weight=3)

    assert hamming_weight(error) == 3


def test_generate_random_error_contains_only_binary_values():
    error = generate_random_error(length=10, weight=3)

    assert set(error).issubset({0, 1})


def test_generate_random_error_invalid_weight():
    with pytest.raises(ValueError):
        generate_random_error(length=5, weight=6)

def test_generate_random_parity_check_matrix_has_correct_size():
    matrix = generate_random_parity_check_matrix(rows=3, columns=5)

    assert len(matrix) == 3
    assert all(len(row) == 5 for row in matrix)


def test_generate_random_parity_check_matrix_contains_only_binary_values():
    matrix = generate_random_parity_check_matrix(rows=3, columns=5)

    assert all(value in {0, 1} for row in matrix for value in row)


def test_generate_random_parity_check_matrix_invalid_rows():
    with pytest.raises(ValueError):
        generate_random_parity_check_matrix(rows=0, columns=5)


def test_generate_random_parity_check_matrix_invalid_columns():
    with pytest.raises(ValueError):
        generate_random_parity_check_matrix(rows=3, columns=0)


def test_compute_syndrome():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    error = [1, 0, 1]

    syndrome = compute_syndrome(parity_check_matrix, error)

    assert syndrome == [0, 1]


def test_verify_solution_valid():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    error = [1, 0, 1]
    syndrome = [0, 1]

    assert verify_solution(parity_check_matrix, error, syndrome)


def test_verify_solution_invalid_syndrome():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    error = [1, 0, 1]
    syndrome = [1, 1]

    assert not verify_solution(parity_check_matrix, error, syndrome)


def test_verify_solution_valid_with_weight():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    error = [1, 0, 1]
    syndrome = [0, 1]

    assert verify_solution(parity_check_matrix, error, syndrome, weight=2)


def test_verify_solution_invalid_weight():
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    error = [1, 0, 1]
    syndrome = [0, 1]

    assert not verify_solution(parity_check_matrix, error, syndrome, weight=1)


#generate instatnce test
def test_generate_sd_instance_has_correct_dimensions():
    parity_check_matrix, error, syndrome = generate_sd_instance(
        rows=3,
        columns=5,
        weight=2,
    )

    assert len(parity_check_matrix) == 3
    assert all(len(row) == 5 for row in parity_check_matrix)
    assert len(error) == 5
    assert len(syndrome) == 3


def test_generate_sd_instance_has_correct_error_weight():
    _, error, _ = generate_sd_instance(
        rows=3,
        columns=5,
        weight=2,
    )

    assert hamming_weight(error) == 2


def test_generate_sd_instance_is_valid():
    parity_check_matrix, error, syndrome = generate_sd_instance(
        rows=3,
        columns=5,
        weight=2,
    )

    assert verify_solution(
        parity_check_matrix=parity_check_matrix,
        error=error,
        syndrome=syndrome,
        weight=2,
    )