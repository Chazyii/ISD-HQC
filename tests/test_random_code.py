import random

from isd_hqc.instances.random_code import (
    generate_random_code_instance,
    generate_random_parity_check_matrix,
)
from isd_hqc.syndrome import verify_solution


def test_generate_random_parity_check_matrix_dimensions():
    matrix = generate_random_parity_check_matrix(
        rows=4,
        columns=8,
        rng=random.Random(42),
    )

    assert len(matrix) == 4
    assert all(len(row) == 8 for row in matrix)


def test_generate_random_parity_check_matrix_is_binary():
    matrix = generate_random_parity_check_matrix(
        rows=4,
        columns=8,
        rng=random.Random(42),
    )

    assert all(
        value in (0, 1)
        for row in matrix
        for value in row
    )


def test_generate_random_code_instance():
    instance = generate_random_code_instance(
        rows=4,
        columns=8,
        weight=2,
        rng=random.Random(42),
    )

    assert instance.rows == 4
    assert instance.columns == 8
    assert instance.weight == 2

    assert verify_solution(
        parity_check_matrix=instance.parity_check_matrix,
        syndrome=instance.syndrome,
        error=instance.error,
        weight=2,
    )


def test_generate_random_code_instance_is_reproducible():
    instance1 = generate_random_code_instance(
        rows=4,
        columns=8,
        weight=2,
        rng=random.Random(42),
    )

    instance2 = generate_random_code_instance(
        rows=4,
        columns=8,
        weight=2,
        rng=random.Random(42),
    )

    assert instance1 == instance2