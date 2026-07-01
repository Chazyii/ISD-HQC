import pytest

from isd_hqc.linear_algebra import hamming_weight
from isd_hqc.syndrome import generate_random_error


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