import pytest

from isd_hqc.linear_algebra import gf2_add_vectors, hamming_weight


def test_gf2_add_vectors():
    a = [1, 0, 1]
    b = [1, 1, 0]

    result = gf2_add_vectors(a, b)

    assert result == [0, 1, 1]


def test_gf2_add_vectors_different_lengths():
    a = [1, 0, 1]
    b = [1, 1]

    with pytest.raises(ValueError):
        gf2_add_vectors(a, b)


def test_hamming_weight():
    v = [1, 0, 1, 1, 0]

    assert hamming_weight(v) == 3