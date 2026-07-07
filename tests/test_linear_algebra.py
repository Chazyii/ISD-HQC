import pytest

from isd_hqc.linear_algebra import (
    gf2_add_vectors,
    gf2_matrix_vector_mul,
    hamming_weight,
    transpose_matrix,
    identity_matrix,
    gf2_matrix_matrix_mul,
    gf2_row_echelon_form,
)


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

def test_gf2_matrix_vector_mul():
    matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    vector = [1, 0, 1]

    result = gf2_matrix_vector_mul(matrix, vector)

    assert result == [0, 1]


def test_gf2_matrix_vector_mul_empty_matrix():
    with pytest.raises(ValueError):
        gf2_matrix_vector_mul([], [1, 0, 1])


def test_gf2_matrix_vector_mul_invalid_dimensions():
    matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    vector = [1, 0]

    with pytest.raises(ValueError):
        gf2_matrix_vector_mul(matrix, vector)


def test_transpose_matrix():
    matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    transposed = transpose_matrix(matrix)

    assert transposed == [
        [1, 0],
        [0, 1],
        [1, 1],
    ]


def test_transpose_empty_matrix():
    with pytest.raises(ValueError):
        transpose_matrix([])


def test_transpose_invalid_matrix():
    matrix = [
        [1, 0],
        [1],
    ]

    with pytest.raises(ValueError):
        transpose_matrix(matrix)


def test_identity_matrix():
    result = identity_matrix(3)

    assert result == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]


def test_identity_matrix_invalid_size():
    with pytest.raises(ValueError):
        identity_matrix(0)


def test_gf2_matrix_matrix_mul():
    matrix_a = [
        [1, 0],
        [1, 1],
    ]

    matrix_b = [
        [1, 1],
        [0, 1],
    ]

    result = gf2_matrix_matrix_mul(matrix_a, matrix_b)

    assert result == [
        [1, 1],
        [1, 0],
    ]


def test_gf2_matrix_matrix_mul_invalid_dimensions():
    matrix_a = [
        [1, 0],
    ]

    matrix_b = [
        [1, 1],
        [0, 1],
        [1, 0],
    ]

    with pytest.raises(ValueError):
        gf2_matrix_matrix_mul(matrix_a, matrix_b)


def test_gf2_matrix_matrix_mul_empty_matrix():
    with pytest.raises(ValueError):
        gf2_matrix_matrix_mul([], [[1]])


def test_gf2_row_echelon_form():
    matrix = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]

    result = gf2_row_echelon_form(matrix)

    assert result == [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]


def test_gf2_row_echelon_form_requires_row_swap():
    matrix = [
        [0, 1, 1],
        [1, 0, 1],
    ]

    result = gf2_row_echelon_form(matrix)

    assert result == [
        [1, 0, 1],
        [0, 1, 1],
    ]


def test_gf2_row_echelon_form_empty_matrix():
    with pytest.raises(ValueError):
        gf2_row_echelon_form([])


def test_gf2_row_echelon_form_invalid_matrix():
    matrix = [
        [1, 0],
        [1],
    ]

    with pytest.raises(ValueError):
        gf2_row_echelon_form(matrix)