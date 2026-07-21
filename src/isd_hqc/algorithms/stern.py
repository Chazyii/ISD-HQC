"""
Implementation of the Stern ISD algorithm.
"""
from itertools import combinations


def generate_weight_vectors(
    length: int,
    weight: int,
) -> list[list[int]]:
    """
    Generate all binary vectors of a given length and Hamming weight.

    Args:
        length:
            Length of each generated binary vector.
        weight:
            Required Hamming weight of each vector.

    Returns:
        A list containing all binary vectors of the specified length
        and Hamming weight.

    Raises:
        ValueError:
            If length is negative, weight is negative, or weight
            is greater than length.
    """

    if length < 0:
        raise ValueError("Length must be non-negative.")

    if weight < 0:
        raise ValueError("Weight must be non-negative.")

    if weight > length:
        raise ValueError("Weight cannot be greater than vector length.")

    vectors: list[list[int]] = []

    for positions in combinations(range(length), weight):
        vector = [0] * length

        for position in positions:
            vector[position] = 1

        vectors.append(vector)

    return vectors