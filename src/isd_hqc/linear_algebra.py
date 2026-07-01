from typing import Iterable

Vector = list[int]


def gf2_add_vectors(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")

    return [(x + y) % 2 for x, y in zip(a, b)]


def hamming_weight(v: Iterable[int]) -> int:
    return sum(v)