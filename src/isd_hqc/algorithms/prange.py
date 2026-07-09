"""
Implementation of the Prange ISD algorithm.
"""

import random


def select_information_set(length: int, dimension: int) -> list[int]:

    if length <= 0:
        raise ValueError("Length must be positive.")

    if dimension <= 0:
        raise ValueError("Dimension must be positive.")

    if dimension > length:
        raise ValueError("Dimension cannot exceed code length.")

    return sorted(random.sample(range(length), dimension))