import random

from isd_hqc.linear_algebra import Vector, hamming_weight


def generate_random_error(length: int, weight: int) -> Vector:
    if length <= 0:
        raise ValueError("Length must be positive.")

    if weight < 0:
        raise ValueError("Weight must not be negative.")

    if weight > length:
        raise ValueError("Weight must not exceed vector length.")

    error = [0] * length
    positions = random.sample(range(length), weight)

    for position in positions:
        error[position] = 1

    return error