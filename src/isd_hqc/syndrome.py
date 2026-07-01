import random

from isd_hqc.linear_algebra import Matrix, Vector, gf2_matrix_vector_mul, hamming_weight


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


def generate_random_parity_check_matrix(rows: int, columns: int) -> Matrix:
   
    if rows <= 0:
        raise ValueError("Number of rows must be positive.")

    if columns <= 0:
        raise ValueError("Number of columns must be positive.")

    return [
        [random.randint(0, 1) for _ in range(columns)]
        for _ in range(rows)
    ]

def compute_syndrome(parity_check_matrix: Matrix, error: Vector) -> Vector:
   
    return gf2_matrix_vector_mul(parity_check_matrix, error)


def verify_solution(
    parity_check_matrix: Matrix,
    error: Vector,
    syndrome: Vector,
    weight: int | None = None,
) -> bool:
    
    if weight is not None and hamming_weight(error) != weight:
        return False

    return compute_syndrome(parity_check_matrix, error) == syndrome