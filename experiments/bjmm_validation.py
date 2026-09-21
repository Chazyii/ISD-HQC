import random
import time

from isd_hqc.algorithms.bjmm import bjmm_decode
from isd_hqc.linear_algebra import gf2_matrix_vector_mul
from isd_hqc.syndrome import verify_solution


def generate_random_binary_matrix(
    rows: int,
    columns: int,
    rng: random.Random,
) -> list[list[int]]:
    """
    Generate a random binary matrix.
    """

    return [
        [
            rng.randint(0, 1)
            for _ in range(columns)
        ]
        for _ in range(rows)
    ]


def generate_error_vector(
    length: int,
    weight: int,
    rng: random.Random,
) -> list[int]:
    """
    Generate a binary vector of the requested Hamming weight.
    """

    if weight < 0:
        raise ValueError(
            "Weight must not be negative."
        )

    if weight > length:
        raise ValueError(
            "Weight must not exceed vector length."
        )

    error = [0] * length

    positions = rng.sample(
        range(length),
        weight,
    )

    for position in positions:
        error[position] = 1

    return error


def run_bjmm_validation(
    number_of_experiments: int,
    rows: int,
    columns: int,
    weight: int,
    p: int,
    p1: int,
    ell1: int,
    ell2: int,
    max_iterations: int,
    seed: int | None = None,
) -> dict:
    """
    Run multiple small BJMM decoding experiments.

    The generated error and decoded error do not need to be equal.
    """

    if number_of_experiments <= 0:
        raise ValueError(
            "Number of experiments must be positive."
        )

    rng = random.Random(seed)

    successes = 0
    failures = 0
    execution_times = []

    for _ in range(number_of_experiments):
        parity_check_matrix = (
            generate_random_binary_matrix(
                rows=rows,
                columns=columns,
                rng=rng,
            )
        )

        original_error = generate_error_vector(
            length=columns,
            weight=weight,
            rng=rng,
        )

        syndrome = gf2_matrix_vector_mul(
            parity_check_matrix,
            original_error,
        )

        start_time = time.perf_counter()

        decoded_error = bjmm_decode(
            parity_check_matrix=parity_check_matrix,
            syndrome=syndrome,
            target_weight=weight,
            p=p,
            p1=p1,
            ell1=ell1,
            ell2=ell2,
            max_iterations=max_iterations,
            rng=rng,
        )

        elapsed_time = time.perf_counter() - start_time

        execution_times.append(elapsed_time)

        if decoded_error is None:
            failures += 1
            continue

        if verify_solution(
            parity_check_matrix,
            syndrome,
            decoded_error,
        ):
            successes += 1
        else:
            failures += 1

    total_time = sum(execution_times)

    return {
        "experiments": number_of_experiments,
        "successes": successes,
        "failures": failures,
        "success_rate": (
            successes / number_of_experiments
        ),
        "total_time": total_time,
        "average_time": (
            total_time / number_of_experiments
        ),
        "minimum_time": min(execution_times),
        "maximum_time": max(execution_times),
    }


if __name__ == "__main__":
    results = run_bjmm_validation(
        number_of_experiments=100,
        rows=4,
        columns=8,
        weight=2,
        p=2,
        p1=2,
        ell1=1,
        ell2=1,
        max_iterations=100,
        seed=42,
    )

    for key, value in results.items():
        print(f"{key}: {value}")