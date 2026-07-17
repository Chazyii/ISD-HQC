"""
Experimental validation of the Prange ISD implementation.
"""

import random

from isd_hqc.algorithms.prange import prange_decode
from isd_hqc.syndrome import (
    generate_sd_instance,
    verify_solution,
)


def run_prange_validation(
    number_of_experiments: int,
    rows: int,
    columns: int,
    weight: int,
    max_iterations: int,
    seed: int,
) -> dict[str, int | float]:

    if number_of_experiments <= 0:
        raise ValueError("Number of experiments must be positive.")

    random.seed(seed)

    successful = 0
    failed = 0

    for _ in range(number_of_experiments):
        parity_check_matrix, _, syndrome = generate_sd_instance(
            rows=rows,
            columns=columns,
            weight=weight,
        )

        decoded_error = prange_decode(
            parity_check_matrix=parity_check_matrix,
            syndrome=syndrome,
            weight=weight,
            max_iterations=max_iterations,
        )

        if decoded_error is None:
            failed += 1
            continue

        if verify_solution(
            parity_check_matrix=parity_check_matrix,
            error=decoded_error,
            syndrome=syndrome,
            weight=weight,
        ):
            successful += 1
        else:
            failed += 1

    success_rate = successful / number_of_experiments

    return {
        "experiments": number_of_experiments,
        "successful": successful,
        "failed": failed,
        "success_rate": success_rate,
    }


def main() -> None:
    results = run_prange_validation(
        number_of_experiments=100,
        rows=4,
        columns=8,
        weight=1,
        max_iterations=1000,
        seed=42,
    )

    print("Prange validation results")
    print("-------------------------")
    print(f"Experiments: {results['experiments']}")
    print(f"Successful:  {results['successful']}")
    print(f"Failed:      {results['failed']}")
    print(f"Success rate: {results['success_rate']:.2%}")


if __name__ == "__main__":
    main()