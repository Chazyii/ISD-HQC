"""
Experimental validation of the Prange ISD implementation.
"""

import random
import time

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
    execution_times: list[float] = []

    for _ in range(number_of_experiments):
        parity_check_matrix, _, syndrome = generate_sd_instance(
            rows=rows,
            columns=columns,
            weight=weight,
        )

        start_time = time.perf_counter()

        decoded_error = prange_decode(
            parity_check_matrix=parity_check_matrix,
            syndrome=syndrome,
            weight=weight,
            max_iterations=max_iterations,
        )

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        execution_times.append(elapsed_time)

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

    total_time = sum(execution_times)
    average_time = total_time / number_of_experiments
    minimum_time = min(execution_times)
    maximum_time = max(execution_times)

    return {
        "experiments": number_of_experiments,
        "successful": successful,
        "failed": failed,
        "success_rate": success_rate,
        "total_time": total_time,
        "average_time": average_time,
        "minimum_time": minimum_time,
        "maximum_time": maximum_time,
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
    print(f"Experiments:  {results['experiments']}")
    print(f"Successful:   {results['successful']}")
    print(f"Failed:       {results['failed']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print()

    print("Execution time")
    print(f"Total time:   {results['total_time']:.6f} s")
    print(f"Average time: {results['average_time']:.6f} s")
    print(f"Minimum time: {results['minimum_time']:.6f} s")
    print(f"Maximum time: {results['maximum_time']:.6f} s")


if __name__ == "__main__":
    main()