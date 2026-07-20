"""
Experimental validation of the Prange ISD implementation.
"""

import csv
import random
import time
from pathlib import Path

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
) -> dict:

    if number_of_experiments <= 0:
        raise ValueError("Number of experiments must be positive.")

    random.seed(seed)

    successful = 0
    failed = 0
    execution_times: list[float] = []
    experiment_results: list[dict] = []

    for experiment_id in range(1, number_of_experiments + 1):
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

        is_successful = False

        if decoded_error is not None:
            is_successful = verify_solution(
                parity_check_matrix=parity_check_matrix,
                error=decoded_error,
                syndrome=syndrome,
                weight=weight,
            )

        if is_successful:
            successful += 1
        else:
            failed += 1

        experiment_results.append(
            {
                "experiment_id": experiment_id,
                "rows": rows,
                "columns": columns,
                "weight": weight,
                "max_iterations": max_iterations,
                "seed": seed,
                "success": is_successful,
                "execution_time": elapsed_time,
            }
        )

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
        "experiment_results": experiment_results,
    }


def save_results_to_csv(
    experiment_results: list[dict],
    output_path: str | Path,
) -> None:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "experiment_id",
        "rows",
        "columns",
        "weight",
        "max_iterations",
        "seed",
        "success",
        "execution_time",
    ]

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(experiment_results)


def main() -> None:
    results = run_prange_validation(
        number_of_experiments=100,
        rows=4,
        columns=8,
        weight=1,
        max_iterations=1000,
        seed=42,
    )

    save_results_to_csv(
        experiment_results=results["experiment_results"],
        output_path="results/prange_validation.csv",
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
    print()

    print(
        "Detailed results saved to: "
        "results/prange_validation.csv"
    )


if __name__ == "__main__":
    main()