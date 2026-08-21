"""
Small parameter sweep for the classical Stern ISD implementation.

This program is intended for correctness and behavioral checks on
small educational parameters.
"""

from experiments.stern_validation import run_stern_validation


def run_stern_parameter_sweep(
    parameter_sets: list[tuple[int, int]],
    number_of_experiments: int,
    rows: int,
    columns: int,
    weight: int,
    max_iterations: int,
    max_systematic_attempts: int,
    seed: int,
) -> list[dict[str, int | float]]:

    if not parameter_sets:
        raise ValueError(
            "At least one Stern parameter set must be provided."
        )

    results: list[dict[str, int | float]] = []

    for parameter_index, (p, ell) in enumerate(parameter_sets):
        validation_result = run_stern_validation(
            number_of_experiments=number_of_experiments,
            rows=rows,
            columns=columns,
            weight=weight,
            p=p,
            ell=ell,
            max_iterations=max_iterations,
            max_systematic_attempts=max_systematic_attempts,
            seed=seed + parameter_index,
        )

        results.append(
            {
                "p": p,
                "ell": ell,
                **validation_result,
            }
        )

    return results


def main() -> None:
    parameter_sets = [
        (1, 1),
        (1, 2),
        (1, 3),
    ]

    results = run_stern_parameter_sweep(
        parameter_sets=parameter_sets,
        number_of_experiments=100,
        rows=4,
        columns=8,
        weight=2,
        max_iterations=100,
        max_systematic_attempts=50,
        seed=42,
    )

    print("Stern parameter sweep")

    for result in results:
        print(
            f"p={result['p']}, "
            f"ell={result['ell']} | "
            f"success={result['success_rate']:.2%} | "
            f"avg_time={result['average_time']:.6f} s"
        )


if __name__ == "__main__":
    main()