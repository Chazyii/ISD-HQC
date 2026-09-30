"""
Benchmarks for the Prange ISD algorithm.
"""

import random

from experiments.benchmark import (
    BenchmarkSummary,
    run_benchmark,
)
from isd_hqc.algorithms.prange import (
    prange_decode,
)
from isd_hqc.instances.common import (
    SyndromeDecodingInstance,
)
from isd_hqc.instances.quasi_cyclic import (
    generate_quasi_cyclic_code_instance,
)
from isd_hqc.instances.random_code import (
    generate_random_code_instance,
)


def build_prange_decoder(
    max_iterations: int,
):
    """
    Build a Prange decoder compatible with the common
    benchmarking interface.
    """

    def decoder(
        instance: SyndromeDecodingInstance,
        rng: random.Random,
    ):
        return prange_decode(
            parity_check_matrix=(
                instance.parity_check_matrix
            ),
            syndrome=instance.syndrome,
            weight=instance.weight,
            max_iterations=max_iterations,
            rng=rng,
        )

    return decoder




def run_prange_random_vs_qc(
    block_size: int,
    weight: int,
    number_of_experiments: int,
    max_iterations: int,
    seed: int,
) -> tuple[
    BenchmarkSummary,
    BenchmarkSummary,
]:
    """
    Compare Prange on random and quasi-cyclic codes
    with identical dimensions and target error weight.
    """

    rows = block_size
    columns = 2 * block_size

    decoder = build_prange_decoder(
        max_iterations=max_iterations,
    )

    def random_instance_generator(
        rng: random.Random,
    ):
        return generate_random_code_instance(
            rows=rows,
            columns=columns,
            weight=weight,
            rng=rng,
        )

    def qc_instance_generator(
        rng: random.Random,
    ):
        return generate_quasi_cyclic_code_instance(
            block_size=block_size,
            weight=weight,
            rng=rng,
        )

    random_summary = run_benchmark(
        instance_generator=random_instance_generator,
        decoder=decoder,
        algorithm="Prange",
        code_type="random",
        number_of_experiments=number_of_experiments,
        seed=seed,
    )

    qc_summary = run_benchmark(
        instance_generator=qc_instance_generator,
        decoder=decoder,
        algorithm="Prange",
        code_type="quasi-cyclic",
        number_of_experiments=number_of_experiments,
        seed=seed,
    )

    return random_summary, qc_summary




def print_summary(
    summary: BenchmarkSummary,
) -> None:
    """
    Print a benchmark summary.
    """

    print(
        f"\n{summary.algorithm} / "
        f"{summary.code_type}"
    )

    print(
        "-" * 40
    )

    print(
        f"parameters: "
        f"r={summary.rows}, "
        f"n={summary.columns}, "
        f"w={summary.weight}"
    )

    print(
        f"experiments: {summary.experiments}"
    )

    print(
        f"successes: {summary.successes}"
    )

    print(
        f"failures: {summary.failures}"
    )

    print(
        f"success rate: "
        f"{summary.success_rate:.2%}"
    )

    print(
        f"total time: "
        f"{summary.total_time:.6f} s"
    )

    print(
        f"average time: "
        f"{summary.average_time:.6f} s"
    )

    print(
        f"minimum time: "
        f"{summary.minimum_time:.6f} s"
    )

    print(
        f"maximum time: "
        f"{summary.maximum_time:.6f} s"
    )



if __name__ == "__main__":
    random_summary, qc_summary = (
        run_prange_random_vs_qc(
            block_size=8,
            weight=2,
            number_of_experiments=100,
            max_iterations=1000,
            seed=42,
        )
    )

    print_summary(
        random_summary
    )

    print_summary(
        qc_summary
    )