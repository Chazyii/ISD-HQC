"""
Common benchmarking infrastructure for ISD experiments.
"""

from dataclasses import dataclass
import random
import time
from typing import Callable

from isd_hqc.instances.common import (
    SyndromeDecodingInstance,
)
from isd_hqc.instances.quasi_cyclic import (
    generate_quasi_cyclic_code_instance,
)
from isd_hqc.instances.random_code import (
    generate_random_code_instance,
)
from isd_hqc.linear_algebra import Vector
from isd_hqc.syndrome import verify_solution


Decoder = Callable[
    [SyndromeDecodingInstance, random.Random],
    Vector | None,
]


@dataclass
class BenchmarkResult:
    """
    Result of one decoder run on one decoding instance.
    """

    code_type: str
    algorithm: str

    rows: int
    columns: int
    weight: int

    success: bool
    execution_time: float


@dataclass
class BenchmarkSummary:
    """
    Aggregated results of multiple benchmark runs.
    """

    code_type: str
    algorithm: str

    rows: int
    columns: int
    weight: int

    experiments: int
    successes: int
    failures: int

    success_rate: float

    total_time: float
    average_time: float
    minimum_time: float
    maximum_time: float





def benchmark_decoder(
    instance: SyndromeDecodingInstance,
    decoder: Decoder,
    algorithm: str,
    code_type: str,
    rng: random.Random,
) -> BenchmarkResult:
    """
    Benchmark one decoder on one syndrome-decoding instance.
    """

    start_time = time.perf_counter()

    decoded_error = decoder(
        instance,
        rng,
    )

    execution_time = (
        time.perf_counter()
        - start_time
    )

    success = False

    if decoded_error is not None:
        success = verify_solution(
            parity_check_matrix=(
                instance.parity_check_matrix
            ),
            syndrome=instance.syndrome,
            error=decoded_error,
            weight=instance.weight,
        )

    return BenchmarkResult(
        code_type=code_type,
        algorithm=algorithm,
        rows=instance.rows,
        columns=instance.columns,
        weight=instance.weight,
        success=success,
        execution_time=execution_time,
    )





def summarise_benchmark_results(
    results: list[BenchmarkResult],
) -> BenchmarkSummary:
    """
    Aggregate benchmark results for one experimental setup.
    """

    if not results:
        raise ValueError(
            "Benchmark results must not be empty."
        )

    first = results[0]

    for result in results:
        if result.code_type != first.code_type:
            raise ValueError(
                "All results must use the same code type."
            )

        if result.algorithm != first.algorithm:
            raise ValueError(
                "All results must use the same algorithm."
            )

        if (
            result.rows != first.rows
            or result.columns != first.columns
            or result.weight != first.weight
        ):
            raise ValueError(
                "All results must use the same code parameters."
            )

    successes = sum(
        result.success
        for result in results
    )

    experiments = len(results)
    failures = experiments - successes

    execution_times = [
        result.execution_time
        for result in results
    ]

    total_time = sum(execution_times)

    return BenchmarkSummary(
        code_type=first.code_type,
        algorithm=first.algorithm,
        rows=first.rows,
        columns=first.columns,
        weight=first.weight,
        experiments=experiments,
        successes=successes,
        failures=failures,
        success_rate=successes / experiments,
        total_time=total_time,
        average_time=total_time / experiments,
        minimum_time=min(execution_times),
        maximum_time=max(execution_times),
    )






def run_benchmark(
    instance_generator: Callable[
        [random.Random],
        SyndromeDecodingInstance,
    ],
    decoder: Decoder,
    algorithm: str,
    code_type: str,
    number_of_experiments: int,
    seed: int,
) -> BenchmarkSummary:
    """
    Run a decoder on multiple independently generated instances.
    """

    if number_of_experiments <= 0:
        raise ValueError(
            "Number of experiments must be positive."
        )

    instance_rng = random.Random(seed)

    decoder_rng = random.Random(
        seed + 1
    )

    results = []

    for _ in range(number_of_experiments):
        instance = instance_generator(
            instance_rng
        )

        result = benchmark_decoder(
            instance=instance,
            decoder=decoder,
            algorithm=algorithm,
            code_type=code_type,
            rng=decoder_rng,
        )

        results.append(result)

    return summarise_benchmark_results(
        results
    )