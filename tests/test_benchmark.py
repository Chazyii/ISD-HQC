import random

import pytest

from experiments.benchmark import (
    BenchmarkResult,
    benchmark_decoder,
    run_benchmark,
    summarise_benchmark_results,
)
from isd_hqc.instances.random_code import (
    generate_random_code_instance,
)


def test_benchmark_decoder_success():
    instance = generate_random_code_instance(
        rows=2,
        columns=4,
        weight=1,
        rng=random.Random(42),
    )

    def decoder(instance, rng):
        return instance.error.copy()

    result = benchmark_decoder(
        instance=instance,
        decoder=decoder,
        algorithm="Test",
        code_type="random",
        rng=random.Random(43),
    )

    assert result.success
    assert result.algorithm == "Test"
    assert result.code_type == "random"
    assert result.rows == 2
    assert result.columns == 4
    assert result.weight == 1
    assert result.execution_time >= 0.0


def test_benchmark_decoder_failure():
    instance = generate_random_code_instance(
        rows=2,
        columns=4,
        weight=1,
        rng=random.Random(42),
    )

    def decoder(instance, rng):
        return None

    result = benchmark_decoder(
        instance=instance,
        decoder=decoder,
        algorithm="Test",
        code_type="random",
        rng=random.Random(43),
    )

    assert not result.success


def test_summarise_benchmark_results():
    results = [
        BenchmarkResult(
            code_type="random",
            algorithm="Prange",
            rows=4,
            columns=8,
            weight=2,
            success=True,
            execution_time=1.0,
        ),
        BenchmarkResult(
            code_type="random",
            algorithm="Prange",
            rows=4,
            columns=8,
            weight=2,
            success=False,
            execution_time=2.0,
        ),
    ]

    summary = summarise_benchmark_results(
        results
    )

    assert summary.experiments == 2
    assert summary.successes == 1
    assert summary.failures == 1
    assert summary.success_rate == 0.5

    assert summary.total_time == 3.0
    assert summary.average_time == 1.5
    assert summary.minimum_time == 1.0
    assert summary.maximum_time == 2.0


def test_summarise_benchmark_results_rejects_empty_results():
    with pytest.raises(
        ValueError,
        match="Benchmark results must not be empty.",
    ):
        summarise_benchmark_results([])


def test_run_benchmark():
    def generator(rng):
        return generate_random_code_instance(
            rows=2,
            columns=4,
            weight=1,
            rng=rng,
        )

    def decoder(instance, rng):
        return instance.error.copy()

    summary = run_benchmark(
        instance_generator=generator,
        decoder=decoder,
        algorithm="Test",
        code_type="random",
        number_of_experiments=5,
        seed=42,
    )

    assert summary.experiments == 5
    assert summary.successes == 5
    assert summary.failures == 0
    assert summary.success_rate == 1.0


def test_run_benchmark_rejects_zero_experiments():
    def generator(rng):
        return generate_random_code_instance(
            rows=2,
            columns=4,
            weight=1,
            rng=rng,
        )

    def decoder(instance, rng):
        return instance.error.copy()

    with pytest.raises(
        ValueError,
        match="Number of experiments must be positive.",
    ):
        run_benchmark(
            instance_generator=generator,
            decoder=decoder,
            algorithm="Test",
            code_type="random",
            number_of_experiments=0,
            seed=42,
        )