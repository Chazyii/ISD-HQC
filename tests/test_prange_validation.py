import pytest

from experiments.prange_validation import run_prange_validation


def test_run_prange_validation(monkeypatch):
    parity_check_matrix = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    original_error = [1, 0, 0]
    syndrome = [1, 0]

    def fixed_generate_sd_instance(rows, columns, weight):
        return parity_check_matrix, original_error, syndrome

    decoded_results = [
        [1, 0, 0],
        None,
        [1, 0, 0],
    ]

    def controlled_prange_decode(
        parity_check_matrix,
        syndrome,
        weight,
        max_iterations,
    ):
        return decoded_results.pop(0)

    monkeypatch.setattr(
        "experiments.prange_validation.generate_sd_instance",
        fixed_generate_sd_instance,
    )

    monkeypatch.setattr(
        "experiments.prange_validation.prange_decode",
        controlled_prange_decode,
    )

    results = run_prange_validation(
        number_of_experiments=3,
        rows=2,
        columns=3,
        weight=1,
        max_iterations=100,
        seed=42,
    )

    assert results["experiments"] == 3
    assert results["successful"] == 2
    assert results["failed"] == 1
    assert results["success_rate"] == pytest.approx(2 / 3)

    assert results["total_time"] >= 0
    assert results["average_time"] >= 0
    assert results["minimum_time"] >= 0
    assert results["maximum_time"] >= 0

    assert results["minimum_time"] <= results["maximum_time"]

    assert results["average_time"] == pytest.approx(
        results["total_time"] / results["experiments"]
    )


def test_run_prange_validation_rejects_non_positive_experiment_count():
    with pytest.raises(
        ValueError,
        match="Number of experiments must be positive.",
    ):
        run_prange_validation(
            number_of_experiments=0,
            rows=2,
            columns=3,
            weight=1,
            max_iterations=100,
            seed=42,
        )