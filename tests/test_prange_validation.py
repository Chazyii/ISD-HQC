import csv

import pytest

from experiments.prange_validation import (
    run_prange_validation,
    save_results_to_csv,
)


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

    experiment_results = results["experiment_results"]

    assert len(experiment_results) == 3

    assert experiment_results[0]["experiment_id"] == 1
    assert experiment_results[0]["success"] is True

    assert experiment_results[1]["experiment_id"] == 2
    assert experiment_results[1]["success"] is False

    assert experiment_results[2]["experiment_id"] == 3
    assert experiment_results[2]["success"] is True

    for result in experiment_results:
        assert result["rows"] == 2
        assert result["columns"] == 3
        assert result["weight"] == 1
        assert result["max_iterations"] == 100
        assert result["seed"] == 42
        assert result["execution_time"] >= 0


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


def test_save_results_to_csv(tmp_path):
    experiment_results = [
        {
            "experiment_id": 1,
            "rows": 2,
            "columns": 3,
            "weight": 1,
            "max_iterations": 100,
            "seed": 42,
            "success": True,
            "execution_time": 0.001,
        },
        {
            "experiment_id": 2,
            "rows": 2,
            "columns": 3,
            "weight": 1,
            "max_iterations": 100,
            "seed": 42,
            "success": False,
            "execution_time": 0.002,
        },
    ]

    output_path = tmp_path / "results.csv"

    save_results_to_csv(
        experiment_results=experiment_results,
        output_path=output_path,
    )

    assert output_path.exists()

    with output_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert len(rows) == 2

    assert rows[0]["experiment_id"] == "1"
    assert rows[0]["success"] == "True"
    assert rows[0]["execution_time"] == "0.001"

    assert rows[1]["experiment_id"] == "2"
    assert rows[1]["success"] == "False"
    assert rows[1]["execution_time"] == "0.002"