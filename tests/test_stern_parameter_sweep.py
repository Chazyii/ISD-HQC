import pytest

from experiments.stern_parameter_sweep import (
    run_stern_parameter_sweep,
)


def test_run_stern_parameter_sweep(monkeypatch):
    calls = []

    def controlled_validation(
        number_of_experiments,
        rows,
        columns,
        weight,
        p,
        ell,
        max_iterations,
        max_systematic_attempts,
        seed,
    ):
        calls.append(
            (
                p,
                ell,
                seed,
            )
        )

        return {
            "experiments": number_of_experiments,
            "successful": 8,
            "failed": 2,
            "success_rate": 0.8,
            "total_time": 0.1,
            "average_time": 0.01,
            "minimum_time": 0.001,
            "maximum_time": 0.02,
        }

    monkeypatch.setattr(
        "experiments.stern_parameter_sweep.run_stern_validation",
        controlled_validation,
    )

    results = run_stern_parameter_sweep(
        parameter_sets=[
            (1, 1),
            (1, 2),
        ],
        number_of_experiments=10,
        rows=4,
        columns=8,
        weight=2,
        max_iterations=100,
        max_systematic_attempts=50,
        seed=42,
    )

    assert len(results) == 2

    assert results[0]["p"] == 1
    assert results[0]["ell"] == 1
    assert results[0]["success_rate"] == pytest.approx(0.8)

    assert results[1]["p"] == 1
    assert results[1]["ell"] == 2
    assert results[1]["success_rate"] == pytest.approx(0.8)

    assert calls == [
        (1, 1, 42),
        (1, 2, 43),
    ]


def test_run_stern_parameter_sweep_rejects_empty_parameter_list():
    with pytest.raises(
        ValueError,
        match="At least one Stern parameter set must be provided.",
    ):
        run_stern_parameter_sweep(
            parameter_sets=[],
            number_of_experiments=10,
            rows=4,
            columns=8,
            weight=2,
            max_iterations=100,
            max_systematic_attempts=50,
            seed=42,
        )