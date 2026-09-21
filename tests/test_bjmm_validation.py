import pytest

from experiments.bjmm_validation import (
    generate_error_vector,
    run_bjmm_validation,
)


def test_generate_error_vector_has_correct_weight():
    import random

    rng = random.Random(42)

    error = generate_error_vector(
        length=10,
        weight=3,
        rng=rng,
    )

    assert len(error) == 10
    assert sum(error) == 3
    assert all(value in (0, 1) for value in error)


def test_generate_error_vector_rejects_negative_weight():
    import random

    with pytest.raises(
        ValueError,
        match="Weight must not be negative.",
    ):
        generate_error_vector(
            length=10,
            weight=-1,
            rng=random.Random(42),
        )


def test_generate_error_vector_rejects_excessive_weight():
    import random

    with pytest.raises(
        ValueError,
        match="Weight must not exceed vector length.",
    ):
        generate_error_vector(
            length=5,
            weight=6,
            rng=random.Random(42),
        )


def test_run_bjmm_validation_returns_expected_statistics():
    results = run_bjmm_validation(
        number_of_experiments=5,
        rows=4,
        columns=8,
        weight=2,
        p=2,
        p1=2,
        ell1=1,
        ell2=1,
        max_iterations=20,
        seed=42,
    )

    assert results["experiments"] == 5

    assert (
        results["successes"]
        + results["failures"]
        == 5
    )

    assert 0.0 <= results["success_rate"] <= 1.0

    assert results["total_time"] >= 0.0
    assert results["average_time"] >= 0.0
    assert results["minimum_time"] >= 0.0
    assert results["maximum_time"] >= 0.0


def test_run_bjmm_validation_is_reproducible_in_outcomes():
    result1 = run_bjmm_validation(
        number_of_experiments=5,
        rows=4,
        columns=8,
        weight=2,
        p=2,
        p1=2,
        ell1=1,
        ell2=1,
        max_iterations=20,
        seed=123,
    )

    result2 = run_bjmm_validation(
        number_of_experiments=5,
        rows=4,
        columns=8,
        weight=2,
        p=2,
        p1=2,
        ell1=1,
        ell2=1,
        max_iterations=20,
        seed=123,
    )

    assert result1["successes"] == result2["successes"]
    assert result1["failures"] == result2["failures"]
    assert result1["success_rate"] == result2["success_rate"]


def test_run_bjmm_validation_rejects_non_positive_number_of_experiments():
    with pytest.raises(
        ValueError,
        match="Number of experiments must be positive.",
    ):
        run_bjmm_validation(
            number_of_experiments=0,
            rows=4,
            columns=8,
            weight=2,
            p=2,
            p1=2,
            ell1=1,
            ell2=1,
            max_iterations=20,
            seed=42,
        )