from experiments.prange_benchmark import (
    run_prange_random_vs_qc,
)


def test_run_prange_random_vs_qc():
    random_summary, qc_summary = (
        run_prange_random_vs_qc(
            block_size=4,
            weight=1,
            number_of_experiments=5,
            max_iterations=100,
            seed=42,
        )
    )

    assert random_summary.algorithm == "Prange"
    assert qc_summary.algorithm == "Prange"

    assert random_summary.code_type == "random"
    assert qc_summary.code_type == "quasi-cyclic"

    assert random_summary.rows == 4
    assert qc_summary.rows == 4

    assert random_summary.columns == 8
    assert qc_summary.columns == 8

    assert random_summary.weight == 1
    assert qc_summary.weight == 1

    assert random_summary.experiments == 5
    assert qc_summary.experiments == 5