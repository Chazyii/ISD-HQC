import csv
import pytest
from experiments.prange_analysis import summarize, wilson_interval, write_summary
from experiments.prange_study import run


def test_wilson_interval_extremes():
    a, b = wilson_interval(0, 20)
    assert a >= 0 and b < 0.2
    a, b = wilson_interval(20, 20)
    assert a > 0.8 and b <= 1
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_aggregate_actual_runs(tmp_path):
    inputs = []
    for mode, targets in [('random', 1), ('qc', 1), ('qc-doom', 3)]:
        path = tmp_path / (mode + '.csv')
        run(mode, 5, 2, targets, 4, 100, 42, path)
        inputs.append(path)
    summaries = summarize(inputs)
    assert len(summaries) == 3
    assert all(int(row['trials']) == 4 for row in summaries)
    assert all(float(row['wilson95_low']) <= float(row['success_rate']) <= float(row['wilson95_high']) for row in summaries)
    output = tmp_path / 'summary.csv'
    write_summary(summaries, output)
    with output.open() as f:
        assert len(list(csv.DictReader(f))) == 3
    with pytest.raises(FileExistsError):
        write_summary(summaries, output)
