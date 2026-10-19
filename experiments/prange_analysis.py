"""Aggregate publication study CSVs without third-party dependencies.

Run: python -m experiments.prange_analysis results/cluster/*.csv --output results/summary.csv
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

GROUP_KEYS = ('mode', 'block_size', 'n', 'r', 'w', 'targets', 'max_iterations')


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    """Two-sided 95% Wilson score interval for independent Bernoulli trials."""
    if total < 1:
        raise ValueError('Total must be positive')
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - margin, center + margin


def summarize(paths):
    groups = defaultdict(list)
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {'mode','success','iterations','systems_solved','elapsed_seconds'}.issubset(reader.fieldnames):
                raise ValueError(f'Not a Prange study CSV: {path}')
            for row in reader:
                key = tuple(row.get(k, '') for k in GROUP_KEYS)
                groups[key].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        total = len(rows)
        successes = sum(int(r['success']) for r in rows)
        low, high = wilson_interval(successes, total)
        times = [float(r['elapsed_seconds']) for r in rows]
        successful_times = [float(r['elapsed_seconds']) for r in rows if int(r['success'])]
        attempts = sum(int(r['iterations']) for r in rows)
        systems = sum(int(r['systems_solved']) for r in rows)
        singular = sum(int(r['singular_systems']) for r in rows)
        summary = dict(zip(GROUP_KEYS, key))
        summary.update(trials=total, successes=successes, failures=total-successes,
                       success_rate=successes/total, wilson95_low=low, wilson95_high=high,
                       mean_seconds=sum(times)/total,
                       mean_seconds_success_only=(sum(successful_times)/len(successful_times)
                                                  if successful_times else ''),
                       mean_iterations=attempts/total,
                       mean_systems_solved=systems/total,
                       mean_singular=singular/total,
                       total_seconds=sum(times),
                       total_systems_solved=systems,
                       systems_per_success=(systems/successes if successes else ''),
                       seconds_per_success=(sum(times)/successes if successes else ''))
        summaries.append(summary)
    return summaries


def write_summary(summaries, path):
    if not summaries:
        raise ValueError('No experiment records')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results = summarize(args.inputs)
    write_summary(results, args.output)
    for row in results:
        print(f"{row['mode']:8s} n={row['n']} w={row['w']} targets={row['targets']} "
              f"success={row['success_rate']:.3f} "
              f"95%CI=[{row['wilson95_low']:.3f}, {row['wilson95_high']:.3f}] "
              f"mean={row['mean_seconds']:.6f}s")
