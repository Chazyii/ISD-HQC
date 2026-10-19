# Publication experiment protocol — Prange, Random / QC / QC-DOOM

## Scope and scientific interpretation

We study a *toy, controlled* index-2 quasi-cyclic parity-check matrix H=[C0|C1],
with r=block_size and n=2r, versus an i.i.d. binary matrix of the same size.
This is **not** an HQC parameter set or a demonstrated attack on HQC.
Both ensembles are conditioned on full row rank.

Modes:
- `random`: one planted syndrome, random full-rank binary H.
- `qc`: one planted syndrome, random full-rank QC H.
- `qc-doom`: M cyclic shifts of the **same** planted error/syndrome under a QC automorphism.

The last mode uses **correlated, symmetry-related targets**, not M independent
syndromes. It is an orbit-based multi-target experiment, and should not be
presented as the general independent-syndrome DOOM complexity. The multi-target
solver reuses a sampled information set and checks multiple right-hand sides,
but the reference implementation currently solves each RHS separately.
It does **not** implement a shared factorization or optimized batched GF(2) solve.

## What the measurements mean

- `iterations`: number of sampled information sets, including singular trials.
- `singular_systems`: information sets whose selected r-by-r submatrix is singular.
- `systems_solved`: attempted RHS solves for nonsingular submatrices.
- `elapsed_seconds`: wall-clock decoder time only; matrix generation, rank
  rejection sampling, and syndrome-orbit construction are excluded.
- `success`: at least one weight-w solution found within `max_iterations`.

**Important:** Comparing modes by iteration count alone is misleading: an
iteration in QC-DOOM can involve multiple RHS solves. Compare both wall time
and `systems_solved`, report singular rates and success rates, and keep the
same iteration cap when comparing success probability. `systems_per_success`
is aggregate attempted solves divided by successful trials, **not** an
unbiased expected runtime-to-success estimator under a finite cap.

## Reproducibility

`seed` is the per-trial instance seed, generated as `base_seed + 1000003*trial`;
the decoder uses that seed +7919. In this version, random and QC ensembles
use identical seed numbers but different generation procedures: this is not
a mathematically paired sample of matrices. Run multiple independent base
seeds for confidence, and never mix rows with different `max_iterations`.

## Example local runs (PowerShell, repository root)

```powershell
$env:PYTHONPATH="src"
python -m pytest -q tests/test_prange_study.py tests/test_prange_analysis.py
python -m experiments.prange_study --mode random --block-size 8 --weight 2 --targets 1 --trials 100 --max-iterations 1000 --seed 42 --output results/random_8.csv
python -m experiments.prange_study --mode qc --block-size 8 --weight 2 --targets 1 --trials 100 --max-iterations 1000 --seed 42 --output results/qc_8.csv
python -m experiments.prange_study --mode qc-doom --block-size 8 --weight 2 --targets 4 --trials 100 --max-iterations 1000 --seed 42 --output results/qc_doom_8.csv
python -m experiments.prange_analysis results/random_8.csv results/qc_8.csv results/qc_doom_8.csv --output results/summary_8.csv
```

CSV output refuses to overwrite existing files. Choose new filenames for reruns.

## Before claiming publication-grade results

1. Implement and validate a *separate* independent-syndrome DOOM control.
2. Confirm equal-work or equal-time budgets and add memory accounting.
3. Use larger parameter sweeps and independent replications; plot confidence
   intervals and conditional-on-success timing separately.
4. Profile and optimize GF(2) arithmetic before attempting HQC-like sizes.
5. Verify cluster software, allocation and SLURM options on WCSS Lem.
