# MTH321 Project 2 — Team 17

**Mixed-boundary Poisson: boundary closure, verification, and matched-accuracy cost.**

Implemented on Ziming's `ziming` branch. The `hua-nb` notebook and branch are not modified.
The mathematical model is a verification benchmark, not a calibrated device simulation.

## Start here

- 中文工作说明：`docs/WORK_COMPLETED_ZH.md`
- 实际实验结果摘要：`results/EXPERIMENT_SUMMARY.md`
- 模型与边界行：`docs/model.md`
- 针对本题的稳定性/阶数推导草稿：`docs/theory.md`
- 计算前的预测、参数和验收标准：`docs/experiment_protocol.md`
- AI 审查与仍需人工完成的部分：`docs/ai_audit.md`

## Setup and reproduce

Python 3.11+ is supported (the recorded run uses 3.12). From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python code/run_all.py
```

Windows activation: `.venv\Scripts\activate`. If exact pinned wheels are unavailable
on another platform, install `requirements.txt`, record the actual versions, and rerun
the numerical comparisons. No system Python installation is modified.

Defaults: `N=16,32,64,128`, last-three-level diagnostic fits, 5 independent timing
repetitions per configuration, 1 BLAS thread. Outputs resolve relative to the repository,
so the script also works when invoked from another directory.

```sh
# All tests separately
python -m unittest discover -s code -p 'test_*.py' -v
# Preserve the archived results while reproducing into a fresh directory
python code/run_all.py --output-dir /tmp/pde_reproduction
# Only assembly checks, MMS convergence and spatial diagnostics (no timings)
python code/run_all.py --baseline-only
# Explicit alternative four-level halving study
python code/run_all.py --meshes 8 16 32 64 --repeats 5
```

Existing output filenames are replaced on reproduction. Timing changes with hardware,
load and library builds; exact wall-clock reproduction is not claimed. Numerical values
must agree to justified reported digits. Code hashes, parameters, package versions,
thread limits and output hashes are written with every complete run.

## Implemented experiments

1. Both prescribed right boundary types (Neumann and Robin beta=2), with true
   first-order derivative rows and ghost-eliminated boundary PDE rows.
2. Four-mesh exponential MMS studies: global/interior/right L2 and Linf, per-step
   observed orders, fitted order/deviation, scaled/backward residuals and asymmetry.
3. Hand boundary rows, affine/quadratic exactness, independent Kronecker assembly
   with a second non-polynomial analytic dataset, matrix structure and barrier checks.
4. Finest-mesh GMRES/AMG-GMRES tolerance sweeps, unscaled direct and independent
   BiCGSTAB comparisons, true convergence flags and an actual solution-error budget.
5. Isolated falsifiers: wrong normal sign, wrong inward ghost coefficient, omitted
   boundary forcing, and loose solver tolerance. Production matrices remain unchanged.
6. Matched-accuracy LU vs GMRES vs AMG-preconditioned GMRES. Every measured run
   passes the same two-norm solution-difference/MMS criterion of 1e-3.
7. First tested mesh meeting fixed global Linf targets 1e-2 and 1e-3, including
   explicit records when a configuration does not reach a target.

## Code map

```text
code/mixed_boundary_fd.py    Explicit COO/CSR assembly, norms and baseline verification
code/reference_hua_ghost.py  Verbatim selected reference definitions, fixed source commit
code/kron_reference.py       Independent tensor assembly and separate first-order adapter
code/linear_solvers.py       Shared scaling, LU/GMRES/AMG, timing and convergence metadata
code/experiment_suite.py     Independent checks, tolerance, falsification, costs and plots
code/run_all.py              Single complete reproduction entry point
code/test_*.py               Mathematical/structural regression tests
docs/                       Model, theory, provenance, protocol and audit
results/                    Generated CSVs, logs, summary and metadata
figures/                    Generated scientific plots
report/, slides/            Team-owned report and presentation work
```

## Evidence map

| Question | Saved evidence |
|---|---|
| Do the prescribed boundary rows match? | `hand_rows.csv`, `verification.txt`, `unit_tests.txt` |
| Do independent assemblies agree? | `assembly_crosscheck.csv`, `matrix_checks.csv` |
| How do global and regional errors converge? | `mms_convergence.csv`, `convergence_fits.csv`, `regional_errors.png` |
| Why can boundary residual and solution error have different orders? | `truncation_diagnostics.csv`, `docs/theory.md` |
| Where is the spatial error? | `error_comparison.png`, `signed_error_sections.png`, `error_geometry.csv` |
| Is the linear solve accurate enough? | `tolerance_sweep.csv`, `mms_summary.txt`, `solver_residual_history.csv` |
| Would a plausible wrong code be detected? | `falsification.csv`, `falsification.png` |
| What does matched accuracy cost? | `performance_raw.csv`, `performance_summary.csv`, `solver_cost.png` |
| What does a PDE accuracy target cost? | `accuracy_targets.csv`, `accuracy_cost.png` |
| Can these exact working files be traced? | `environment.json`, `output_manifest.json`, Git commit |

Names without directories in this table are in `results/` for data and `figures/` for PNGs.

## Fairness, norms and limits

All performance methods solve the same diagonally row-scaled system. Include assembly,
scaling/conversion, factorization or AMG setup, and solve; exclude warm-ups, reference
solutions, calibration, validation, imports, plotting and I/O. Calibration and raw timing
records remain available. Five repeats produce medians and min/max ranges. Factor and
hierarchy nonzeros are storage proxies, not exact process-memory measurements.

Raw matrices are generally nonsymmetric; CG is not used. SciPy GMRES `pr_norm` callbacks
count inner iterations and represent preconditioned residuals with AMG. The two callback
histories therefore are not the same norm; final true residuals are computed separately.
Stopping tolerance is not forward solution error. RMS, area L2 and boundary line L2 are
also different quantities; the experiment outputs name and define their norms.

The reference source is pinned to hua-nb commit
`b3466958df84f0faa7d6f9ed79588a77558f05aa`; selected function bodies are preserved and
attributed. Its notebook is never executed, its pure-Neumann helper is not used, and its
False forcing switch is not relabelled as a first-order closure. See `source_reference.json`.

Automated evidence does not certify human understanding or instructor approval. The
team must still independently audit the equations, tests, library boundaries and claims,
write the English report, prepare the presentation, and record genuine contributions.
Pure Neumann, singular domains and real-world calibration remain outside this scope.

## Git workflow

Work on your personal branch, commit explicit files with meaningful descriptions,
and push to `origin ziming`. `main` and `hua-nb` are preserved. Integrate shared work
only through a separately reviewed merge. No lecture PDFs or virtual environments
belong in the repository.
