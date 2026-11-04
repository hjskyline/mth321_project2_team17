# Experiment protocol and predictions

Recorded before running the new experiment suite, 2026-11-04. These are predictions
and acceptance choices, not numbers fitted after seeing the output. Earlier baseline
evidence is already known and is explicitly not presented as a blind prediction.

## Fixed model and scope

Unit square; -Delta u=-2 exp(x+y); left/top/bottom and all corners Dirichlet;
open right edge has Neumann q=exp(1+y) or Robin u_x+2u=3exp(1+y).
Unknowns: i=1..N, j=1..N-1, k=(i-1)(N-1)+j-1, M=N(N-1).
Compare the prescribed first-order derivative row and ghost-eliminated PDE row.
Do not interpret a boundary-forcing switch as a boundary-row replacement.

## Predictions to reconcile

1. Exponential solution and outward right derivative are positive; source is negative.
   Dirichlet corner values must remain exact. No gauge is needed for this mixed BVP.
2. Affine data should be nodally exact for all four configurations. Quadratic data
   should be exact for the ghost closure, but should not be used to fit its rate.
3. On smooth exponential data, one-sided boundary Linf should approach first order;
   ghost global L2 and Linf should approach second order. Last-three-level fits are
   diagnostics until per-step rate stability and the finest algebraic gate are checked.
4. Interior truncation residual should approach O(h^2); ghost boundary PDE residual
   can approach O(h), despite second-order global solution error. See theory.md.
5. Explicit COO and independent tensor assembly should agree coefficient by
   coefficient on N=2,4,8,16 for exponential, affine and a sine-plus-polynomial test.
6. Tightening tolerances should eventually stop changing PDE errors and the right
   midpoint quantity of interest. An arbitrary small residual alone proves neither
   the error budget nor correct discretisation.
7. A wrong outward sign or inward ghost coefficient may still give a very small
   algebraic residual; independent exact data should expose it. Omitting boundary
   forcing may still converge but is not the prescribed one-sided method.
8. AMG may reduce iterations, but setup overhead may make sparse LU cheaper on
   some or all tested sizes. A speedup and a crossover are hypotheses, not criteria
   imposed on the measured data. We do not claim an asymptotic complexity from four sizes.

## Runs and acceptance choices

- Default meshes: N=16,32,64,128: THREE successive halvings across FOUR levels.
- Global L2: tensor trapezoidal area weights on the complete nodal grid (right
  unknown edge half weight); Linf includes the right edge. Interior L2 uses h^2,
  boundary L2 uses h; they are different area/line norms. All six norms are saved.
- Fit last three levels by default, retain all levels and all per-step orders;
  disclose maximum absolute log-fit deviation and rate stability. No data point is
  deleted to achieve a desired order. If the window is unstable, report and extend.
- Algebraic gate: both solution-difference norms / LU-based MMS errors <=1e-3.
  Independent BiCGSTAB on the finest systems also checks the unscaled LU results.
  Agreement of solvers with shared assembly is algebraic evidence, not independent
  discretisation proof or a rigorous exact forward-error bound.
- Tolerance sweep at finest mesh: GMRES and AMG-GMRES at 1e-6,1e-8,1e-10,1e-12,
  zero initial guess, restart=80, at most 100 restart cycles. Record final info,
  true scaled residual, backwards residual, actual solution differences and QoI.
  The tightest solve must pass the LU comparison; choose the loosest tested passing
  tolerance, and recheck every performance configuration rather than trusting it blindly.
- GMRES callbacks use SciPy pr_norm and count inner iterations. With AMG they
  represent preconditioned residuals: do not interpret the two histories as the same
  norm. Final true row-scaled residuals are recomputed and saved separately.
- Benchmarks: same diagonally row-scaled A,b for LU/GMRES/AMG-GMRES, one BLAS thread,
  one RHS, fresh assembly/factorization/preconditioner per repetition, five repetitions.
  Report median, minimum, maximum of assembly, preprocessing, setup, solve and total.
  Exclude imports, warm-up, oracle solve, calibration, acceptance diagnostics, plots,
  file I/O. Calibration rows are saved separately, including any failed attempts.
- Storage is reported as factor/hierarchy/transfer nonzeros, not exact RAM usage.
- Fixed PDE targets: global Linf <=1e-2 and <=1e-3. Report the first successful
  TESTED mesh and its measured cost; a missing target is retained as met=False,
  never replaced by an extrapolated/interpolated cost.
- No pure Neumann, singular domains, real equipment calibration or unsigned student
  contributions are included. The pure-Neumann notebook helper is not executed.

## Records

run_all.py executes the tests and writes a single run log, all CSVs and figures,
environment versions, code hashes and output hashes. EXPERIMENT_SUMMARY.md is
generated from those same result objects. Reconcile discrepancies in a human audit;
do not overwrite this prediction record to match the output.
