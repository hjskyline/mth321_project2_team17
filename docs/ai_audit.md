# AI assistance and numerical audit record

## Extension audit — 2026-11-04

Scope: complete reproducible boundary/verification/AMG-cost suite on ziming.
AI drafted the extended code, comments, tests, derivation and explanations. The
reference function bodies are copied verbatim and attributed to fixed hua-nb
commit b3466958df84f0faa7d6f9ed79588a77558f05aa. No teammate notebook is modified.

### Claim audit 1: the two prescribed closures are actually different formulations

- Assumptions: smooth square model; nonempty Dirichlet portion owns all corners;
  q is the outward right derivative; beta=2 for Robin.
- Risk/falsifier: a forcing switch could be mislabeled first_order, or a corner
  could be overwritten. Compare explicit hand rows and affine exactness; isolate
  a wrong normal sign, missing inward factor and omitted boundary forcing.
- Independent evidence: hand constants and tests in test_mixed_boundary.py;
  48 COO/Kronecker comparisons on three analytic datasets; hand_rows.csv and
  falsification.csv. The independent implementation uses common mathematics, so
  it is not an independent proof of the continuous problem.
- Decision: automated checks support the named formulations on the tested cases.
  Human derivation review remains pending; no student signature is fabricated.

### Claim audit 2: measured convergence is not merely a small residual

- Assumptions: named complete-grid area L2 and nodal Linf; separate interior and
  right-edge norms; smooth exact exponential data; fixed beta.
- Risk/falsifier: tolerance contamination, ignored boundary error or local/global
  truncation confusion. Tighten two iterative paths; compare to LU and the
  independently coded BiCGSTAB algorithm; save all per-step rates and fit windows.
- Evidence: tolerance_sweep.csv, mms_summary.txt, truncation_diagnostics.csv,
  convergence_fits.csv, regional_errors.png. The exact barrier argument in theory.md
  uses M-matrix assumptions explicitly; small eigenvalue checks only corroborate it.
- Decision: all measured timing configurations satisfy both solution-difference/MMS
  fractions <=1e-3. Tighter solves remain approximate; LU comparison is numerical
  evidence, not an exact universal error bound. One full-run integration bug (mesh
  sizes passed as a Python list) was fixed by normalising inputs and adding a
  regression check before accepting the final complete run.

### Claim audit 3: efficiency comparison uses matched accuracy and honest costs

- Assumptions: one RHS, fresh setup, same diagonal row scaling, zero iterative
  initial guess, fixed restart/cycle limits, one BLAS thread, five repeats.
- Risk/falsifier: omit AMG setup, compare a loose iterative solve to a round-off
  direct solve, silently reuse factors, or discard a method that fails a PDE target.
- Evidence: performance_calibration.csv, performance_raw.csv (240 measured runs),
  performance_summary.csv and accuracy_targets.csv; true residuals and converged
  flags are recorded. Numerical targets are actual measured solutions, not fitted costs.
- Decision: the recorded medians describe this machine and tested sizes only.
  Target failures are retained. Warmups, oracle solves, tolerance calibration,
  acceptance checks, imports, plotting and I/O are explicitly excluded. Nonzeros
  are storage proxies, not exact RAM. GMRES callback curves use preconditioned
  norms and are labelled accordingly; final true residuals are separate.

### Five-gate status and ownership

Gates 1–2: equations, corner/ordering, boundary rows and signs have automated
evidence. Gate 3 nonlinear Jacobian is not applicable; linear stopping/cap flags
are tested. Gate 4: complementary analytic/assembly/solver checks pass. Gate 5:
isolated falsification experiments expose plausible wrong computations. These
statements describe AI-run checks, not a human-signed completion of the course audit.

Team members remain responsible for independently checking the theory, library
interfaces, assumptions and results, and for genuine ICS/issue-ticket/discussion
records. No report-ready claim of real-device validation, pure-Neumann correctness,
or universal AMG superiority is made.

Date: 2026-10-28. Scope: initial Direction 2 repository baseline.

AI actions: inspected the current course scaffolding; reused the existing
mixed_boundary_fd.py; adapted output locations and defaults; wrote run_all.py,
tests, model/provenance documentation, and README instructions. No report text
or experimental performance comparison is claimed by this baseline.

## Gate 1: transcription

See model.md for the prescribed PDE, exact solution, right-edge data, outward
normal, and separate boundary equations. beta=2; f=-2 exp(x+y). Corner ownership
is explicitly Dirichlet. Direct inspection matches the model to assembly.

## Gate 2: structure and indexing

The unknown count is N*(N-1), not N^2 or (N-1)^2. Hand constants for N=4,
right node (4,1) are documented independently of the assembly loop and tested.
An affine solution checks all four combinations and N=2,4,8; a quadratic checks
the ghost closure. Sparse assembly, coefficient signs, and corners are checked.

## Gate 3: linearisation and stopping

Nonlinear Jacobian check: not applicable (linear Poisson problem).
Main solver is sparse direct LU, no user tolerance. The independent BiCGSTAB
check requires info=0 and compares solutions, not only residuals; it records
rtol and iterations. A failed check raises an error rather than reporting success.

## Gate 4: independent reproduction

Checks include affine and quadratic exact solutions, explicit boundary rows,
small-system dense versus sparse linear algebra, and finest-grid BiCGSTAB versus
LU. Four-mesh exponential MMS results and scaled/backward residuals are recorded
by run_all.py. Runtime evidence is in results/verification.txt,
results/unit_tests.txt, results/mms_summary.txt and results/mms_convergence.csv.
These are complementary diagnostics, not a theorem for all problems.

## Gate 5: ways a plausible converging solver could still be wrong

1. Both solvers could share the same boundary sign or indexing error. Countercheck:
   hand boundary coefficients, affine exactness, corners and ghost quadratic.
2. An area norm could omit boundary unknowns or incorrectly count corners.
   Countercheck: analytic unit-error boundary-weight test and separate boundary norms.
3. A small raw residual could hide scaling differences or forward error.
   Countercheck: rescaling-invariance test, componentwise backward error and
   independent solution differences compared with measured MMS error.
4. A refinement fit could hide a poor window. Countercheck: retain all per-step
   rates, disclose the last-three-level diagnostic fit, and report its deviation.

## Human review still required

Automated checks do not replace the course's student/team audit. Team members
must independently explain and approve the equations, index map, norms,
library interfaces, and interpretation before using numbers as report evidence.
Do not label this file as a student-signed or instructor-approved audit.
