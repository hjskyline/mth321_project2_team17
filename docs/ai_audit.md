# AI assistance and numerical audit record

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
