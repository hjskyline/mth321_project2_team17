"""Project 2, Direction 2: mixed-boundary Poisson FDM and MMS experiments.

Run: python mixed_boundary_fd.py
     python mixed_boundary_fd.py --meshes 8 16 32 64
     python mixed_boundary_fd.py --verify-only

PDE: -Delta u = -2 exp(x+y), on (0,1)^2.
Dirichlet: x=0, y=0, y=1, INCLUDING all four corners.
Right open edge: u_x=q=exp(1+y), or u_x+beta*u=r=(1+beta)*exp(1+y).
Default beta=2.  Compare a first-order derivative row with a centred ghost
closure that enforces the PDE at the right boundary.  The matrices are
nonsymmetric; use sparse LU, not CG.  Adapted from the existing learning-material file 03_Scaffolding/mixed_boundary_fd.py.
See docs/scaffolding_reference.md for provenance and docs/model.md for derivation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.sparse import coo_matrix, diags, issparse
from scipy.sparse.linalg import bicgstab, norm as sparse_norm, spsolve


@dataclass(frozen=True)
class ManufacturedData:
    """All functions accept NumPy arrays; q is the outward derivative at x=1."""
    exact: Callable
    forcing: Callable
    q: Callable
    name: str


def exact(x, y):
    return np.exp(np.asarray(x) + np.asarray(y))


def forcing(x, y):
    return -2.0 * exact(x, y)


def neumann(y):
    return np.exp(1.0 + np.asarray(y))


def dirichlet(x, y):
    return exact(x, y)


def robin(y, beta=2.0):
    return (1.0 + beta) * np.exp(1.0 + np.asarray(y))


MMS = ManufacturedData(exact, forcing, neumann, "exp(x+y)")


@dataclass
class DiscreteSystem:
    A: object
    b: np.ndarray
    N: int
    h: float
    bc_type: str
    closure: str
    beta: float
    data: ManufacturedData

    def index(self, i, j):
        """Unknowns: i=1..N, j=1..N-1; j varies fastest (C ordering)."""
        if not (1 <= i <= self.N and 1 <= j < self.N):
            raise IndexError("node is not an unknown")
        return (i - 1) * (self.N - 1) + j - 1

    def coordinates(self):
        return np.meshgrid(np.arange(1, self.N + 1) * self.h,
                           np.arange(1, self.N) * self.h, indexing="ij")

    def full_solution(self, uh):
        grid = np.arange(self.N + 1) * self.h
        X, Y = np.meshgrid(grid, grid, indexing="ij")
        U = np.array(self.data.exact(X, Y), dtype=float, copy=True)
        U[1:, 1:-1] = np.asarray(uh).reshape(self.N, self.N - 1)
        return U


def assemble_mixed_poisson(N, bc_type="neumann", closure="ghost", beta=2.0,
                           data=MMS):
    """Eliminate Dirichlet nodes; retain right-edge noncorner unknowns.

    Let s=q and gamma=0 (Neumann), or s=r and gamma=beta (Robin).
    First order: (1/h+gamma)*U_Nj - U_(N-1)j/h = s_j.
    Ghost: (4/h^2+2*gamma/h)*U_Nj - 2*U_(N-1)j/h^2
           - (U_N,j-1 + U_N,j+1)/h^2 = f_Nj + 2*s_j/h.
    Known neighbours in either formula are moved to b with a positive sign.
    Ghost values U_(N+1)j are eliminated analytically and never numbered.
    """
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer >= 2 (number of intervals)")
    if bc_type not in ("neumann", "robin"):
        raise ValueError("bc_type must be 'neumann' or 'robin'")
    if closure not in ("first_order", "ghost"):
        raise ValueError("closure must be 'first_order' or 'ghost'")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    h = 1.0 / N
    size = N * (N - 1)
    rows, cols, values = [], [], []
    b = np.zeros(size)
    gamma = beta if bc_type == "robin" else 0.0

    def index(i, j):
        return (i - 1) * (N - 1) + j - 1

    def add(k, i, j, coefficient):
        # Top/bottom own the corners, even if i=N.
        if i == 0 or j == 0 or j == N:
            b[k] -= coefficient * data.exact(i * h, j * h)
        else:
            rows.append(k)
            cols.append(index(i, j))
            values.append(coefficient)

    for i in range(1, N + 1):
        for j in range(1, N):
            k = index(i, j)
            x, y = i * h, j * h
            if i < N:
                b[k] = data.forcing(x, y)
                add(k, i, j, 4.0 / h**2)
                for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    add(k, ni, nj, -1.0 / h**2)
            else:
                s = data.q(y) + gamma * data.exact(1.0, y)
                if closure == "first_order":
                    b[k] = s
                    add(k, i, j, 1.0 / h + gamma)
                    add(k, i-1, j, -1.0 / h)
                else:
                    b[k] = data.forcing(x, y) + 2.0 * s / h
                    add(k, i, j, 4.0 / h**2 + 2.0 * gamma / h)
                    add(k, i-1, j, -2.0 / h**2)
                    add(k, i, j-1, -1.0 / h**2)
                    add(k, i, j+1, -1.0 / h**2)

    A = coo_matrix((values, (rows, cols)), shape=(size, size)).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    return DiscreteSystem(A, b, N, h, bc_type, closure, beta, data)


def error_metrics(system, uh):
    """Global L2 uses tensor trapezoidal weights, including half weight at x=1.

    All prescribed Dirichlet nodes have zero error.  Interior L2 uses h^2
    area weights; right-edge L2 uses h length weights.  They have different
    dimensions and are not interchangeable with the global L2 norm.
    """
    N, h = system.N, system.h
    grid = np.arange(N + 1) * h
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    e = system.full_solution(uh) - system.data.exact(X, Y)
    w = np.ones(N + 1)
    w[[0, -1]] = 0.5
    interior, right = e[1:N, 1:N], e[N, 1:N]
    return {
        "L2": h * np.sqrt(np.sum(np.outer(w, w) * e**2)),
        "Linf": np.max(np.abs(e)),
        "interior_L2": h * np.linalg.norm(interior),
        "interior_Linf": np.max(np.abs(interior)),
        "right_L2": np.sqrt(h) * np.linalg.norm(right),
        "right_Linf": np.max(np.abs(right)),
    }


def residual_metrics(system, uh):
    """Residual definitions are invariant to positive rescaling of each row.

    row_scaled_residual_inf = max_i |r_i|/|a_ii| (solution units).
    componentwise_backward_error = max_i |r_i|/(|A||uh|+|b|)_i.
    Raw and RHS-relative residuals are also recorded, but should not alone
    be compared across derivative rows (h^-1) and PDE rows (h^-2).
    """
    A, b = system.A, system.b
    r = b - A @ uh
    denominator = abs(A) @ np.abs(uh) + np.abs(b)
    ratios = np.divide(np.abs(r), denominator, out=np.zeros_like(r),
                       where=denominator > 0)
    ratios[(denominator == 0) & (r != 0)] = np.inf
    rinf = np.linalg.norm(r, np.inf)
    bnorm = np.linalg.norm(b, np.inf)
    return {
        "residual_inf": rinf,
        "relative_residual_inf": rinf / bnorm if bnorm else rinf,
        "row_scaled_residual_inf": np.max(np.abs(r) / np.abs(A.diagonal())),
        "componentwise_backward_error": np.max(ratios),
    }


def solve_and_measure(N, bc_type="neumann", closure="ghost", beta=2.0):
    system = assemble_mixed_poisson(N, bc_type, closure, beta)
    uh = spsolve(system.A, system.b)
    if not np.all(np.isfinite(uh)):
        raise FloatingPointError("direct solve returned nonfinite values")
    row = {
        "bc_type": bc_type, "closure": closure, "N": N, "h": system.h,
        "unknowns": system.A.shape[0], "nnz": system.A.nnz,
        "solver": "spsolve", "solver_tolerance": "N/A (direct LU)",
        **error_metrics(system, uh), **residual_metrics(system, uh),
        "asymmetry_fro": float(sparse_norm(system.A - system.A.T)
                               / sparse_norm(system.A)),
    }
    return system, uh, row


def independent_solve_check(system, direct, target=1e-3):
    """Independent BiCGSTAB algorithm on a diagonally row-scaled system.

    Compare solution vectors, not residuals alone.  BiCGSTAB rtol refers to
    the scaled system's relative 2-norm residual; it is not a forward-error
    tolerance.  Require agreement <= target times measured MMS error in
    BOTH reported global norms.  Sparse LU is still used for the MMS table.
    """
    R = diags(1.0 / system.A.diagonal(), format="csr")
    B, rhs = R @ system.A, R @ system.b
    iterations = [0]

    def callback(_):
        iterations[0] += 1

    other, info = bicgstab(B, rhs, rtol=1e-12, atol=0.0, maxiter=20000,
                          callback=callback)
    if info != 0:
        raise RuntimeError(f"independent BiCGSTAB failed: info={info}")
    # Make a zero-Dirichlet error grid for the difference of two solutions.
    difference = np.zeros((system.N + 1, system.N + 1))
    difference[1:, 1:-1] = (other - direct).reshape(system.N, system.N - 1)
    w = np.ones(system.N + 1)
    w[[0, -1]] = 0.5
    diff_l2 = system.h * np.sqrt(np.sum(np.outer(w, w) * difference**2))
    diff_inf = np.max(np.abs(difference))
    errors = error_metrics(system, direct)
    ratios = (diff_l2 / errors["L2"], diff_inf / errors["Linf"])
    if max(ratios) > target:
        raise RuntimeError(f"algebraic error check failed: ratios={ratios}")
    return {"independent_solver": "BiCGSTAB on row-scaled system",
            "rtol": 1e-12, "iterations": iterations[0],
            "L2_difference_over_MMS_error": float(ratios[0]),
            "Linf_difference_over_MMS_error": float(ratios[1]),
            "backward_error": float(residual_metrics(system, other)["componentwise_backward_error"])}


def verify_implementation():
    """Small-grid checks independent of exponential convergence fits.

    Includes hand-built boundary rows and an affine exact solution.  The
    affine solution is exact for both derivative formulas and the interior
    stencil, exposing boundary signs/indexing without assuming a fitted rate.
    """
    affine = ManufacturedData(lambda x, y: 1.0 + x + 2.0*y,
                              lambda x, y: 0.0,
                              lambda y: 1.0, "1+x+2y")
    messages = []
    for bc in ("neumann", "robin"):
        for closure in ("first_order", "ghost"):
            for N in (2, 4, 8):
                s = assemble_mixed_poisson(N, bc, closure, data=affine)
                assert issparse(s.A) and s.A.shape == (N*(N-1),)*2
                assert s.index(1, 1) == 0
                assert s.index(N, N-1) == s.A.shape[0] - 1
                np.testing.assert_array_equal(s.A.diagonal() > 0, True)
                off = s.A - diags(s.A.diagonal())
                assert np.all(off.data <= 0)
                x, y = s.coordinates()
                sampled = affine.exact(x, y).ravel()
                np.testing.assert_allclose(s.A @ sampled, s.b, atol=2e-12, rtol=0)
                computed = spsolve(s.A, s.b)
                np.testing.assert_allclose(computed, sampled, atol=2e-12, rtol=0)
                U = s.full_solution(computed)
                for i, j in ((0, 0), (0, N), (N, 0), (N, N)):
                    assert U[i, j] == affine.exact(i/N, j/N)

            # Hand transcription at N=4, j=1, touching the bottom Dirichlet edge.
            s = assemble_mixed_poisson(4, bc, closure)
            k, inward = s.index(4, 1), s.index(3, 1)
            gamma = 2.0 if bc == "robin" else 0.0
            t = np.exp(1.25)
            expected = np.zeros(s.A.shape[1])
            if closure == "first_order":
                expected[k], expected[inward] = 4.0+gamma, -4.0
                expected_b = (1.0+gamma)*t
            else:
                expected[k], expected[inward] = 64.0+8.0*gamma, -32.0
                expected[s.index(4, 2)] = -16.0
                expected_b = -2.0*t + 8.0*(1.0+gamma)*t + 16.0*np.e
            np.testing.assert_allclose(s.A.getrow(k).toarray()[0], expected)
            np.testing.assert_allclose(s.b[k], expected_b)
            messages.append(f"PASS {bc}/{closure}: affine solution N=2,4,8; "
                            "hand boundary row; corner ownership; sparse indexing/signs")
    return messages


def observed_order(hs, errors):
    errors = np.asarray(errors)
    if np.any(errors <= 0) or not np.all(np.isfinite(errors)):
        raise ValueError("need positive finite errors for fitting")
    rates = np.full(len(hs), np.nan)
    rates[1:] = np.log(errors[:-1] / errors[1:]) / np.log(hs[:-1] / hs[1:])
    return rates


def run_study(meshes, outdir, fit_last=3, figure_dir=None):
    """Fit a declared diagnostic window; never enforce the expected slope."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(figure_dir) if figure_dir is not None else outdir
    figure_dir.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, checks = [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    spyfig, spyaxes = plt.subplots(2, 2, figsize=(9, 8))
    for position, (bc, closure) in enumerate(
            (b, c) for b in ("neumann", "robin") for c in ("first_order", "ghost")):
        rows = []
        for N in meshes:
            system, uh, row = solve_and_measure(N, bc, closure)
            rows.append(row)
        hs = np.array([r["h"] for r in rows])
        for key in ("L2", "Linf", "right_Linf", "interior_Linf"):
            rates = observed_order(hs, [r[key] for r in rows])
            for row, rate in zip(rows, rates):
                row[f"p_{key}"] = rate
        all_rows.extend(rows)
        check = independent_solve_check(system, uh)
        checks.append(f"{bc}/{closure}, N={meshes[-1]}: {check}")
        ax = axes.flat[position]
        for key, marker in (("L2", "o"), ("Linf", "s"), ("right_Linf", "^")):
            errors = np.array([r[key] for r in rows])
            ax.loglog(hs, errors, marker+"-", label=key)
            p, intercept = np.polyfit(np.log(hs[-fit_last:]), np.log(errors[-fit_last:]), 1)
            deviation = np.max(np.abs(np.log(errors[-fit_last:])
                                      - (p*np.log(hs[-fit_last:])+intercept)))
            monotone = bool(np.all(np.diff(errors) < 0))
            summaries.append(f"{bc}/{closure} {key}: slope={p:.8f}, "
                             f"max_log_deviation={deviation:.3e}, "
                             f"monotone={monotone}, fit_N={meshes[-fit_last:]}")
            if key == "Linf":
                fit_label = (f"Linf fit: p={p:.4f}\n"
                             f"max log-deviation={deviation:.2e}\n"
                             f"N={meshes[-fit_last:]}")
            if not monotone:
                summaries.append("WARNING: nonmonotone errors; inspect window before reporting slope")
        expected = 1 if closure == "first_order" else 2
        ref = rows[0]["Linf"] * (hs/hs[0])**expected
        ax.loglog(hs, ref, "k--", alpha=0.6, label=f"reference h^{expected}")
        ax.set_xticks(hs, labels=[f"1/{N}" for N in meshes])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"{bc.capitalize()} / {closure}")
        ax.set(xlabel="Mesh size h", ylabel="Error norm")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        ax.text(0.97, 0.04, fit_label, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))
        small = assemble_mixed_poisson(8, bc, closure)
        spyaxes.flat[position].spy(small.A, markersize=1)
        spyaxes.flat[position].set_title(f"{bc} / {closure} (N=8)")
        print(f"\n{bc} / {closure}")
        print(" N      h         L2          Linf      p_L2   p_Linf   scaled_residual")
        for r in rows:
            print(f"{r['N']:3d} {r['h']:.6f} {r['L2']:.4e} {r['Linf']:.4e} "
                  f"{r['p_L2']:6.3f} {r['p_Linf']:7.3f} "
                  f"{r['row_scaled_residual_inf']:.3e}")

    with (outdir / "mms_convergence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    report = ["Direction 2: mixed-boundary MMS", "u*=exp(x+y); f=-2 exp(x+y); beta=2",
              "Direct solve: spsolve; no user stopping tolerance.",
              "Global L2: tensor trapezoidal h^2 weights; right edge half weight.",
              "Boundary L2: h weights on right-edge noncorner nodes.",
              "Scaled residual: max |b-Au|/|diag(A)|.",
              "Backward error: max_i |b-Au|_i/(|A||u|+|b|)_i.",
              "Asymmetry: Frobenius norm ||A-A.T||_F/||A||_F.",
              f"Provisional fits: last {fit_last} levels; inspect per-step rates in CSV.",
              "First-order rows replace the PDE; ghost rows enforce boundary PDE.",
              "Thus this comparison changes the formulation as well as derivative accuracy.",
              "", *summaries, "", "Independent algebraic checks:", *checks]
    (outdir / "mms_summary.txt").write_text("\n".join(report)+"\n")
    fig.tight_layout()
    fig.savefig(figure_dir / "mms_convergence.png", dpi=160)
    plt.close(fig)
    spyfig.tight_layout()
    spyfig.savefig(figure_dir / "matrix_sparsity.png", dpi=160)
    plt.close(spyfig)
    (outdir / "verification.txt").write_text("\n".join(verify_implementation())+"\n")
    print("\n" + "\n".join(summaries))
    print(f"\nResults saved to {outdir.resolve()}")
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--fit-last", type=int, default=3)
    parser.add_argument("--outdir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--figure-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "figures")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for message in verify_implementation():
        print(message)
    if args.verify_only:
        return
    if (len(args.meshes) < 4 or any(n < 2 for n in args.meshes)
            or any(b <= a for a, b in zip(args.meshes, args.meshes[1:]))):
        parser.error("provide at least four strictly increasing N >= 2")
    if not 3 <= args.fit_last <= len(args.meshes):
        parser.error("fit-last must select at least 3 levels and no more than available")
    print("Predictions: first-order right-edge Linf -> order 1; "
          "ghost global L2/Linf -> order 2.")
    print("Global norms include right-edge unknowns; corners remain Dirichlet.")
    run_study(args.meshes, args.outdir, args.fit_last, args.figure_dir)


if __name__ == "__main__":
    main()
