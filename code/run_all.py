"""Reproduce Direction 2 baseline tables and figures from any working directory."""
from pathlib import Path
import argparse
import json
import platform
import sys

import numpy as np
import scipy

from mixed_boundary_fd import assemble_mixed_poisson, run_study, verify_implementation


def solution_figures(N, figure_dir, result_dir):
    import matplotlib.pyplot as plt
    from scipy.sparse.linalg import spsolve

    axis = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    exact = np.exp(X + Y)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    errors, error_axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    archive = {"x": axis, "y": axis, "exact": exact}
    for position, (bc, closure) in enumerate(
            (b, c) for b in ("neumann", "robin") for c in ("first_order", "ghost")):
        system = assemble_mixed_poisson(N, bc, closure)
        U = system.full_solution(spsolve(system.A, system.b))
        error = U - exact
        archive[f"{bc}_{closure}"] = U
        ax, eax = axes.flat[position], error_axes.flat[position]
        cf = ax.contourf(X, Y, U, levels=np.linspace(1.0, np.e**2, 21),
                         extend="both", cmap="viridis")
        fig.colorbar(cf, ax=ax, label="u_h")
        limit = max(float(np.max(np.abs(error))), np.finfo(float).eps)
        ecf = eax.contourf(X, Y, error, levels=np.linspace(-limit, limit, 21),
                           cmap="coolwarm")
        errors.colorbar(ecf, ax=eax, label="u_h - u*")
        ax.set_title(f"{bc} / {closure}")
        eax.set_title(f"{bc} / {closure}\nmax error={limit:.3e}", fontsize=10)
        for panel in (ax, eax):
            panel.set(xlabel="x", ylabel="y", aspect="equal")
    fig.suptitle(f"Mixed-boundary Poisson: N={N} intervals, h={1/N:.5f}")
    errors.suptitle("Signed nodal error (each panel has its own colour scale)")
    fig.savefig(figure_dir / "solution_comparison.png", dpi=150)
    errors.savefig(figure_dir / "error_comparison.png", dpi=150)
    plt.close(fig)
    plt.close(errors)
    np.savez_compressed(result_dir / "baseline_solutions.npz", **archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--fit-last", type=int, default=3)
    args = parser.parse_args()
    if (len(args.meshes) < 4 or any(n < 2 for n in args.meshes)
            or any(b <= a for a, b in zip(args.meshes, args.meshes[1:]))):
        parser.error("provide at least four strictly increasing interval counts N >= 2")
    if not 3 <= args.fit_last <= len(args.meshes):
        parser.error("fit-last must be between 3 and the number of meshes")

    root = Path(__file__).resolve().parents[1]
    result_dir, figure_dir = root / "results", root / "figures"
    for directory in (result_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    print("Recorded predictions and equations: docs/model.md")
    for message in verify_implementation():
        print(message)
    rows = run_study(args.meshes, result_dir, args.fit_last, figure_dir)
    solution_figures(args.meshes[-1], figure_dir, result_dir)

    import matplotlib
    metadata = {
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "meshes": args.meshes,
        "beta": 2.0, "fit_last": args.fit_last,
        "unknown_count": "N*(N-1); N is intervals per coordinate",
        "linear_solver": "scipy.sparse.linalg.spsolve (direct, no tolerance parameter)",
        "independent_check": "row-scaled BiCGSTAB rtol=1e-12 on each finest system",
        "timing_scope": "No performance comparison is made by this baseline.",
        "claim_scope": "Smooth MMS benchmark; not proof for general mixed-boundary data."
    }
    (result_dir / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nCompleted {len(rows)} cases. Tables: {result_dir}; figures: {figure_dir}")


if __name__ == "__main__":
    main()
