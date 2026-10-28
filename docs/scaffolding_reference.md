# Reference map for the course scaffolding

Reviewed 2026-10-28. Sources are the user's existing files in the learning-material
folder `03_Scaffolding/`; a file being there does not establish its original author.
`scaffolding_inventory.json` records source hashes and public functions at review time.
The learning files are not imported at runtime, so teammates do not need that folder.

| Existing file | Use in this project |
| --- | --- |
| mixed_boundary_fd.py | Primary numerical baseline: model data, indexing, boundary closures, norms, residual scaling, checks and refinement driver. Copied with attribution and reviewed against the problem pack. |
| poisson_fd_skeleton.py | Reference for interior five-point assembly and Dirichlet RHS signs. Its n means interior points, unlike the new N meaning intervals. The current source contains `import plt`, so it is not imported into the team baseline. |
| poisson_fd_setup_check.py | Independent training/environment check for the zero-Dirichlet P1 problem. |
| test_problem_bank.py | Analytic P1-P5 benchmarks. P3 places Neumann data on the TOP edge and uses different data; it is not the right-edge exp(x+y) problem required for Direction 2. |
| convergence_plot_template.py | Log-log slopes, declared fitting windows, fit deviations and tolerance separation. The current local version uses measured rather than synthetic data. Its CG example is for a different symmetric Dirichlet matrix, not our nonsymmetric mixed matrix. |
| iterative_solvers_skeleton.py | Jacobi, Gauss-Seidel, SOR and residual-based stopping, useful for later solver experiments. The square Dirichlet SOR optimum is not automatically valid for this mixed system. |
| fem1d_skeleton.py | Local element matrices, nodal load quadrature, global assembly and boundary elimination. One-dimensional FEM cannot serve as a matched comparison with our 2D solver. |
| fem2d_scikitfem_skeleton.py | Possible future independent 2D FEM benchmark with explicit weak form and error norms; core Direction 2 still uses prescribed FDM closures. |
| fem_amg_demo.py | Training example for FEM/AMG and symmetric Dirichlet condensation, not a drop-in CG solver for our matrix. |
| failure_cases.py | Boundary-sign and pure-Neumann compatibility pitfalls. The local `np.trapz` example may require adaptation for the installed NumPy; it is not a baseline dependency. |
| domain_mask_skeleton.py | Unfinished L-shaped domain mask/index-map training. Not needed for the current square. |
| multigrid_simple.py | 1D multigrid training and work/iteration scaling; not a ready-made mixed 2D solver. |
| mems_skeleton.py | Nonlinear Newton/Jacobian/continuation training with placeholders; outside current project scope. |

Changes to the primary baseline: default four meshes 8,16,32,64; separate numerical
results and figures; repository-relative output paths; a unified run_all.py entry;
additional solution/error plots, saved nodal arrays, environment metadata and tests.
The original learning-material files remain unchanged.
