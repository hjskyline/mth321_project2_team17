"""可计时的线性求解器；所有方法求解同一个正对角行缩放系统。

原始混合边界矩阵通常非对称，因此这里使用 GMRES，而不是直接套用 CG。
缩放 B=D^{-1}A, c=D^{-1}b 不改变解。tol 指 SciPy 对缩放系统使用的
相对二范数残差容差；不能将 tol 当成解误差。解误差由实验中的 LU 对照检查。
"""
from time import perf_counter

import numpy as np
import pyamg
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import gmres, splu


METHODS = ("lu", "gmres", "amg_gmres")


def difference_norms(system, difference):
    """两个离散解的差：Dirichlet 处差为零，右边界使用半面积权重。"""
    d = np.asarray(difference).reshape(system.N, system.N - 1)
    weights = np.ones_like(d)
    weights[-1, :] = 0.5
    return {"L2": float(system.h * np.sqrt(np.sum(weights * d*d))),
            "Linf": float(np.max(np.abs(d)))}


def solve_linear(system, method, rtol=1e-10, restart=80, max_cycles=100):
    """返回 (解, 测量信息, 回调残差历史)，不把未收敛运行伪装成成功。

    timing_scope: 缩放和稀疏格式转换 + 分解/AMG 建立 + 求解。
    矩阵组装由调用方计时；下方结束计时之后的残差检查不计入求解时间。
    GMRES 的 maxiter 是 restart cycles，不是内迭代次数。
    callback_type='pr_norm' 每个内迭代触发一次；AMG 的回调值是预条件
    残差，与无 AMG 的曲线范数不同，必须在图和表中明确标注。
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not 0 < rtol < 1 or restart < 1 or max_cycles < 1:
        raise ValueError("need 0 < rtol < 1 and positive iteration limits")
    start = perf_counter()
    diagonal = np.asarray(system.A.diagonal())
    if np.any(diagonal <= 0):
        raise ValueError("positive diagonal required for shared row scaling")
    scale = 1.0 / diagonal
    B = csr_matrix(diags(scale) @ system.A)
    c = scale * system.b
    factor = hierarchy = preconditioner = None
    preprocessing_s = perf_counter() - start

    setup_start = perf_counter()
    if method == "lu":
        # CSC conversion is included in LU setup, rather than hidden outside timing.
        factor = splu(B.tocsc())
    elif method == "amg_gmres":
        # Explicit settings keep the library boundary reproducible and explainable.
        hierarchy = pyamg.ruge_stuben_solver(
            B, strength=("classical", {"theta": 0.25}),
            CF=("RS", {"second_pass": False}), interpolation="classical",
            presmoother=("gauss_seidel", {"sweep": "symmetric"}),
            postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
            max_levels=30, max_coarse=10, coarse_solver="pinv")
        preconditioner = hierarchy.aspreconditioner(cycle="V")
    setup_s = perf_counter() - setup_start
    history = []
    solve_start = perf_counter()
    if method == "lu":
        x, info = factor.solve(c), 0
    else:
        x, info = gmres(
            B, c, x0=np.zeros_like(c), M=preconditioner,
            rtol=rtol, atol=0.0, restart=restart, maxiter=max_cycles,
            callback=lambda value: history.append(float(value)),
            callback_type="pr_norm")
    solve_s = perf_counter() - solve_start
    # Count storage by factor/hierarchy nonzeros; this is not total process RAM.
    factor_nnz = factor.L.nnz + factor.U.nnz if factor is not None else 0
    hierarchy_nnz = (sum(level.A.nnz for level in hierarchy.levels)
                     if hierarchy is not None else 0)
    transfer_nnz = (sum(level.P.nnz + level.R.nnz for level in hierarchy.levels[:-1])
                    if hierarchy is not None else 0)
    scaled_res = float(np.linalg.norm(c - B @ x) / max(np.linalg.norm(c), 1e-300))
    return x, {
        "method": method, "rtol": "N/A" if method == "lu" else rtol,
        "restart": "N/A" if method == "lu" else restart,
        "max_restart_cycles": "N/A" if method == "lu" else max_cycles,
        "info": int(info), "converged": bool(info == 0 and np.all(np.isfinite(x))),
        "iterations": len(history), "preprocessing_s": preprocessing_s,
        "setup_s": setup_s, "solve_s": solve_s,
        "solver_total_s": preprocessing_s + setup_s + solve_s,
        "scaled_relative_residual_2": scaled_res,
        "factor_nnz": int(factor_nnz), "hierarchy_A_nnz": int(hierarchy_nnz),
        "hierarchy_transfer_nnz": int(transfer_nnz),
        "operator_complexity": float(hierarchy_nnz / B.nnz) if hierarchy is not None else 0.,
        "levels": len(hierarchy.levels) if hierarchy is not None else 0,
        "callback_norm": ("not applicable" if method == "lu" else
                          "scipy pr_norm: relative preconditioned residual 2-norm"),
    }, history
