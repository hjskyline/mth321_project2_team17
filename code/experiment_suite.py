"""项目实验：独立组装、局部/全局误差、容差、反证与匹配精度成本。

数据一律来自实际求解。预期阶数只用于解释，不写入误差数组、不强迫拟合。
所有图表在 results/ 和 figures/ 留有对应数据；原 notebook 不作为运行依赖。
"""
from dataclasses import replace
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import norm as sparse_norm, spsolve
import matplotlib.pyplot as plt

from mixed_boundary_fd import (
    MMS, ManufacturedData, assemble_mixed_poisson, error_metrics,
    observed_order, residual_metrics,
)
from kron_reference import assemble_kron
from linear_solvers import METHODS, difference_norms, solve_linear


CASES = [(bc, closure) for bc in ("neumann", "robin")
         for closure in ("first_order", "ghost")]
ERROR_KEYS = ("L2", "Linf", "interior_L2", "interior_Linf", "right_L2", "right_Linf")
ALGEBRAIC_FRACTION = 1e-3  # 本项目选定的验收值，不等同于库的 rtol。
TOLERANCES = (1e-6, 1e-8, 1e-10, 1e-12)


def save_csv(path, rows):
    """保留每一行，包括失败/未达到目标的行；不能只导出成功数据。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def max_sparse(matrix):
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.


def assembly_and_structure_checks(result_dir):
    """不同组装路径对照 + 手写行导出 + 低阶精确/非多项式补充验证。"""
    affine = ManufacturedData(lambda x, y: 1+x+2*y, lambda x, y: 0.,
                              lambda y: 1., "affine")
    companion = ManufacturedData(
        lambda x, y: np.sin(np.pi*x)*np.cos(np.pi*y)+x+y*y,
        lambda x, y: 2*np.pi**2*np.sin(np.pi*x)*np.cos(np.pi*y)-2,
        lambda y: 1-np.pi*np.cos(np.pi*y), "sine_plus_polynomial")
    checks, rows, structures = [], [], []
    for bc, closure in CASES:
        for data in (MMS, affine, companion):
            for N in (2, 4, 8, 16):
                s = assemble_mixed_poisson(N, bc, closure, data=data)
                A2, b2 = assemble_kron(N, bc, closure, data)
                a_diff, b_diff = max_sparse(s.A-A2), float(np.max(np.abs(s.b-b2)))
                np.testing.assert_allclose(b2, s.b, atol=1e-10, rtol=1e-13)
                if a_diff > 1e-12:
                    raise AssertionError("COO and Kronecker matrices disagree")
                u1, u2 = spsolve(s.A, s.b), spsolve(A2, b2)
                difference = float(np.max(np.abs(u1-u2)))
                if difference > 1e-10:
                    raise AssertionError("independently assembled solutions disagree")
                checks.append(dict(bc_type=bc, closure=closure, data=data.name, N=N,
                                   matrix_max_difference=a_diff, rhs_max_difference=b_diff,
                                   solution_max_difference=difference, passed=True))
        # Coefficient dump includes all nonzeros and RHS, not a claimed human signature.
        s = assemble_mixed_poisson(4, bc, closure)
        for i, j, kind in ((1, 1, "Dirichlet-adjacent"), (2, 2, "interior"),
                              (4, 1, "right-bottom-adjacent"), (4, 2, "right-middle"),
                              (4, 3, "right-top-adjacent")):
            row = s.A.getrow(s.index(i, j))
            rows.append(dict(bc_type=bc, closure=closure, N=4, i=i, j=j, kind=kind,
                             row=s.index(i,j), columns=json.dumps(row.indices.tolist()),
                             coefficients=json.dumps(row.data.tolist()), rhs=float(s.b[s.index(i,j)])))
        for N in (2, 4, 8, 16):
            s = assemble_mixed_poisson(N, bc, closure)
            diagonal = s.A.diagonal()
            off = s.A - diags(diagonal)
            margin = diagonal-np.asarray(abs(off).sum(axis=1)).ravel()
            components = connected_components(off, directed=True, connection="strong", return_labels=False)
            # This checks the assumptions used in docs/theory.md; proof is separate.
            assert np.all(diagonal > 0) and np.all(off.data <= 0)
            assert np.min(margin) >= -1e-10 and np.max(margin) > 0 and components == 1
            weights = np.ones(s.A.shape[0])
            weights[-(N-1):] = 1/s.h if closure == "first_order" else .5
            B = diags(weights) @ s.A
            assert max_sparse(B-B.T) < 1e-12
            x, _ = s.coordinates()
            barrier = (2*x-.5*x*x).ravel()
            assert np.min(s.A @ barrier) >= 1-1e-10
            eigenvalue = float(np.linalg.eigvalsh(B.toarray())[0]) if N <= 8 else float("nan")
            if N <= 8:
                assert eigenvalue > 0
            structures.append(dict(bc_type=bc, closure=closure, N=N,
                asymmetry_fro=float(sparse_norm(s.A-s.A.T)/sparse_norm(s.A)),
                min_diagonal_dominance=float(np.min(margin)), strong_components=components,
                weighted_symmetry_max=max_sparse(B-B.T), small_min_eigenvalue=eigenvalue,
                barrier_min_action=float(np.min(s.A@barrier)), passed=True))
    save_csv(result_dir/"assembly_crosscheck.csv", checks)
    save_csv(result_dir/"hand_rows.csv", rows)
    save_csv(result_dir/"matrix_checks.csv", structures)
    print(f"PASS independent assembly: {len(checks)} cases; structure: {len(structures)} cases", flush=True)
    return checks


def convergence_details(core_rows, meshes, result_dir, figure_dir, fit_last=3):
    """输出所有区域范数、截断残差和声明的拟合区间；不自动删除偏离预期的数据。"""
    fits, truncations, geometry = [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(11,8), constrained_layout=True)
    section_fig, section_axes = plt.subplots(2, 2, figsize=(11,8), constrained_layout=True)
    for panel, (bc, closure) in enumerate(CASES):
        case = [r for r in core_rows if (r["bc_type"],r["closure"]) == (bc,closure)]
        hs = np.array([r["h"] for r in case])
        for key in ERROR_KEYS:
            errs = np.array([r[key] for r in case])
            rates = observed_order(hs, errs)
            p, intercept = np.polyfit(np.log(hs[-fit_last:]), np.log(errs[-fit_last:]), 1)
            fit_deviation = float(np.max(np.abs(np.log(errs[-fit_last:]) - (p*np.log(hs[-fit_last:])+intercept))))
            stable = bool(np.all(np.diff(errs) < 0) and
                          np.ptp(rates[-(fit_last-1):]) < .2)
            fits.append(dict(bc_type=bc,closure=closure,norm=key,fit_N=json.dumps(meshes[-fit_last:]),
                             slope=float(p),max_log_deviation=fit_deviation,
                             monotone=bool(np.all(np.diff(errs)<0)),rates_stable=stable,
                             note="diagnostic window; algebraic gate checked separately"))
        ax = axes.flat[panel]
        for key, marker in (("Linf","o"),("right_Linf","s"),("interior_Linf","^")):
            ax.loglog(hs,[r[key] for r in case],marker+"-",label=key)
        ax.set(xlabel="h",ylabel="Maximum error",title=f"{bc} / {closure}")
        ax.grid(True,which="both",alpha=.25);ax.legend(fontsize=8)
        for N in meshes:
            s = assemble_mixed_poisson(N,bc,closure)
            x,y = s.coordinates(); sampled=s.data.exact(x,y).ravel()
            tau=(s.b-s.A@sampled).reshape(N,N-1)
            truncations.append(dict(bc_type=bc,closure=closure,N=N,h=s.h,
                                    interior_truncation_Linf=float(np.max(np.abs(tau[:-1]))),
                                    right_truncation_Linf=float(np.max(np.abs(tau[-1])))))
        # A signed cross-section separates cancellation from a genuinely rough solution.
        s=assemble_mixed_poisson(meshes[-1],bc,closure)
        u=spsolve(s.A,s.b);x,y=s.coordinates()
        e=u.reshape(s.N,s.N-1)-s.data.exact(x,y)
        middle=s.N//2-1
        sax=section_axes.flat[panel]
        sax.plot(x[:,middle],e[:,middle],label="signed error at y=0.5")
        sax.axhline(0,color="black",lw=.7)
        sax.set(xlabel="x",ylabel="u_h-u*",title=f"{bc} / {closure}")
        sax.grid(alpha=.25);sax.legend(fontsize=8)
        geometry.append(dict(bc_type=bc,closure=closure,N=s.N,
                             minimum_signed_error=float(e.min()),maximum_signed_error=float(e.max()),
                             x_neighbor_sign_changes=int(np.count_nonzero(e[:-1]*e[1:]<0)),
                             y_neighbor_sign_changes=int(np.count_nonzero(e[:,:-1]*e[:,1:]<0))))
    # Per-step orders on residuals are distinct from those of solution errors.
    for bc,closure in CASES:
        subset=[r for r in truncations if (r['bc_type'],r['closure'])==(bc,closure)]
        for key in ('interior_truncation_Linf','right_truncation_Linf'):
            for row,p in zip(subset,observed_order([r['h'] for r in subset],[r[key] for r in subset])):
                row['p_'+key]=float(p)
    save_csv(result_dir/"convergence_fits.csv",fits)
    save_csv(result_dir/"truncation_diagnostics.csv",truncations)
    save_csv(result_dir/"error_geometry.csv",geometry)
    fig.savefig(figure_dir/"regional_errors.png",dpi=150);plt.close(fig)
    section_fig.savefig(figure_dir/"signed_error_sections.png",dpi=150);plt.close(section_fig)
    return fits


def accuracy_record(system, x, reference):
    """用同一范数比较解差与已测 MMS 误差；不是严格的精确前向误差界。"""
    diff=difference_norms(system,x-reference)
    error=error_metrics(system,reference)
    qidx=system.index(system.N,system.N//2)
    return dict(L2_difference=diff['L2'],Linf_difference=diff['Linf'],
                L2_difference_over_MMS=diff['L2']/error['L2'],
                Linf_difference_over_MMS=diff['Linf']/error['Linf'],
                qoi_right_midpoint=float(x[qidx]),qoi_difference=float(abs(x[qidx]-reference[qidx])),
                algebraic_gate=bool(diff['L2']<=ALGEBRAIC_FRACTION*error['L2'] and
                                    diff['Linf']<=ALGEBRAIC_FRACTION*error['Linf']))


def tolerance_study(N, result_dir, figure_dir):
    """在最细拟合网格收紧容差，选择实际通过验收的最宽松已测值。"""
    rows, histories, chosen = [], [], {}
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for panel,(bc,closure) in enumerate(CASES):
        s=assemble_mixed_poisson(N,bc,closure);ref=spsolve(s.A,s.b)
        for method in METHODS[1:]:
            candidates=[]
            for tol in TOLERANCES:
                x,meta,history=solve_linear(s,method,rtol=tol)
                row=dict(bc_type=bc,closure=closure,N=N,**meta,
                         **accuracy_record(s,x,ref),**residual_metrics(s,x),
                         **{f'PDE_{k}':v for k,v in error_metrics(s,x).items()})
                rows.append(row);candidates.append(row)
                if tol==TOLERANCES[-1]:
                    for k,value in enumerate(history,1):
                        histories.append(dict(bc_type=bc,closure=closure,N=N,method=method,
                                              inner_iteration=k,callback_pr_norm=value,
                                              definition=meta['callback_norm']))
            if not candidates[-1]['converged'] or not candidates[-1]['algebraic_gate']:
                raise RuntimeError(f'tight reference check failed: {bc}/{closure}/{method}')
            working=next(r for r in candidates if r['converged'] and r['algebraic_gate'])
            chosen[(bc,closure,method)]=working['rtol']
            ratios=[max(r['L2_difference_over_MMS'],r['Linf_difference_over_MMS']) for r in candidates]
            ax=axes.flat[panel]
            ax.loglog(TOLERANCES,ratios,'o-',label=method)
        ax=axes.flat[panel]
        ax.axhline(ALGEBRAIC_FRACTION,color='black',ls='--',label='acceptance 1e-3')
        ax.set(xlabel='Requested rtol (scaled system)',ylabel='max(solution difference / MMS error)',
               title=f'{bc} / {closure}, N={N}')
        ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=8)
    save_csv(result_dir/'tolerance_sweep.csv',rows)
    save_csv(result_dir/'solver_residual_history.csv',histories)
    fig.savefig(figure_dir/'tolerance_sweep.png',dpi=150);plt.close(fig)
    hfig,haxes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,(bc,closure) in zip(haxes.flat,CASES):
        for method in METHODS[1:]:
            h=[r for r in histories if (r['bc_type'],r['closure'],r['method'])==(bc,closure,method)]
            ax.semilogy([r['inner_iteration'] for r in h],[r['callback_pr_norm'] for r in h],label=method)
        ax.set(title=f'{bc} / {closure}',xlabel='GMRES inner iterations',ylabel='SciPy callback pr_norm')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    hfig.suptitle('Callback norms differ with preconditioning; final true residual is in the CSV',fontsize=10)
    hfig.savefig(figure_dir/'solver_residual_history.png',dpi=150);plt.close(hfig)
    print(f'PASS tolerance separation at N={N} for all four cases and both iterative paths',flush=True)
    return rows,chosen


def falsification_study(result_dir, figure_dir):
    """只在局部副本引入错误；错误矩阵永远不进入正式求解函数。"""
    rows=[]
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    variants=('correct','wrong_normal_sign','missing_inward_factor','omit_boundary_forcing')
    for variant in variants:
        errs=[]
        for N in (16,32,64):
            good=assemble_mixed_poisson(N,'neumann','ghost')
            if variant=='wrong_normal_sign':
                bad=replace(MMS,q=lambda y:-np.exp(1+np.asarray(y)))
                s=assemble_mixed_poisson(N,'neumann','ghost',data=bad)
            else:
                s=replace(good,A=good.A.copy(),b=good.b.copy())
                if variant=='missing_inward_factor':
                    A=s.A.tolil()
                    for j in range(1,N):
                        A[s.index(N,j),s.index(N-1,j)]+=1/s.h**2
                    s=replace(s,A=A.tocsr())
                elif variant=='omit_boundary_forcing':
                    s.b[-(N-1):]-=MMS.forcing(1.,np.arange(1,N)/N)
            x=spsolve(s.A,s.b)
            # Measure against the ORIGINAL prescribed exact solution.
            error=error_metrics(good,x)['Linf']
            base=error_metrics(good,spsolve(good.A,good.b))['Linf']
            rows.append(dict(variant=variant,N=N,h=s.h,Linf=error,error_over_correct=error/base,
                             small_residual_of_own_equation=residual_metrics(s,x)['row_scaled_residual_inf'],
                             detected=variant!='correct' and error>5*base))
            if variant!='correct' and not error>5*base:
                raise AssertionError('falsifier did not expose a measurable defect')
            errs.append(error)
        ax.loglog([1/16,1/32,1/64],errs,'o-',label=variant)
    # Another plausible failure: a solver can stop successfully at a loose tol.
    s=assemble_mixed_poisson(64,'neumann','ghost');ref=spsolve(s.A,s.b)
    x,meta,_=solve_linear(s,'gmres',rtol=1e-2)
    record=accuracy_record(s,x,ref)
    assert not record['algebraic_gate']
    rows.append(dict(variant='loose_tolerance',N=64,h=s.h,Linf=error_metrics(s,x)['Linf'],
                     detected=True,**meta,**record))
    ax.set(xlabel='h',ylabel='Global maximum error',title='Wrong codes can have small algebraic residuals')
    ax.legend(fontsize=8);ax.grid(True,which='both',alpha=.25)
    fig.savefig(figure_dir/'falsification.png',dpi=150);plt.close(fig)
    save_csv(result_dir/'falsification.csv',rows)
    print('PASS falsification: wrong normal, inward coefficient, omitted forcing and loose tolerance detected',flush=True)
    return rows


def benchmark(meshes,repeats,chosen,result_dir,figure_dir,targets=(1e-2,1e-3)):
    """单 RHS、零初值、每次重新组装/建立。参考解和验收检查不计入生产成本。

    所有迭代方法在实际运行中满足相同的前向解差/MMS目标；容差先校准再固定。
    除准确性外还记录 info；失败运行会保留在 pilot 表且不能算合格计时。
    """
    raw,pilots,summary=[],[],[]
    warm=assemble_mixed_poisson(meshes[0])
    for method in METHODS: solve_linear(warm,method,rtol=1e-10)
    for bc,closure in CASES:
        for N in meshes:
            ref_system=assemble_mixed_poisson(N,bc,closure)
            reference=spsolve(ref_system.A,ref_system.b)
            for method in METHODS:
                tol=1e-12 if method=='lu' else chosen[(bc,closure,method)]
                # Calibration cost is excluded from per-production-solve time, and disclosed.
                for attempt in range(4):
                    x,meta,_=solve_linear(ref_system,method,rtol=tol)
                    a=accuracy_record(ref_system,x,reference)
                    pilots.append(dict(bc_type=bc,closure=closure,N=N,attempt=attempt,**meta,**a))
                    if meta['converged'] and a['algebraic_gate']: break
                    tol*=.01
                else:
                    save_csv(result_dir/'performance_calibration.csv',pilots)
                    raise RuntimeError(f'cannot calibrate {bc}/{closure}/{N}/{method}')
                samples=[]
                for rep in range(repeats):
                    start=perf_counter();s=assemble_mixed_poisson(N,bc,closure)
                    assembly_s=perf_counter()-start
                    x,meta,_=solve_linear(s,method,rtol=tol)
                    a=accuracy_record(s,x,reference)
                    if not meta['converged'] or not a['algebraic_gate']:
                        raise RuntimeError('measured run failed common accuracy criterion')
                    row=dict(bc_type=bc,closure=closure,N=N,h=s.h,unknowns=s.A.shape[0],nnz=s.A.nnz,
                             repeat=rep+1,assembly_s=assembly_s,total_s=assembly_s+meta['solver_total_s'],
                             **meta,**a,PDE_Linf=float(error_metrics(s,x)['Linf']),
                             backward_error=residual_metrics(s,x)['componentwise_backward_error'])
                    raw.append(row);samples.append(row)
                group=dict(bc_type=bc,closure=closure,N=N,h=1/N,method=method,unknowns=N*(N-1),
                           nnz=ref_system.A.nnz,rtol=samples[0]['rtol'],repeats=repeats,
                           all_passed=True,max_PDE_Linf=max(r['PDE_Linf'] for r in samples),
                           max_algebraic_fraction=max(max(r['L2_difference_over_MMS'],r['Linf_difference_over_MMS']) for r in samples),
                           median_iterations=float(np.median([r['iterations'] for r in samples])),
                           factor_nnz=samples[0]['factor_nnz'],hierarchy_A_nnz=samples[0]['hierarchy_A_nnz'],
                           hierarchy_transfer_nnz=samples[0]['hierarchy_transfer_nnz'])
                for key in ('assembly_s','preprocessing_s','setup_s','solve_s','solver_total_s','total_s'):
                    values=[r[key] for r in samples]
                    group['median_'+key]=float(np.median(values))
                    group['min_'+key]=float(min(values));group['max_'+key]=float(max(values))
                summary.append(group)
            print(f'Benchmarked {bc}/{closure}, N={N}, {repeats} fresh setups per method',flush=True)
    save_csv(result_dir/'performance_raw.csv',raw)
    save_csv(result_dir/'performance_calibration.csv',pilots)
    save_csv(result_dir/'performance_summary.csv',summary)
    target_rows=[]
    for bc,closure in CASES:
        for target in targets:
            for method in METHODS:
                candidates=[r for r in summary if (r['bc_type'],r['closure'],r['method'])==(bc,closure,method)
                            and r['max_PDE_Linf']<=target]
                found=min(candidates,key=lambda r:r['N']) if candidates else None
                target_rows.append(dict(bc_type=bc,closure=closure,method=method,target_Linf=target,
                    met=found is not None,first_tested_N=found['N'] if found else '',
                    unknowns=found['unknowns'] if found else '',
                    measured_Linf=found['max_PDE_Linf'] if found else '',
                    median_total_s=found['median_total_s'] if found else '',
                    note='first successful TESTED mesh, not an interpolated optimum' if found else f'not reached on N={meshes}'))
    save_csv(result_dir/'accuracy_targets.csv',target_rows)
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    costfig,costaxes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,cax,(bc,closure) in zip(axes.flat,costaxes.flat,CASES):
        for method in METHODS:
            group=[r for r in summary if (r['bc_type'],r['closure'],r['method'])==(bc,closure,method)]
            median=np.array([r['median_total_s'] for r in group])
            spread=np.array([median-[r['min_total_s'] for r in group],
                             [r['max_total_s'] for r in group]-median])
            ax.errorbar([r['unknowns'] for r in group],median,yerr=spread,marker='o',label=method,capsize=2)
            cax.loglog([r['max_PDE_Linf'] for r in group],median,'o-',label=method)
        ax.set(xscale='log',yscale='log',xlabel='Unknowns',ylabel='Assembly + setup + solve (s)',title=f'{bc} / {closure}')
        cax.set(xlabel='Measured global maximum error',ylabel='Assembly + setup + solve (s)',title=f'{bc} / {closure}')
        for panel in (ax,cax):panel.grid(True,which='both',alpha=.25);panel.legend(fontsize=8)
    fig.savefig(figure_dir/'solver_cost.png',dpi=150);plt.close(fig)
    costfig.savefig(figure_dir/'accuracy_cost.png',dpi=150);plt.close(costfig)
    return summary,target_rows


def write_summary(core_rows,fits,tolerances,performance,targets,result_dir):
    """从保存前的同一结果对象生成可读结论，不手工粘贴实验数字。"""
    lines=['# 自动生成的实验结果摘要','',
           '数据由 code/run_all.py 实际计算；人工解释和课程审核仍需全组完成。','',
           '## 1. 四组合的全局收敛阶','',
           '| 边界 | 闭合 | L2 拟合阶 | Linf 拟合阶 | 拟合 N |',
           '|---|---|---:|---:|---|']
    for bc,closure in CASES:
        pair={r['norm']:r for r in fits if (r['bc_type'],r['closure'])==(bc,closure)}
        lines.append(f"| {bc} | {closure} | {pair['L2']['slope']:.5f} | {pair['Linf']['slope']:.5f} | {pair['L2']['fit_N']} |")
    lines+=['','## 2. 最细网格代数误差检查','',
            '下面为 rtol=1e-12 时解差/MMS误差的较大值；不是由残差直接推定的上界。','',
            '| 边界 | 闭合 | 方法 | 比值 | 收敛 |','|---|---|---|---:|---|']
    for r in tolerances:
        if r['rtol']==TOLERANCES[-1]:
            ratio=max(r['L2_difference_over_MMS'],r['Linf_difference_over_MMS'])
            lines.append(f"| {r['bc_type']} | {r['closure']} | {r['method']} | {ratio:.3e} | {r['converged']} |")
    lines+=['','## 3. 实测效率','',
            '单右端、每次重新组装/建立，单个 BLAS 线程；中位数来自重复实验。',
            '不计导入、参考解、容差校准、验收检查、绘图和文件输出。时间不是跨机器固定常数。','']
    for bc,closure in CASES:
        N=max(r['N'] for r in performance)
        group=[r for r in performance if (r['bc_type'],r['closure'],r['N'])==(bc,closure,N)]
        best=min(group,key=lambda r:r['median_total_s'])
        text=', '.join(f"{r['method']}={r['median_total_s']:.4g}s" for r in group)
        lines.append(f"- {bc}/{closure}, N={N}: {text}；本机此网格最快为 {best['method']}。")
    lines+=['','不能从局部规模和一次硬件测试声称 AMG 总是更快或已经证明线性复杂度。','',
            '## 4. 目标精度','',
            '完整明细见 accuracy_targets.csv；未达到目标的组合保留为 met=False。','']
    for r in targets:
        if r['method']=='lu':
            result=f"首个通过的已测 N={r['first_tested_N']}" if r['met'] else r['note']
            lines.append(f"- {r['bc_type']}/{r['closure']}, Linf<={r['target_Linf']}: {result}。")
    lines+=['','## 5. 如何读空间误差图','',
            '绝对误差在有符号误差过零处形成凹槽，对数轴会强调接近零的部分；',
            'signed_error_sections.png 与 error_geometry.csv 支持检查这种抵消。',
            '局部凹槽不是单独的错误或收敛证据，应结合全局/分区范数和截断残差。','',
            '## 6. 限制','',
            '仅验证光滑单位正方形制造解及列出的补充测试；未做真实设备标定、纯 Neumann 或复杂区域。',
            '参考 Kronecker 组装与主程序共享目标公式，不能排除共同的建模错误；手工推导仍必要。',
            '数值稳定性检查不是一般性数学证明。报告中的重要结论须完成人工五关审查。','']
    (result_dir/'EXPERIMENT_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
