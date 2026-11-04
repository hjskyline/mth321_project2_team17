"""统一复现：python code/run_all.py。先验证，再运行收敛/容差/反证/成本实验。

默认 N=16,32,64,128；每个计时配置重新组装和建立五次，使用一个 BLAS 线程。
可用 --output-dir 保存到独立输出目录；--baseline-only 只运行基础与组装验证。
自动化证据不代替课程规定的学生理解、审查和签字。
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

# 在导入数值库前固定线程，避免线程数量差异污染本机计时。
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[variable]='1'
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits, threadpool_info
from mixed_boundary_fd import assemble_mixed_poisson, run_study, verify_implementation
from experiment_suite import (
    assembly_and_structure_checks, convergence_details, tolerance_study,
    falsification_study, benchmark, write_summary, CASES,
)


class Tee:
    """终端与日志保存同一结果，避免手抄数字造成差异。"""
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
    def flush(self):
        for stream in self.streams:stream.flush()


def solution_figures(N,figure_dir,result_dir):
    """用二维场定位误差，颜色标尺明确区分各组范围。"""
    import matplotlib.pyplot as plt
    axis=np.linspace(0.,1.,N+1);X,Y=np.meshgrid(axis,axis,indexing='ij')
    exact=np.exp(X+Y)
    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    efig,eaxes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    archive={'x':axis,'y':axis,'exact':exact}
    for ax,eax,(bc,closure) in zip(axes.flat,eaxes.flat,CASES):
        s=assemble_mixed_poisson(N,bc,closure);U=s.full_solution(spsolve(s.A,s.b))
        error=U-exact;archive[f'{bc}_{closure}']=U
        cf=ax.contourf(X,Y,U,levels=np.linspace(1.,np.e**2,21),extend='both',cmap='viridis')
        fig.colorbar(cf,ax=ax,label='u_h')
        maximum=max(float(np.max(np.abs(error))),np.finfo(float).eps)
        ecf=eax.contourf(X,Y,error,levels=np.linspace(-maximum,maximum,21),cmap='coolwarm')
        efig.colorbar(ecf,ax=eax,label='u_h-u*')
        ax.set_title(f'{bc} / {closure}')
        eax.set_title(f'{bc} / {closure}: max |error|={maximum:.3e}',fontsize=10)
        for panel in (ax,eax):panel.set(xlabel='x',ylabel='y',aspect='equal')
    fig.suptitle(f'Mixed-boundary MMS: N={N} intervals')
    efig.suptitle('Signed errors; each panel uses its own stated colour scale')
    fig.savefig(figure_dir/'solution_comparison.png',dpi=150);plt.close(fig)
    efig.savefig(figure_dir/'error_comparison.png',dpi=150);plt.close(efig)
    np.savez_compressed(result_dir/'baseline_solutions.npz',**archive)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meshes',nargs='+',type=int,default=[16,32,64,128])
    parser.add_argument('--fit-last',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output-dir',type=Path,help='a separate root for fresh outputs')
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args()
    if (len(args.meshes)<4 or any(N<4 for N in args.meshes) or
            any(b!=2*a for a,b in zip(args.meshes,args.meshes[1:]))):
        parser.error('use >=4 factor-two refinements, starting at N>=4')
    if not 3<=args.fit_last<=len(args.meshes) or args.repeats<1:
        parser.error('fit-last must select >=3 levels; repeats must be positive')
    root=Path(__file__).resolve().parents[1]
    output=args.output_dir.resolve() if args.output_dir else root
    results,figures=output/'results',output/'figures'
    results.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    original_stdout=sys.stdout
    with (results/'run_log.txt').open('w',encoding='utf-8') as log:
        sys.stdout=Tee(original_stdout,log)
        try:
            with threadpool_limits(limits=1):
                print('Predictions and scope: docs/experiment_protocol.md',flush=True)
                test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'code'),
                                     '-p','test_*.py','-v'],capture_output=True,text=True)
                (results/'unit_tests.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
                if test.returncode:raise RuntimeError('unit tests failed; see unit_tests.txt')
                print('PASS unit tests',flush=True)
                for message in verify_implementation():print(message)
                assembly_and_structure_checks(results)
                rows=run_study(args.meshes,results,args.fit_last,figures)
                fits=convergence_details(rows,args.meshes,results,figures,args.fit_last)
                solution_figures(args.meshes[-1],figures,results)
                if not args.baseline_only:
                    tolerances,chosen=tolerance_study(args.meshes[-1],results,figures)
                    falsification_study(results,figures)
                    performance,targets=benchmark(args.meshes,args.repeats,chosen,results,figures)
                    write_summary(rows,fits,tolerances,performance,targets,results)
                try:
                    parent=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,
                                          text=True,check=True).stdout.strip()
                except (OSError,subprocess.CalledProcessError):parent='not a git checkout'
                metadata={
                    'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
                    'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
                    'meshes':args.meshes,'fit_last':args.fit_last,'timing_repeats':args.repeats,
                    'baseline_only':args.baseline_only,'blas_thread_limit':1,
                    'threadpools':threadpool_info(),'checkout_HEAD_at_run':parent,
                    'source_identity_note':'code hashes identify working files; HEAD may be the parent before commit',
                    'packages':{name:importlib.metadata.version(name) for name in
                                ('numpy','scipy','matplotlib','pyamg','threadpoolctl')},
                    'code_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((root/'code').glob('*.py'))},
                    'cost_includes':'assembly + diagonal scaling + conversion + factorization/AMG setup + solve',
                    'cost_excludes':'imports, warmups, reference solutions, calibration, validation, plotting, I/O',
                    'algebraic_acceptance':'both solution differences / LU-based MMS errors <= 1e-3',
                    'accuracy_targets_Linf':[1e-2,1e-3],
                    'claim_scope':'smooth MMS; no real-world calibration or proof of AMG asymptotic complexity',
                }
                (results/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
                print(f'COMPLETE: {len(rows)} MMS rows; outputs: {output}',flush=True)
        finally:sys.stdout=original_stdout
    # Exclude the manifest itself to avoid a self-referential hash.
    manifest={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
              for directory in (results,figures) for p in sorted(directory.iterdir())
              if p.is_file() and p.name!='output_manifest.json'}
    (results/'output_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
