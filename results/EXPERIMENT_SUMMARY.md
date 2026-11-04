# 自动生成的实验结果摘要

数据由 code/run_all.py 实际计算；人工解释和课程审核仍需全组完成。

## 1. 四组合的全局收敛阶

| 边界 | 闭合 | L2 拟合阶 | Linf 拟合阶 | 拟合 N |
|---|---|---:|---:|---|
| neumann | first_order | 1.02454 | 1.01915 | [32, 64, 128] |
| neumann | ghost | 2.00067 | 1.99887 | [32, 64, 128] |
| robin | first_order | 1.01495 | 1.00930 | [32, 64, 128] |
| robin | ghost | 2.00159 | 1.99871 | [32, 64, 128] |

## 2. 最细网格代数误差检查

下面为 rtol=1e-12 时解差/MMS误差的较大值；不是由残差直接推定的上界。

| 边界 | 闭合 | 方法 | 比值 | 收敛 |
|---|---|---|---:|---|
| neumann | first_order | gmres | 2.894e-07 | True |
| neumann | first_order | amg_gmres | 6.079e-11 | True |
| neumann | ghost | gmres | 1.755e-04 | True |
| neumann | ghost | amg_gmres | 5.414e-09 | True |
| robin | first_order | gmres | 3.851e-07 | True |
| robin | first_order | amg_gmres | 2.407e-09 | True |
| robin | ghost | gmres | 2.464e-04 | True |
| robin | ghost | amg_gmres | 9.979e-08 | True |

## 3. 实测效率

单右端、每次重新组装/建立，单个 BLAS 线程；中位数来自重复实验。
不计导入、参考解、容差校准、验收检查、绘图和文件输出。时间不是跨机器固定常数。

- neumann/first_order, N=128: lu=0.04815s, gmres=0.6894s, amg_gmres=0.04288s；本机此网格最快为 amg_gmres。
- neumann/ghost, N=128: lu=0.04794s, gmres=0.8541s, amg_gmres=0.0408s；本机此网格最快为 amg_gmres。
- robin/first_order, N=128: lu=0.04843s, gmres=0.5881s, amg_gmres=0.0431s；本机此网格最快为 amg_gmres。
- robin/ghost, N=128: lu=0.04805s, gmres=0.7267s, amg_gmres=0.04244s；本机此网格最快为 amg_gmres。

不能从局部规模和一次硬件测试声称 AMG 总是更快或已经证明线性复杂度。

## 4. 目标精度

完整明细见 accuracy_targets.csv；未达到目标的组合保留为 met=False。

- neumann/first_order, Linf<=0.01: 首个通过的已测 N=128。
- neumann/first_order, Linf<=0.001: not reached on N=[16, 32, 64, 128]。
- neumann/ghost, Linf<=0.01: 首个通过的已测 N=16。
- neumann/ghost, Linf<=0.001: 首个通过的已测 N=16。
- robin/first_order, Linf<=0.01: 首个通过的已测 N=64。
- robin/first_order, Linf<=0.001: not reached on N=[16, 32, 64, 128]。
- robin/ghost, Linf<=0.01: 首个通过的已测 N=16。
- robin/ghost, Linf<=0.001: 首个通过的已测 N=16。

## 5. 如何读空间误差图

绝对误差在有符号误差过零处形成凹槽，对数轴会强调接近零的部分；
signed_error_sections.png 与 error_geometry.csv 支持检查这种抵消。
局部凹槽不是单独的错误或收敛证据，应结合全局/分区范数和截断残差。

## 6. 限制

仅验证光滑单位正方形制造解及列出的补充测试；未做真实设备标定、纯 Neumann 或复杂区域。
参考 Kronecker 组装与主程序共享目标公式，不能排除共同的建模错误；手工推导仍必要。
数值稳定性检查不是一般性数学证明。报告中的重要结论须完成人工五关审查。
