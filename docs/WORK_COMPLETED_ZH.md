# 这次完成了什么，后续怎样使用

本次工作只更新 ziming 分支的实现与证据，不改 hua-nb 的 notebook 或其分支。

## 已实现的实验

1. 规定指数解、Neumann/Robin × 真正单边/虚点四组合，默认 N=16,32,64,128。
2. 全局/内部/右边界六种范数，逐层阶数、拟合窗口与偏差、原始/缩放/后向残差。
3. 手写边界行检查与系数表；仿射和二次精确性；额外解析数据；独立张量组装。
4. 最细网格的直接解、BiCGSTAB、GMRES、AMG-GMRES 对照及逐级容差实验。
5. 故意错法向、少写虚点系数、漏掉边界源项、过松容差的隔离反证实验。
6. 同一精度标准下 LU/GMRES/AMG-GMRES 的组装、建立、求解与总时间；默认五次重复。
7. 同一 PDE 最大误差目标下的实测成本，并保留“已测网格没有达到”的情况。
8. 稳定性与全局阶数的针对性推导草稿；不把数值测试当作人工批准的证明。

## 最先看的文件

- results/EXPERIMENT_SUMMARY.md：从实际结果生成的中文摘要。
- docs/model.md：模型、编号、两条边界行、误差范数。
- docs/theory.md：为何可逆、为何两种闭合可能有不同全局阶数。
- docs/experiment_protocol.md：运行前冻结的预测、参数和验收标准。
- code/run_all.py：统一入口和实验执行顺序。
- code/mixed_boundary_fd.py：主求解器的显式 COO 组装。
- code/linear_solvers.py：计时、共同缩放、GMRES/AMG、收敛状态。
- code/experiment_suite.py：组装对照、容差、反证、成本和图表。
- code/reference_hua_ghost.py：固定提交中四格定义的原样摘录；不执行原 notebook。
- code/kron_reference.py：安全适配参考组装，并独立加入规定的一阶边界行。

## 运行

在项目目录、使用本项目的 .venv：

```sh
.venv/bin/python code/run_all.py
```

一次运行生成 results/ 的表格、日志、元信息与 figures/ 的图。已有同名实验结果会重新生成。
若要保留旧结果并独立复现：

```sh
.venv/bin/python code/run_all.py --output-dir /tmp/pde_reproduction
```

只运行核心验证可加 --baseline-only；该模式不代表已复现成本实验。
时间会随电脑负载变化；数值表应在报告指定有效数字内复现。输出记录库版本、
线程、代码 SHA256 和生成文件哈希，不把 HEAD 的旧父提交冒充当前未提交代码。

## 如何接入参考代码

来源固定为 hua-nb 的 b3466958df84f0faa7d6f9ed79588a77558f05aa。
只摘录区域/矩阵/坐标/边界数据函数。它们的函数体原样保留，清楚标注出处。
不执行其纯 Neumann helper，不使用原来的误差上界估计，不把 False 开关当单边法。
匹配矩阵、右端与坐标后再比较求解器。独立实现仍使用同一数学公式，不能排除共同建模错误。

## 仍由人完成的部分

这些代码和实验为后续讨论提供完整起点。全员仍需学懂并独立核对推导、分析数据、
完成课程五关审查、确认分工和真实贡献，再写英文报告与展示。没有自动编造 ICS、
签名、教师批准或真实团队讨论。此工作不包含实物标定、纯 Neumann 或奇异区域扩展。
