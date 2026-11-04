"""独立的张量积组装对照；不调用主求解器的组装函数。

ghost 使用 reference_hua_ghost.py 中按固定提交原样摘录的函数。
first_order 在此独立构造导数行替换最后一个 x 节点块，不修改参考函数。
这检查实现差异，不宣称是另一个独立的 PDE 离散理论。
"""
import numpy as np
from scipy.sparse import csr_matrix, eye, kron, vstack

from reference_hua_ghost import (
    BoxDomain, Dimension, fdm_neg_lapl_unif_disc_mat,
    make_coordinate_grids, manufactured_boundary_terms,
)


def assemble_kron(N, bc_type, closure, data, beta=2.0):
    """与主程序使用相同编号，返回未经整体缩放的 A,b。"""
    gamma = beta if bc_type == "robin" else 0.0
    domain = BoxDomain([Dimension(0., 1., N, None, gamma),
                        Dimension(0., 1., N, None, None)])
    coords = make_coordinate_grids(domain)
    A = csr_matrix(fdm_neg_lapl_unif_disc_mat(domain))
    x, y = coords
    if closure == "ghost":
        # Only the x derivative is used on the Robin face; the y components
        # are dummy zeros because y=0,1 are both Dirichlet in this project.
        def gradient(c):
            return np.array([data.q(c[1]) + np.zeros_like(c[0] + c[1]),
                             np.zeros_like(c[0] + c[1])])
        terms = manufactured_boundary_terms(
            domain, coords, lambda c: data.exact(*c), gradient)
        b = np.asarray(data.forcing(x, y) + np.zeros_like(x) + terms).ravel()
    elif closure == "first_order":
        h, ny = 1./N, N-1
        # The derivative row has no tangential Laplacian. It REPLACES the PDE.
        derivative = csr_matrix(([-1/h, 1/h + gamma], ([0, 0], [N-2, N-1])),
                                shape=(1, N))
        A = vstack([A[:-ny, :], kron(derivative, eye(ny))], format="csr")
        b = np.asarray(data.forcing(x, y) + np.zeros_like(x)).copy()
        b[0, :] += data.exact(0., y[0, :]) / h**2
        b[:, 0] += data.exact(x[:, 0], 0.) / h**2
        b[:, -1] += data.exact(x[:, -1], 1.) / h**2
        # Override AFTER Dirichlet additions: derivative rows do not use corners.
        b[-1, :] = data.q(y[-1, :]) + gamma * data.exact(1., y[-1, :])
        b = b.ravel()
    else:
        raise ValueError("unknown closure")
    return A, b
