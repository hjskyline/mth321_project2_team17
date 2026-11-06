import os
from collections import namedtuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg as spla
import pyamg
from functools import reduce
import itertools
from tabulate import tabulate


plt.style.use("dark_background")
mpl.rcParams["figure.facecolor"] = "none"
mpl.rcParams["axes.facecolor"] = "none"
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["figure.figsize"] = [12.8, 9.6]
mpl.rcParams["axes3d.xaxis.panecolor"] = (0.95, 0.95, 0.95, 0.125)
mpl.rcParams["axes3d.yaxis.panecolor"] = (0.9, 0.9, 0.9, 0.125)
mpl.rcParams["axes3d.zaxis.panecolor"] = (0.925, 0.925, 0.925, 0.125)
mpl.rcParams["grid.alpha"] = 1.0
mpl.rcParams["grid.color"] = (1.0, 1.0, 1.0, 0.125)
mpl.rcParams["grid.linestyle"] = "-"
mpl.rcParams["grid.linewidth"] = 0.5

os.makedirs("figures", exist_ok=True)


Dimension = namedtuple(
    "Dimension", ["low", "high", "num_divs", "low_robin", "high_robin"]
)
BoxDomain = namedtuple("BoxDomain", ["dimensions"])


def get_df(domain: BoxDomain):
    return np.prod(
        [
            dim.num_divs
            - 1
            + int(dim.low_robin is not None)
            + int(dim.high_robin is not None)
            for dim in domain.dimensions
        ]
    )


def integral_row(dom: BoxDomain):
    return reduce(np.kron, map(integral_row_per_dim, dom.dimensions), [1.0])


def integral_row_per_dim(dim: Dimension):
    interior_size = dim.num_divs - 1
    full_size = (
        interior_size + int(dim.low_robin is not None) + int(dim.high_robin is not None)
    )
    dim_len = dim.high - dim.low
    h = dim_len / dim.num_divs
    assert full_size == interior_size + 2, "only neumann here"

    if full_size == 2:
        # midpt, this is kinda degenerate tho
        return np.full(2, 0.5 * h)
    elif full_size % 2 == 1:
        # 1/2 simpson
        return (h / 3) * np.array(
            [
                1.0 if i in (0, full_size - 1) else 2.0 + 2.0 * (i % 2)
                for i in range(full_size)
            ]
        )
    else:
        # 1/3 simpson and a 3/8 simpson at the end
        return np.concat(
            (
                (h / 3)
                * np.array(
                    [
                        1.0 if i == 0 else 2.0 + 2.0 * (i % 2)
                        for i in range(full_size - 4)
                    ]
                ),
                h * np.array([17 / 24, 9 / 8, 9 / 8, 3 / 8]),
            )
        )


def fdm_neg_lapl_1d_mat(dim: Dimension):
    interior_size = dim.num_divs - 1
    full_size = (
        interior_size + int(dim.low_robin is not None) + int(dim.high_robin is not None)
    )
    dim_len = dim.high - dim.low
    h = dim_len / dim.num_divs
    invhsqr = 1 / (h * h)

    return (
        sparse.block_array(
            (
                [
                    [
                        sparse.csr_array(
                            ([2.0 * (1 + h * dim.low_robin), -2.0], [0, 1], [0, 2]),
                            shape=(1, full_size),
                        )
                    ]
                ]
                if dim.low_robin is not None
                else [[None]]
            )
            + (
                [
                    [
                        sparse.diags_array(
                            [2.0, -1, -1],
                            offsets=np.array([0, 1, -1])
                            + int(dim.low_robin is not None),
                            shape=(interior_size, full_size),
                        )
                    ]
                ]
            )
            + (
                [
                    [
                        sparse.csr_array(
                            (
                                [2.0 * (1 + h * dim.high_robin), -2.0],
                                [full_size - 1, full_size - 2],
                                [0, 2],
                            ),
                            shape=(1, full_size),
                        )
                    ]
                ]
                if dim.high_robin is not None
                else [[None]]
            )
        )
        * invhsqr
    )


def lapl_scale(dom: BoxDomain):
    # just a nice number to scale the system by
    return np.exp2(
        np.floor(
            2.0
            * np.mean(
                np.log2([(dim.high - dim.low) / dim.num_divs for dim in dom.dimensions])
            )
        )
    )


def fdm_neg_lapl_unif_disc_mat(dom: BoxDomain):
    return reduce(
        sparse.kronsum,
        (fdm_neg_lapl_1d_mat(dim) for dim in reversed(dom.dimensions)),
    )


def get_gauge_scale(dom: BoxDomain, gauge):
    # a nice number to scale the gauge row by
    gauge_type = gauge[0]
    if gauge_type == "integral":
        return 2.0 * np.prod([dim.num_divs for dim in dom.dimensions])
    elif gauge_type == "value":
        return 2.0


def get_gauge_row(dom: BoxDomain, gauge):
    gauge_type, gauge_index, gauge_value = gauge
    if gauge_type == "integral":
        return integral_row(dom)
    elif gauge_type == "value":
        gauge_row = np.zeros(get_df(dom))
        gauge_row[gauge_index] = 1.0
        return gauge_row
    raise Error(f"unknown gauge type {gauge_type}")


def make_coordinate_grids(dom: BoxDomain):
    return np.meshgrid(
        *(
            np.linspace(dim.low, dim.high, num=dim.num_divs + 1)[
                slice(
                    1 if dim.low_robin is None else 0,
                    -1 if dim.high_robin is None else None,
                )
            ]
            for dim in dom.dimensions
        ),
        indexing="ij",
    )


def calc_forcing_term(
    forcing_fn, coord_grids, ghost_term_bd=True, domain: BoxDomain = None
):
    forcing_term = forcing_fn(coord_grids)
    if not ghost_term_bd:
        num_dims = len(domain.dimensions)
        for i, dim in enumerate(domain.dimensions):
            for face_idx, robin in [
                (0, dim.low_robin),
                (-1, dim.high_robin),
            ]:
                idx_selector = tuple(
                    face_idx if i == j else slice(None) for j in range(num_dims)
                )
                if robin is not None:
                    # Robin
                    forcing_term[idx_selector] = 0.0

    return forcing_term


def manufactured_boundary_terms(dom: BoxDomain, coords, soln, grad):
    bd_terms = np.zeros_like(coords[0])
    num_dims = len(dom.dimensions)
    for i, dim in enumerate(dom.dimensions):
        dim: Dimension = dim
        dim_len = dim.high - dim.low
        h = dim_len / dim.num_divs
        invh = 1.0 / h
        invhsqr = invh * invh

        for face_idx, robin, bd_coord, normal_sign in [
            (0, dim.low_robin, dim.low, -1.0),
            (-1, dim.high_robin, dim.high, 1.0),
        ]:
            idx_selector = tuple(
                face_idx if i == j else slice(None) for j in range(num_dims)
            )
            face_coords = tuple(
                bd_coord if i == j else grid[idx_selector]
                for j, grid in enumerate(coords)
            )
            if robin is None:
                # Dirichlet
                bd_terms[idx_selector] += invhsqr * soln(face_coords)
            else:
                # Robin
                bd_terms[idx_selector] += (
                    2.0
                    * invh
                    * (robin * soln(face_coords) + normal_sign * grad(face_coords)[i])
                )

    return bd_terms


def get_interior_selector(domain: BoxDomain):
    return tuple(
        slice(
            0 if dim.low_robin is None else 1,
            None if dim.high_robin is None else -1,
        )
        for dim in domain.dimensions
    )


def get_bd_values(field_values, domain: BoxDomain):
    int_selector = get_interior_selector(domain)
    mask = np.full_like(field_values, False)
    mask[int_selector] = True
    return np.ma.MaskedArray(field_values, mask)


def interior_rms_norm(field_values, domain: BoxDomain):
    int_selector = get_interior_selector(domain)
    return np.sqrt(np.mean(np.square(field_values[int_selector])))


def interior_linf_norm(field_values, domain: BoxDomain):
    int_selector = get_interior_selector(domain)
    return np.max(np.abs(field_values[int_selector]))


def bd_rms_norm(field_values, domain: BoxDomain):
    bd_values = get_bd_values(field_values, domain)
    return np.sqrt(np.mean(np.square(bd_values)))


def bd_linf_norm(field_values, domain: BoxDomain):
    bd_values = get_bd_values(field_values, domain)
    return np.max(np.abs(bd_values))


def analyze_manufactured_solution(
    exact_soln, exact_grad, forcing, domain: BoxDomain, ghost_term_bd: bool, gauge=None
):
    coords = make_coordinate_grids(domain)

    scale = lapl_scale(domain)

    poisson_lhs = scale * fdm_neg_lapl_unif_disc_mat(domain)
    poisson_rhs = (
        scale
        * (
            calc_forcing_term(forcing, coords, ghost_term_bd, domain)
            + manufactured_boundary_terms(domain, coords, exact_soln, exact_grad)
        ).ravel()
    )

    if gauge is not None:
        gauge_type, gauge_idx, gauge_value = gauge
        gauge_scale = get_gauge_scale(domain, gauge)
        gauge_row = get_gauge_row(domain, gauge)

        poisson_lhs[gauge_idx, :] = gauge_scale * gauge_row
        poisson_rhs[gauge_idx] = gauge_scale * gauge_value

    inv_lhs_inf_norm = np.reciprocal(spla.norm(poisson_lhs, ord=-np.inf))

    if gauge is None or gauge[0] == "integral":
        approx = spla.spsolve(poisson_lhs, poisson_rhs)
    else:
        # PyAMG is fast but chokes on the integral gauge for some reason
        ml = pyamg.ruge_stuben_solver(poisson_lhs)
        approx = ml.solve(poisson_rhs, tol=np.finfo(np.float64).eps)

    residual = poisson_lhs @ approx - poisson_rhs
    approx = approx.reshape(coords[0].shape)
    error = exact_soln(coords) - approx

    rms_rsdl = np.sqrt(np.mean(np.square(residual)))
    linf_rsdl = np.max(np.abs(residual))
    linf_alg_err_bd = inv_lhs_inf_norm * linf_rsdl
    int_rms_err = interior_rms_norm(error, domain)
    int_linf_err = interior_linf_norm(error, domain)
    bd_rms_err = bd_rms_norm(error, domain)
    bd_linf_err = bd_linf_norm(error, domain)

    return dict(
        ghost_term_bd=ghost_term_bd,
        gauge=gauge,
        domain=domain,
        coords=coords,
        scale=scale,
        poisson_lhs=poisson_lhs,
        inv_lhs_inf_norm=inv_lhs_inf_norm,
        poisson_rhs=poisson_rhs,
        approx=approx,
        residual=residual,
        error=error,
        rms_rsdl=rms_rsdl,
        linf_rsdl=linf_rsdl,
        linf_alg_err_bd=linf_alg_err_bd,
        int_rms_err=int_rms_err,
        int_linf_err=int_linf_err,
        bd_rms_err=bd_rms_err,
        bd_linf_err=bd_linf_err,
    )


def u_exact(coords):
    x, y = coords
    return np.exp(x + y)


def u_exact_grad(coords):
    dxg = u_exact(coords)
    return np.stack([dxg, dxg])


def forcing(coords):
    return -2.0 * u_exact(coords)


num_divs_array = 4 << (np.arange(4) << 1)
domains = [
    BoxDomain(
        [
            Dimension(0.0, 1.0, num_divs, None, robin),
            Dimension(0.0, 1.0, num_divs, None, None),
        ]
    )
    for robin, num_divs in itertools.product((0.0, 2.0), num_divs_array)
]

neumann_domains = [
    BoxDomain(
        [
            Dimension(0.0, 1.0, num_divs, 0.0, 0.0),
            Dimension(0.0, 1.0, num_divs, 0.0, 0.0),
        ]
    )
    for num_divs in num_divs_array
]


results = [
    analyze_manufactured_solution(
        exact_soln=u_exact,
        exact_grad=u_exact_grad,
        forcing=forcing,
        domain=domain,
        ghost_term_bd=ghost_term_bd,
        gauge=get_gauge(domain),
    )
    for ghost_term_bd, get_gauge, domain in itertools.chain(
        itertools.product((False, True), (lambda d: None,), domains),
        itertools.product(
            (False, True),
            (
                lambda d: None,
                lambda d: ("value", -1, u_exact((1.0, 1.0))),
                lambda d: ("value", get_df(d) >> 1, u_exact((0.5, 0.5))),
                lambda d: ("integral", get_df(d) >> 1, np.square(np.expm1(1.0))),
                lambda d: ("integral", -1, np.square(np.expm1(1.0))),
            ),
            neumann_domains,
        ),
    )
]


def format_gauge(gauge):
    if gauge is None:
        return "-"
    t, idx, value = gauge
    pos = "1, 1" if idx == -1 else "½, ½"
    if t == "integral":
        return f"rhs({pos}) ← ∫u"
    if t == "value":
        return f"rhs({pos}) ← u({pos})"


num_fmt = "{:.3g}"
print(
    tabulate(
        [
            [
                f"{"ghost" if res["ghost_term_bd"] else "linear"}"
                f" {"Neumann" if (dim := res["domain"].dimensions[0]).high_robin == 0.0 else "Robin"}",
                format_gauge(res["gauge"]),
                dim.num_divs,
                num_fmt.format(res["rms_rsdl"]),
                num_fmt.format(res["linf_rsdl"]),
                num_fmt.format(res["linf_alg_err_bd"]),
                num_fmt.format(res["int_rms_err"]),
                num_fmt.format(res["int_linf_err"]),
                num_fmt.format(res["bd_rms_err"]),
                num_fmt.format(res["bd_linf_err"]),
            ]
            for res in results
        ],
        headers=[
            "bd type",
            "gauge",
            "divs",
            "RMS rsdl",
            "Linf rsdl",
            "Linf alg err bd",
            "int RMS err",
            "int Linf err",
            "bd RMS err",
            "bd Linf err",
        ],
        tablefmt="latex_booktabs",
    )
)


fig = plt.figure(figsize=plt.figaspect(9 / 16))
for i, res_i in enumerate([0, 4, 16, 20, 24, 28, 32]):
    result = results[res_i]
    ax: plt.Axes = fig.add_subplot(2, 4, i + 1)
    title = (
        f"pure Neumann, {format_gauge(result["gauge"])}"
        if result["gauge"] is not None
        else (
            "pure Neumann, gaugeless"
            if (dim := result["domain"].dimensions[0]).low_robin == 0.0
            else (
                "Neumann"
                if dim.high_robin == 0.0
                else f"Robin, coeff = {dim.high_robin}"
            )
        )
    )
    ax.set_title(title)

    ax_im = ax.matshow(
        result["poisson_lhs"].toarray(),
        norm=mpl.colors.CenteredNorm(halfrange=5),
        cmap="berlin",
    )

cax: plt.Axes = fig.add_axes([0.73, 0.14, 0.01, 0.33])

fig.colorbar(ax_im, cax=cax)
fig.suptitle("FDM left-hand side matrices")
plt.savefig("figures/fdm_lhs_mats.svg")


result = results[0]
domain = result["domain"]
xlim = domain.dimensions[0][:2]
ylim = domain.dimensions[1][:2]
X, Y = np.meshgrid(np.linspace(*xlim, 257), np.linspace(*ylim, 257))
uXY = u_exact((X, Y))

fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="3d"))
fig.set_layout_engine("compressed")
fig.set_constrained_layout_pads(w_pad=0.25, h_pad=0.25)

for ax, res_offset in zip(axs.ravel(), range(0, 16, 4)):
    ax.plot_wireframe(
        X,
        Y,
        uXY,
        rstride=X.shape[0] - 1,
        cstride=X.shape[1] - 1,
        linewidths=0.5,
        label="exact",
    )

    for i, result in enumerate(results[res_offset : res_offset + 2]):
        df = result["poisson_lhs"].shape[1]
        num_divs = result["domain"].dimensions[0].num_divs
        linewidth = 4.0 / np.log2(num_divs)
        ax.plot_wireframe(
            *result["coords"],
            result["approx"],
            rstride=num_divs - 2,
            cstride=num_divs - 2,
            colors=f"C{i}",
            linewidths=linewidth,
            label=f"df = {df}",
        )
        ax.plot_wireframe(
            *result["coords"],
            result["approx"],
            rstride=1,
            cstride=1,
            colors=f"C{i}",
            alpha=0.5,
            linewidths=0.25 * linewidth,
        )

    title = (
        f"$\\tilde u$"
        f" with {"ghost" if result['ghost_term_bd'] else "linear"}"
        f" {
            "Neumann" if (robin := result["domain"].dimensions[0].high_robin) == 0.0
            else f"Robin (coeff = {robin})"
        } $\\partial\\Omega$-term"
        f"{'' if result["gauge"] is None else f" and gauge term {format_gauge(result["gauge"])}"} "
    )

    ax.set(
        title=title,
        xlim=xlim,
        ylim=ylim,
        xlabel="$x$",
        ylabel="$y$",
        zlabel="$u$",
    )
    ax.legend()

plt.savefig("figures/approx_plots.svg")


for res_offset in range(0, len(results), 4):
    result = results[res_offset]
    domain = result["domain"]
    xlim = domain.dimensions[0][:2]
    ylim = domain.dimensions[1][:2]

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    for i, result in enumerate(results[res_offset : res_offset + 4]):
        df = result["poisson_lhs"].shape[1]
        num_divs = result["domain"].dimensions[0].num_divs
        linewidth = 4.0 / np.log2(num_divs)
        abserr = np.abs(result["error"])
        ax.plot_wireframe(
            *result["coords"],
            abserr,
            rstride=num_divs,
            cstride=num_divs,
            colors=f"C{i}",
            linewidths=linewidth,
            label=f"df = {df}",
            axlim_clip=True,
        )
        ax.plot_wireframe(
            *result["coords"],
            abserr,
            rstride=1,
            cstride=1,
            colors=f"C{i}",
            linewidths=0.25 * linewidth,
            alpha=0.25 * linewidth,
            axlim_clip=True,
        )

    title = (
        "$\\vert E \\vert$ surface"
        f" with {"ghost" if result['ghost_term_bd'] else "linear"}"
        f" {
            "Neumann" if (robin := result["domain"].dimensions[0].high_robin) == 0.0
            else f"Robin (coeff = {robin})"
        } $\\partial\\Omega$-term"
        f"{'' if result["gauge"] is None else f" and gauge term {format_gauge(result["gauge"])}"} "
    )
    ax.set(
        title=title,
        xlim=xlim,
        ylim=ylim,
        xlabel="$x$",
        ylabel="$y$",
        zlabel="|E|",
        zscale="log",
    )
    min_top = np.max(
        [
            results[res_offset]["int_linf_err"],
            results[res_offset]["bd_linf_err"],
        ]
    )
    range_factor = result["int_rms_err"] / min_top
    max_bottom = result["int_rms_err"] * range_factor

    ax.set_zlim(
        bottom=np.max(
            [
                ax.get_zlim()[0],
                max_bottom,
            ]
        ),
        top=np.min(
            [
                ax.get_zlim()[1],
                min_top,
            ]
        ),
    )
    ax.legend()

    plt.savefig(f"figures/err_surf_{res_offset//4:2d}.svg")


for i, (title, ylabel, spec) in enumerate(
    [
        (
            "RMS error vs degrees of freedom",
            "$\\Vert E \\Vert_{\\mathrm{rms}}$",
            [
                ("int_rms_err", dict(linestyle=":", marker="o", markersize=4.5)),
                ("bd_rms_err", dict(linestyle="-", marker="o", mfc="none")),
            ],
        ),
        (
            "$L_{\\infty}$ error vs degrees of freedom",
            "$\\Vert E \\Vert_{\\infty}$",
            [
                ("int_linf_err", dict(linestyle=":", marker="s", markersize=4.5)),
                ("bd_linf_err", dict(linestyle="-", marker="s", mfc="none")),
            ],
        ),
    ]
):
    fig, ax = plt.subplots()
    ax: plt.Axes = ax

    for res_offset in range(0, 16, 4):
        ghost = results[res_offset]["ghost_term_bd"]
        robin = results[res_offset]["domain"].dimensions[0].high_robin
        dfs = [
            result["poisson_lhs"].shape[1]
            for result in results[res_offset : res_offset + 4]
        ]

        for errt, kwargs in spec:
            errs = [result[errt] for result in results[res_offset : res_offset + 4]]

            label = (
                f"{"ghost" if ghost else "linear"}"
                f" {"Neumann" if robin == 0.0 else "Robin"}"
                f" on {"$\\Omega$" if errt[:3] == "int" else "$\\partial\\Omega$"}"
            )

            ax.plot(
                dfs,
                errs,
                color=f"C{res_offset // 4}",
                label=label,
                **kwargs,
            )

    ax.set(
        title=title,
        xlabel="df",
        xscale="log",
        yscale="log",
    )
    ax.set_ylabel(ylabel, rotation=0, va="top", y=1)

    ax.legend()
    plt.savefig(f"figures/err_vs_df_{i}.svg")


fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot()

for res_offset in range(16, len(results), 4):
    gauge = results[res_offset]["gauge"]
    ghost = results[res_offset]["ghost_term_bd"]
    if gauge is None or not ghost:
        continue
    robin = results[res_offset]["domain"].dimensions[0].high_robin
    dfs = [
        result["poisson_lhs"].shape[1]
        for result in results[res_offset : res_offset + 4]
    ]

    for errt, kwargs in [
        ("int_rms_err", dict(linestyle=":", marker="o", markersize=4.5)),
        # ("int_linf_err", dict(linestyle=":", marker="s", markersize=4.5)),
        ("bd_rms_err", dict(linestyle="-", marker="o", mfc="none")),
        # ("bd_linf_err", dict(linestyle="-", marker="s", mfc="none")),
    ]:
        errs = [result[errt] for result in results[res_offset : res_offset + 4]]

        label = (
            # f"{"ghost" if ghost else "linear"}"
            f"on {"$\\Omega$" if errt[:3] == "int" else "$\\partial\\Omega$"},"
            f" {format_gauge(gauge)}"
        )
        ax.plot(
            dfs,
            errs,
            color=f"C{res_offset // 4}",
            label=label,
            **kwargs,
        )

ax.set(
    title=f"Error vs degrees of freedom (pure Neumann)",
    xlabel="df",
    xscale="log",
    yscale="log",
)
ax.set_ylabel("$\\Vert E\\Vert$", rotation=0, va="top", y=1)

ax.legend()
plt.savefig("figures/err_vs_df_neumann.svg")
