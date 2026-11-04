"""Read-only reference extracted verbatim from hua-nb notebook cells 5--8.

Source: https://github.com/hjskyline/mth321_project2_team17/blob/b3466958df84f0faa7d6f9ed79588a77558f05aa/luhua.ipynb
These definitions are copied WITHOUT changing their bodies. Only the safe imports
below are supplied. No notebook experiment or plotting cell is executed.
Use ghost_term_bd=True for the prescribed ghost problem; False is not one-sided.
The original notebook and its branch are untouched.
"""
import numpy as np
from scipy import sparse
from functools import reduce

from collections import namedtuple

Dimension = namedtuple(
    "Dimension", ["low", "high", "num_divs", "low_robin", "high_robin"]
)
BoxDomain = namedtuple("BoxDomain", ["dimensions"])

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


def problem_scale(dom: BoxDomain):
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
        indexing="ij"
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
