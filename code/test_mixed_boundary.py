"""Independent small-case and regression checks, without fitted-order assumptions."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from mixed_boundary_fd import (
    ManufacturedData, assemble_mixed_poisson, error_metrics,
    residual_metrics, verify_implementation,
)


class MixedBoundaryTests(unittest.TestCase):
    def test_hand_boundary_rows_and_affine_exactness(self):
        self.assertEqual(len(verify_implementation()), 4)

    def test_ghost_quadratic_exactness(self):
        data = ManufacturedData(lambda x, y: x*x + y*y,
                                lambda x, y: -4.0,
                                lambda y: 2.0, "x^2+y^2")
        for bc in ("neumann", "robin"):
            for N in (2, 5, 9):
                with self.subTest(bc=bc, N=N):
                    s = assemble_mixed_poisson(N, bc, "ghost", data=data)
                    x, y = s.coordinates()
                    np.testing.assert_allclose(s.A @ data.exact(x, y).ravel(),
                                               s.b, atol=2e-12, rtol=0)
                    self.assertLess(error_metrics(s, spsolve(s.A, s.b))["Linf"], 1e-12)

    def test_sparse_solve_against_dense_small_system(self):
        for bc in ("neumann", "robin"):
            for closure in ("first_order", "ghost"):
                with self.subTest(bc=bc, closure=closure):
                    s = assemble_mixed_poisson(4, bc, closure)
                    np.testing.assert_allclose(spsolve(s.A, s.b),
                                               np.linalg.solve(s.A.toarray(), s.b),
                                               atol=1e-12, rtol=1e-12)

    def test_residual_invariance_to_row_scaling(self):
        for closure in ("first_order", "ghost"):
            s = assemble_mixed_poisson(4, "robin", closure)
            x, y = s.coordinates()
            trial = s.data.exact(x, y).ravel() + 0.03
            weights = np.geomspace(1e-5, 1e5, s.A.shape[0])
            scaled = replace(s, A=diags(weights) @ s.A, b=weights*s.b)
            original, changed = residual_metrics(s, trial), residual_metrics(scaled, trial)
            for key in ("row_scaled_residual_inf", "componentwise_backward_error"):
                np.testing.assert_allclose(original[key], changed[key], rtol=1e-10, atol=1e-13)

    def test_error_norm_weights(self):
        s = assemble_mixed_poisson(4)
        x, y = s.coordinates()
        sampled = s.data.exact(x, y).ravel()
        # Unit error on three right-edge unknowns only: each has half an area weight.
        sampled[-3:] += 1.0
        metrics = error_metrics(s, sampled)
        self.assertAlmostEqual(metrics["L2"], np.sqrt(3*s.h**2/2))
        self.assertAlmostEqual(metrics["right_L2"], np.sqrt(3*s.h))
        self.assertEqual(metrics["interior_Linf"], 0.0)

    def test_invalid_parameters(self):
        for N in (True, 0, 1, 3.5):
            with self.assertRaises(ValueError):
                assemble_mixed_poisson(N)
        with self.assertRaises(ValueError):
            assemble_mixed_poisson(4, bc_type="dirichlet")
        with self.assertRaises(ValueError):
            assemble_mixed_poisson(4, closure="central")
        with self.assertRaises(ValueError):
            assemble_mixed_poisson(4, beta=-1)


if __name__ == "__main__":
    unittest.main()
