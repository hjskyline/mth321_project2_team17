"""有独立数学依据的检查；不使用“拟合接近预设阶数”作为唯一通过条件。"""
from dataclasses import replace
import unittest
import numpy as np
from scipy.sparse.linalg import spsolve

from mixed_boundary_fd import MMS, ManufacturedData, assemble_mixed_poisson, error_metrics
from linear_solvers import solve_linear, difference_norms
from kron_reference import assemble_kron


class ExtendedVerification(unittest.TestCase):
    def test_observed_order_accepts_a_mesh_list(self):
        from mixed_boundary_fd import observed_order
        rates=observed_order([.25,.125,.0625],[.0625,.015625,.00390625])
        np.testing.assert_allclose(rates[1:],[2.,2.])

    def test_independent_kron_assembly_and_nonzero_boundary_rhs(self):
        for N in (2,5,8):
            for bc in ('neumann','robin'):
                for closure in ('first_order','ghost'):
                    s=assemble_mixed_poisson(N,bc,closure)
                    A,b=assemble_kron(N,bc,closure,MMS)
                    self.assertEqual(A.shape,(N*(N-1),)*2)
                    d=A-s.A
                    self.assertLess(np.max(np.abs(d.data)) if d.nnz else 0,1e-12)
                    np.testing.assert_allclose(b,s.b,rtol=1e-13,atol=1e-10)

    def test_two_norms_do_not_include_prescribed_dirichlet_values(self):
        s=assemble_mixed_poisson(4)
        d=np.zeros(s.A.shape[0]);d[-3:]=2
        n=difference_norms(s,d)
        self.assertAlmostEqual(n['L2'],s.h*np.sqrt(6))
        self.assertEqual(n['Linf'],2)

    def test_gmres_and_amg_on_both_nonsymmetric_formulations(self):
        for bc in ('neumann','robin'):
            for closure in ('first_order','ghost'):
                s=assemble_mixed_poisson(8,bc,closure)
                exact_discrete=spsolve(s.A,s.b)
                for method in ('lu','gmres','amg_gmres'):
                    x,meta,history=solve_linear(s,method,rtol=1e-12)
                    self.assertTrue(meta['converged'])
                    self.assertLess(np.max(np.abs(x-exact_discrete)),1e-8)
                    self.assertEqual(meta['iterations'],len(history))

    def test_iteration_cap_is_exposed_not_reported_as_success(self):
        s=assemble_mixed_poisson(16)
        _,meta,_=solve_linear(s,'gmres',rtol=1e-14,restart=1,max_cycles=1)
        self.assertFalse(meta['converged'])
        self.assertNotEqual(meta['info'],0)

    def test_bad_boundary_sign_is_detected_despite_small_algebraic_residual(self):
        data=ManufacturedData(lambda x,y:1+x+2*y,lambda x,y:0.,lambda y:1.,'affine')
        good=assemble_mixed_poisson(8,'neumann','ghost',data=data)
        bad=assemble_mixed_poisson(8,'neumann','ghost',data=replace(data,q=lambda y:-1.))
        self.assertLess(error_metrics(good,spsolve(good.A,good.b))['Linf'],1e-12)
        u=spsolve(bad.A,bad.b)
        self.assertLess(np.max(np.abs(bad.b-bad.A@u)),1e-10)
        self.assertGreater(error_metrics(good,u)['Linf'],.01)

    def test_flag_false_does_not_replace_ghost_row(self):
        # A regression test for the interpretation of the read-only notebook code.
        # Turning off f on the boundary changes b only; first_order changes A too.
        s=assemble_mixed_poisson(8,'neumann','ghost')
        first=assemble_mixed_poisson(8,'neumann','first_order')
        changed=s.b.copy();y=np.arange(1,8)/8
        changed[-7:]-=MMS.forcing(1.,y)
        self.assertGreater(np.max(np.abs((s.A-first.A).data)),1.)
        self.assertGreater(np.max(np.abs(changed-first.b)),1.)


if __name__=='__main__':
    unittest.main()
