"""
Container for external functions required by the HeFESTo EOS arithmetic engine,
replicating the functionality of Fortran external procedures and BLAS/LAPACK calls.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg.lapack import dgetrf, dgetri
from scipy.linalg.blas import dgemv, ddot, dgemm

from . import bserch as bserch_module
from . import thetacal as thetacal_module
from . import qr19 as qr19_module
from . import landau as landau_module
from . import landauqr as landauqr_module
from .therm_common import apar_value


class HeFESToExtern:
    """Namespace for external functions called by physub and related code."""

    def __init__(self):
        self.state = None

    def set_state(self, state):
        self.state = state
        return state

    def _resolve_state(self, state=None):
        if state is not None:
            return state
        if self.state is None:
            raise ValueError('HeFESToExtern requires a bound state')
        return self.state

    def depth(self, pressure_gpa):
        """Convert pressure to an approximate depth in km for reporting/Q lookup."""
        return 30.0 * float(pressure_gpa)

    def dgemv(self, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        """Wrapper for BLAS DGEMV (matrix-vector multiplication)."""
        A_op = A.T if trans.lower().startswith('t') else A
        x_vec = np.asarray(x)[:n]
        y_vec = np.asarray(y)
        result = alpha * (A_op[:m, :n] @ x_vec)
        if beta != 0.0:
            result = result + beta * y_vec[: result.shape[0]]
        y_vec[: result.shape[0]] = result
        return y_vec

    def ddot(self, n, x, incx, y, incy):
        """Wrapper for BLAS DDOT (dot product)."""
        return float(np.dot(np.asarray(x)[:n], np.asarray(y)[:n]))

    def dlacpy(self, uplo, m, n, A, lda, B, ldb):
        """Wrapper for LAPACK DLACPY (matrix copy)."""
        if uplo.lower().startswith('a'):
            B[:m, :n] = A[:m, :n]
        else:
            raise NotImplementedError("Only full matrix copy ('A') is implemented for dlacpy.")
        return B

    def dgetrf(self, m, n, a, lda, ipiv, info):
        """Wrapper for LAPACK DGETRF (LU factorization)."""
        lu, piv, info_val = dgetrf(a, overwrite_a=False)
        info[0] = info_val
        return lu, piv, info_val

    def dgetri(self, n, a, lda, ipiv, work, lwork, info):
        """Wrapper for LAPACK DGETRI (matrix inverse from LU)."""
        inv_a = np.linalg.pinv(a[:n, :n])
        a[:n, :n] = inv_a
        info[0] = 0
        return a, 0

    def dgemm(self, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
        """Wrapper for BLAS DGEMM (matrix-matrix multiplication)."""
        A_op = A.T if transa.lower().startswith('t') else A
        B_op = B.T if transb.lower().startswith('t') else B
        result = alpha * (A_op[:m, :k] @ B_op[:k, :n])
        if beta != 0.0:
            result = result + beta * C[:m, :n]
        C[:m, :n] = result
        return C

    def nform(self, nnew, n, n1, q2, nspec, nnull):
        """Python translation of nform.f."""
        n[:nspec] = n1[:nspec] + q2[:nspec, :nnull] @ nnew[:nnull]

    def hessfunc(self, nnew, state_or_out):
        """Python translation of hessfunc.f (projected Hessian)."""
        from . import hessfunc as hessfunc_module

        if hasattr(state_or_out, 'nspec'):
            return hessfunc_module.hessfunc(nnew, state_or_out, self)

        state = self._resolve_state()
        hespro = hessfunc_module.hessfunc(nnew, state, self)
        if isinstance(state_or_out, np.ndarray):
            state_or_out[:hespro.shape[0], :hespro.shape[1]] = hespro
            return state_or_out
        return hespro

    def svdsub(self, m, n, A, lda, ldb, b, q1, q2, x, nnulls):
        """Python translation of svdsub.f (via NumPy SVD)."""
        U, s, Vh = np.linalg.svd(A[:m, :n], full_matrices=True)
        small = 1e-6
        s_inv = np.zeros_like(s)
        mask = s >= small
        s_inv[mask] = 1.0 / s[mask]

        if isinstance(nnulls, list):
            nnulls[0] = int(np.sum(~mask))
        else:
            nnulls = int(np.sum(~mask))

        S_inv_mat = np.zeros((n, m))
        np.fill_diagonal(S_inv_mat, s_inv)
        x[:n] = Vh.T @ S_inv_mat @ U.T @ b[:m]

    def parset(self, ispec, apar=None, *args):
        """Return the legacy parset tuple using parameter rows from `apar`."""
        state = self._resolve_state() if apar is None else None
        if apar is None:
            apar = state.apar

        fn = apar_value(apar, ispec, 1, 1.0)
        zu = apar_value(apar, ispec, 2, 1.0)
        wm = apar_value(apar, ispec, 3, 1.0)
        To = apar_value(apar, ispec, 4, 298.15)
        Fo = apar_value(apar, ispec, 5, 0.0)
        Vo = apar_value(apar, ispec, 6, max(1.0, wm))
        Ko = apar_value(apar, ispec, 7, 160.0)
        Kop = apar_value(apar, ispec, 8, 4.0)
        Kopp = apar_value(apar, ispec, 9, 0.0)
        wd1 = apar_value(apar, ispec, 10, 800.0)
        wd2 = apar_value(apar, ispec, 11, 0.0)
        wd3 = apar_value(apar, ispec, 12, 0.0)
        ws1 = apar_value(apar, ispec, 13, 0.0)
        ws2 = apar_value(apar, ispec, 14, 0.0)
        ws3 = apar_value(apar, ispec, 15, 0.0)
        we1 = apar_value(apar, ispec, 16, 0.0)
        qe1 = apar_value(apar, ispec, 17, 0.0)
        we2 = apar_value(apar, ispec, 18, 0.0)
        qe2 = apar_value(apar, ispec, 19, 0.0)
        we3 = apar_value(apar, ispec, 20, 0.0)
        qe3 = apar_value(apar, ispec, 21, 0.0)
        we4 = apar_value(apar, ispec, 22, 0.0)
        qe4 = apar_value(apar, ispec, 23, 0.0)
        wou = apar_value(apar, ispec, 24, 0.0)
        wol = apar_value(apar, ispec, 25, 0.0)
        gam = apar_value(apar, ispec, 26, 1.2)
        qo = apar_value(apar, ispec, 27, 1.0)
        be = apar_value(apar, ispec, 28, 0.0)
        ge = apar_value(apar, ispec, 29, 0.0)
        q2A2 = apar_value(apar, ispec, 30, 0.0)
        htl = int(round(apar_value(apar, ispec, 31, 0.0)))
        ibv = int(round(apar_value(apar, ispec, 32, 0.0)))
        ied = int(round(apar_value(apar, ispec, 33, 0.0)))
        izp = int(round(apar_value(apar, ispec, 34, 0.0)))
        Go = apar_value(apar, ispec, 35, 0.0)
        Gop = apar_value(apar, ispec, 36, 0.0)
        Got = apar_value(apar, ispec, 37, 0.0)
        return (
            fn, zu, wm, To, Fo, Vo, Ko, Kop, Kopp,
            wd1, wd2, wd3, ws1, ws2, ws3,
            we1, qe1, we2, qe2, we3, qe3, we4, qe4, wou, wol,
            gam, qo, be, ge, q2A2, htl, ibv, ied, izp, Go, Gop, Got,
        )

    def cp(self, ispec, ncp, state=None):
        """Python translation of cp.f (chemical potential and properties)."""
        from . import cp as cp_module
        state = self._resolve_state(state)
        chempot, rsum, volsum, smixi = cp_module.cp(ispec, ncp, state)
        return chempot, rsum, volsum, smixi, 0.0

    def gspec(self, ispec, state=None):
        """Python translation of gspec.f (single-species thermodynamics)."""
        from . import gspec as gspec_module
        state = self._resolve_state(state)
        result, spinodal = gspec_module.gspec(ispec, state)
        state.vol = result.vol
        state.Cap = result.Cp
        state.Cv = result.Cv
        state.gamma = result.gamma
        state.K = result.K
        state.Ks = result.Ks
        state.alp = result.alp
        state.Ftot = result.Ftot
        state.ph = result.ph
        state.ent = result.ent
        state.deltas = result.deltas
        state.tcal = result.tcal
        state.zeta = result.zeta
        state.Gsh = result.Gsh
        state.uth = result.uth
        state.uto = result.uto
        state.thet = result.thet
        state.qq = result.qq
        state.etas = result.etas
        state.dGdT = result.dGdT
        state.pzp = result.pzp
        state.Vdeb = result.Vdeb
        state.gamdeb = result.gamdeb
        if hasattr(state, 'spinod') and ispec < len(state.spinod):
            state.spinod[ispec] = bool(spinodal)
        return result.Ftot

    def hessian(self, ispec, ncp, state=None):
        """Python translation of hessian.f (Hessian matrix)."""
        from . import hessian as hessian_module
        return hessian_module.hessian(ispec, ncp, self._resolve_state(state))

    def thetacal(self, x, result=None):
        """Python translation of thetacal.f using Heat table inversion."""
        tcal = thetacal_module.thetacal(x)
        if isinstance(result, list):
            result[0] = tcal
        return tcal

    def tlindeman(self, vol, wmagg, fnagg, thet):
        """Python translation of tlindeman.f."""
        flin = 0.1533
        Angstrom = 1e10
        avn = 6.02214076e23
        hplanck = 6.62607015e-34
        boltzk = 1.380649e-23
        pirad = np.pi

        aspace = (vol / fnagg / avn) ** (1.0 / 3.0) * 0.01 * Angstrom
        amass = wmagg / fnagg
        amu = 1e-3 / avn
        hbar = hplanck / (2 * pirad)
        fac = (hbar**2) / boltzk / amu * (Angstrom**2)

        return amass / fac * (thet**2) * (aspace**2) / 9.0 * (flin**2)

    def qr19(self, depth, Ti):
        """Python translation of qr19.f (attenuation Q model)."""
        return qr19_module.qr19(depth, Ti)

    def vred(self, qs, qp, vsred=None, vpred=None):
        """Python translation of vred.f."""
        alpha = 0.26
        vsred_val = 1.0 - 0.5 / np.tan(alpha * np.pi / 2.0) / qs
        vpred_val = 1.0 - 0.5 / np.tan(alpha * np.pi / 2.0) / qp

        if isinstance(vsred, list):
            vsred[0] = vsred_val
        if isinstance(vpred, list):
            vpred[0] = vpred_val

        return vsred_val, vpred_val

    def bserch(self, xx, x):
        """Binary search helper (wrapper to bserch module)."""
        return bserch_module.bserch(xx, x)

    def landau(self, ispec, Vi, state):
        """Landau phase transition properties."""
        return landau_module.landau(ispec, Vi, state)

    def landauqr(self, ispec, Vi, state, lphase=None):
        """Landau phase transition (Q-referenced)."""
        return landauqr_module.landauqr(ispec, Vi, state, lphase)
