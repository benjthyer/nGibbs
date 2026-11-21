"""
Cpx exsolution ("method A"): split a pooled clinopyroxene (total moles + mole-weighted
composition) into 0/1/2 coexisting cpx by minimizing the MELTS cpx Gibbs
energy -- fully vectorized torch (CPU or GPU), fixed iteration counts,
no host syncs inside the solve loops.

The MELTS cpx model (clinopyroxene.c, translated in melts_vec/clinopyroxene.py)
is G(z) = cubic polynomial in z=(r0..r5, s0, s1) - T*S_conf(y(z)), with y the
18 site-fraction / composite quantities, all LINEAR in z. So everything is
written as dense tensors:
    poly(z)  = a.z + z^T B z + C[z,z,z]     (H, S, V each)
    S_conf   = -R sum_k w_k y_k ln y_k,     y = M z + m0
with analytic gradient and Hessian. Pure-endmember reference terms are linear
in composition and cancel in a fixed-bulk split, so they are omitted.

Algorithm per row (all rows in lockstep):
  1. relax ordering s at the bulk composition (Newton on s, 2x2)
  2. tangent-plane-distance stability test from Ca-poor and Ca-rich seeds
     (8-var saddle-free Newton + batched backtracking)
  3. two-phase Newton in extensive variables (q0, q_r, s1, s2) [11 vars],
     seeded from the most negative TPD minimum; rows that are stable keep
     the single-phase answer.
"""
from __future__ import annotations
import itertools
import numpy as np
import torch

from .melts_vec import clinopyroxene as _cpx

R = _cpx.Rgas
W_CONF = np.array([1, 1, 1, 1, 1, 1, -1, -1, -2, 2, 2, 1, 1, 1, 1, 1, -1, -1], float)


def _raw_y(z):
    """18 linear site/composite quantities (unclipped), order matches W_CONF."""
    r0, r1, r2, r3, r4, r5, s0, s1 = z
    xti4m1 = (r1 + r2) / 2
    xca2m2 = 1 - r4 - r5
    xna1m2 = r4
    xsi4tet = (4 - 2 * r1 - 2 * r2 - 2 * r3 + r4) / 4
    xal3m1 = (2 * r3 + r4 - 2 * s0) / 4
    xfe2m1 = (2 * r0 - r5 - s1) / 2
    xfe3m1 = (2 * r3 + r4 + 2 * s0) / 4
    xmg2m1 = 1 - r0 - r3 - 0.5 * (r1 + r2 + r4 - r5 - s1)
    xfe2m2 = (r5 + s1) / 2
    xmg2m2 = (r5 - s1) / 2
    xal3tet = (4 * r1 + 2 * r3 - r4 + 2 * s0) / 8
    xfe3tet = (4 * r2 + 2 * r3 - r4 - 2 * s0) / 8
    a = xmg2m1 + xfe2m1 - xti4m1
    b = 1 - xti4m1
    c = xmg2m1 + xfe2m1
    d = 1 - xsi4tet
    e = 1 - xmg2m1 - xfe2m1 - xna1m2
    c2 = 1 - xmg2m1 - xfe2m1
    f = 1 - xna1m2
    return np.array([xmg2m1, xfe2m1, xal3m1, xfe3m1, xti4m1, a, b, c, d,
                     xal3tet, xfe3tet, xca2m2, xna1m2, xmg2m2, xfe2m2, e, c2, f])


def _poly_tensors(parsed):
    a = np.zeros(8); B = np.zeros((8, 8)); C = np.zeros((8, 8, 8))
    for val, idx in parsed:
        if len(idx) == 1:
            a[idx[0]] += val
        elif len(idx) == 2:
            i, j = idx; B[i, j] += val / 2; B[j, i] += val / 2
        elif len(idx) == 3:
            perms = set(itertools.permutations(idx))
            for p in perms:
                C[p] += val / len(perms)
        else:
            raise ValueError(idx)
    return a, B, C


# x (7 endmember fractions) = E r + e0 ; r = Xr x
_E = np.array([[-1, -1, -1, -1, -0.5, -1],
               [0, 0, 0, 0, 0, 1],
               [1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, -0.5, 0],
               [0, 0, 1, 0, 0.5, 0],
               [0, 0, 0, 1, -0.5, 0],
               [0, 0, 0, 0, 1, 0]], float)
_E0 = np.array([1, 0, 0, 0, 0, 0, 0], float)
_XR = np.array([[0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0.5],
                [0, 0, 0, 0, 1, 0, -0.5],
                [0, 0, 0, 0, 0, 1, 0.5],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0]], float)


class CpxTorch:
    def __init__(self, device='cpu', dtype=torch.float64, solver='hybrid'):
        self.device, self.dtype, self.solver = torch.device(device), dtype, solver
        t = lambda x: torch.as_tensor(np.asarray(x), device=self.device, dtype=dtype)
        polys = [_poly_tensors(p) for p in (_cpx._H_PARSED, _cpx._S_PARSED, _cpx._V_PARSED)]
        A = np.stack([p[0] for p in polys])          # (3,8)
        B = np.stack([p[1] for p in polys])          # (3,8,8)
        C = np.stack([p[2] for p in polys])          # (3,8,8,8)
        self.A = t(A)
        self.Bflat = t(B.reshape(3, 64))
        # cubic Hessian 6*C(.,.,z) weighted by (1,-T,P-1): ((wts (x) z) @ Cflat).view(8,8)
        self.Cflat = t(6.0 * np.transpose(C, (0, 3, 1, 2)).reshape(24, 64))
        m0 = _raw_y(np.zeros(8))
        M = np.stack([_raw_y(np.eye(8)[k]) - m0 for k in range(8)], axis=1)  # (18,8)
        self.M, self.m0, self.w = t(M), t(m0), t(W_CONF)
        self.Mouter = t(np.einsum('ki,kj->kij', M, M).reshape(18, 64))
        self.E, self.E0, self.XR = t(_E), t(_E0), t(_XR)
        self.tiny = 1e-30
        self.taus = t([0.0, 1e-6, 1e-3, 1e-1, 1.0, 12.0])

    # ---------------------------------------------------------------- core
    def _wts(self, T, P):
        return torch.stack([torch.ones_like(T), -T, P - 1.0], -1)  # (N,3)

    def y(self, z):
        return z @ self.M.T + self.m0

    def _poly(self, z, T, P):
        """value (N,), grad (N,8), Hessian (N,8,8) of the T,P-weighted cubic polynomial."""
        N = z.shape[0]
        wts = self._wts(T, P)
        Aw = wts @ self.A                                                  # (N,8)
        Bw = (wts @ self.Bflat).view(N, 8, 8)
        Hc = ((wts[:, :, None] * z[:, None, :]).reshape(N, 24) @ self.Cflat).view(N, 8, 8)
        Hp = 2.0 * Bw + Hc
        Bz = (Bw @ z[:, :, None]).squeeze(-1)
        Hcz = (Hc @ z[:, :, None]).squeeze(-1)
        gp = Aw + 2.0 * Bz + 0.5 * Hcz
        vp = (z * (Aw + Bz + Hcz / 6.0)).sum(-1)
        return vp, gp, Hp

    def value(self, z, T, P, ymask=None):
        """g (N,), min_y over active y (N,). z (N,8)."""
        N = z.shape[0]
        wts = self._wts(T, P)
        Bw = (wts @ self.Bflat).view(N, 8, 8)
        Hc = ((wts[:, :, None] * z[:, None, :]).reshape(N, 24) @ self.Cflat).view(N, 8, 8)
        Q = Bw + Hc / 6.0
        g = (z * ((wts @ self.A) + (Q @ z[:, :, None]).squeeze(-1))).sum(-1)
        y = self.y(z)
        yc = y.clamp_min(self.tiny)
        g = g + R * T * (self.w * yc * torch.log(yc)).sum(-1)
        if ymask is not None:
            y = torch.where(ymask, y, torch.ones_like(y))
        return g, y.min(-1).values

    def derivs(self, z, T, P):
        """g (N,), grad (N,8), hess (N,8,8), min_y (N,)."""
        N = z.shape[0]
        g, gr, H = self._poly(z, T, P)
        y = self.y(z)
        yc = y.clamp_min(self.tiny)
        RT = R * T
        lny = torch.log(yc)
        act = y > self.tiny          # inactive (absent-species) y contribute a constant 0
        g = g + RT * (self.w * yc * lny).sum(-1)
        gr = gr + RT[:, None] * (torch.where(act, self.w * (lny + 1.0), 0.0) @ self.M)
        H = H + RT[:, None, None] * (torch.where(act, self.w / yc, 0.0) @ self.Mouter).view(N, 8, 8)
        return g, gr, H, y.min(-1).values

    # ------------------------------------------------------------- helpers
    def x_to_r(self, X):
        return X @ self.XR.T

    def r_to_x(self, r):
        return r @ self.E.T + self.E0

    def s_guess_and_gate(self, r):
        totAl = r[:, 1] + r[:, 3]
        totCa = 1 - r[:, 4] - r[:, 5]
        totFe2 = r[:, 0]
        totFe3 = r[:, 2] + r[:, 3]
        totMg = 1 - r[:, 0] - 0.5 * r[:, 1] - 0.5 * r[:, 2] - r[:, 3] - 0.5 * r[:, 4] + r[:, 5]
        totNa = r[:, 4]
        totTi = 0.5 * (r[:, 1] + r[:, 2])
        totM2 = totCa + totNa
        totM1 = totTi + (totMg + totFe2 - (1 - totM2))
        d0 = totFe3 + totAl
        d1 = totFe2 + totMg
        s0 = torch.where(d0 != 0, (1 - totM1) * (totFe3 - totAl) / torch.where(d0 == 0, 1.0, d0), 0.0)
        s1 = torch.where(d1 != 0, (1 - totM2) * (totFe2 - totMg) / torch.where(d1 == 0, 1.0, d1), 0.0)
        gate = torch.stack([(totFe3 != 0) & (totAl != 0), (totFe2 != 0) & (totMg != 0)], -1)
        return torch.stack([s0, s1], -1), gate

    # -------------------------------------------------- generic Newton step
    def _sf_direction(self, grad, H):
        """Newton direction for a possibly indefinite Hessian.
        'chol': Jacobi-scaled modified Cholesky -- factor Hs + tau*I for a ladder
        of shifts in one batched call, keep the smallest tau that factorizes.
        'eigh': saddle-free Newton -V |L|^-1 V^T grad.
        'hybrid': plain Cholesky Newton where H is PD (identical step to eigh there),
        eigh only on the non-PD rows (compacted)."""
        if self.solver == 'eigh':
            L, V = torch.linalg.eigh(H)
            Lm = L.abs()
            floor = 1e-10 * Lm.max(-1, keepdim=True).values + 1e-6
            Lm = torch.maximum(Lm, floor)
            return -(V @ ((V.transpose(-1, -2) @ grad[..., None]).squeeze(-1) / Lm)[..., None]).squeeze(-1)
        if self.solver == 'hybrid':
            L, info = torch.linalg.cholesky_ex(H)
            pd = info == 0
            step = -torch.cholesky_solve(grad[..., None], L).squeeze(-1)
            bad = torch.nonzero(~pd).squeeze(-1)
            if bad.numel():
                Lb, V = torch.linalg.eigh(H[bad])
                Lm = Lb.abs()
                Lm = torch.maximum(Lm, 1e-10 * Lm.max(-1, keepdim=True).values + 1e-6)
                step[bad] = -(V @ ((V.transpose(-1, -2) @ grad[bad][..., None]).squeeze(-1) / Lm)[..., None]).squeeze(-1)
            return torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
        n = H.shape[-1]
        d = H.diagonal(dim1=-2, dim2=-1).abs().clamp_min(1e-12).sqrt()
        Hs = H / (d[..., :, None] * d[..., None, :])
        gs = grad / d
        eye = torch.eye(n, device=H.device, dtype=H.dtype)
        K = self.taus.numel()
        Hk = Hs[None] + self.taus[:, None, None, None] * eye             # (K,N,n,n)
        Lk, info = torch.linalg.cholesky_ex(Hk)
        ok = info == 0                                                      # (K,N)
        first = torch.where(ok.any(0), ok.int().argmax(0), torch.full_like(info[0], K - 1))
        L = Lk[first, torch.arange(H.shape[0], device=H.device)]
        step = torch.cholesky_solve(gs[..., None], L).squeeze(-1)
        step = torch.where(ok.any(0)[:, None], step, gs)
        step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
        return -step / d

    ALPHAS = (1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 0.0)

    def _line_search(self, x, d, obj_fn):
        """Batched backtracking: evaluate all alphas at once, keep best feasible."""
        N, D = x.shape
        al = torch.tensor(self.ALPHAS, device=x.device, dtype=x.dtype)
        K = al.numel()
        xt = (x[:, None, :] + al[None, :, None] * d[:, None, :]).reshape(N * K, D)
        f, feas = obj_fn(xt, K)
        f = torch.where(feas, f, torch.full_like(f, float('inf'))).reshape(N, K)
        f = torch.nan_to_num(f, nan=float('inf'))
        k = f.argmin(-1)
        return xt.reshape(N, K, D)[torch.arange(N, device=x.device), k], f.gather(1, k[:, None]).squeeze(1)

    # ------------------------------------------------------- 1. s relax
    def relax_s(self, r, s, T, P, gate, iters=12, ymask=None):
        for _ in range(iters):
            z = torch.cat([r, s], -1)
            _, gr, H, _ = self.derivs(z, T, P)
            gs = torch.where(gate, gr[:, 6:], 0.0)
            Hss = H[:, 6:, 6:]
            d = self._sf_direction(gs, Hss)
            d = torch.where(gate, d, 0.0)

            def obj(st, K):
                zt = torch.cat([r.repeat_interleave(K, 0), st], -1)
                g, ymin = self.value(zt, T.repeat_interleave(K), P.repeat_interleave(K),
                                     None if ymask is None else ymask.repeat_interleave(K, 0))
                return g, ymin > 0
            s, _ = self._line_search(s, d, obj)
        return s

    # ------------------------------------------------------- 2. TPD
    def tpd_min(self, z, lam, T, P, gate, rfree, ymask, iters=25):
        """Minimize g(z) - lam.r over z=(r,s). Returns z, tpd-objective."""
        gate8 = torch.cat([rfree, gate], -1)
        for _ in range(iters):
            _, gr, H, _ = self.derivs(z, T, P)
            grad = gr - torch.cat([lam, torch.zeros_like(lam[:, :2])], -1)
            grad = torch.where(gate8, grad, 0.0)
            d = torch.where(gate8, self._sf_direction(grad, H), 0.0)

            def obj(zt, K):
                g, ymin = self.value(zt, T.repeat_interleave(K), P.repeat_interleave(K),
                                     ymask.repeat_interleave(K, 0))
                return g - (zt[:, :6] * lam.repeat_interleave(K, 0)).sum(-1), ymin > 0
            z, f = self._line_search(z, d, obj)
        return z, f

    # ------------------------------------------------------- 3. two-phase
    def _two_phase_parts(self, u, r0):
        q0 = u[:, :1]
        n1, n2 = q0, 1 - q0
        c1 = u[:, 1:7] / n1
        c2 = (r0 - u[:, 1:7]) / n2
        return n1, n2, torch.cat([c1, u[:, 7:9]], -1), torch.cat([c2, u[:, 9:11]], -1)

    def two_phase_obj(self, u, r0, T, P, ymask):
        n1, n2, z1, z2 = self._two_phase_parts(u, r0)
        g1, y1 = self.value(z1, T, P, ymask)
        g2, y2 = self.value(z2, T, P, ymask)
        feas = (n1[:, 0] > 0) & (n2[:, 0] > 0) & (y1 > 0) & (y2 > 0)
        return n1[:, 0] * g1 + n2[:, 0] * g2, feas

    def two_phase(self, u, r0, T, P, gate, rfree, ymask, iters=30):
        N = u.shape[0]
        eye6 = torch.eye(6, device=u.device, dtype=u.dtype)
        gmask = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=u.device), rfree, gate, gate], -1)
        for _ in range(iters):
            n1, n2, z1, z2 = self._two_phase_parts(u, r0)
            G = torch.zeros(N, 11, device=u.device, dtype=u.dtype)
            Hm = torch.zeros(N, 11, 11, device=u.device, dtype=u.dtype)
            for sgn, n, z, sl in ((1.0, n1, z1, slice(7, 9)), (-1.0, n2, z2, slice(9, 11))):
                g, gr, H, _ = self.derivs(z, T, P)
                c = z[:, :6]
                Pm = torch.cat([-c[:, :, None], eye6.expand(N, 6, 6)], -1)        # (N,6,7)
                Hrr, Hrs, Hss = H[:, :6, :6], H[:, :6, 6:], H[:, 6:, 6:]
                gnm = torch.cat([(g - (gr[:, :6] * c).sum(-1))[:, None], gr[:, :6]], -1)
                G[:, :7] += sgn * gnm
                G[:, sl] = n * gr[:, 6:]
                Hm[:, :7, :7] += Pm.transpose(1, 2) @ Hrr @ Pm / n[:, :, None]
                cross = Pm.transpose(1, 2) @ Hrs                                  # (N,7,2)
                cross[:, 0, :] += gr[:, 6:]
                Hm[:, :7, sl] = sgn * cross
                Hm[:, sl, :7] = sgn * cross.transpose(1, 2)
                Hm[:, sl, sl] = n[:, :, None] * Hss
            G = torch.where(gmask, G, 0.0)
            d = torch.where(gmask, self._sf_direction(G, Hm), 0.0)

            def obj(ut, K):
                return self.two_phase_obj(ut, r0.repeat_interleave(K, 0),
                                          T.repeat_interleave(K), P.repeat_interleave(K),
                                          ymask.repeat_interleave(K, 0))
            u, f = self._line_search(u, d, obj)
        return u, f

    # ------------------------------------------------------- driver
    @torch.no_grad()
    def split(self, X, T, P, tpd_iters=8, split_iters=10, s_iters=12,
              seeds_spec=(('ca', 0.15), ('ca', 0.45), ('r1', 0.75), ('r1', 0.5), ('r1', 0.1)),
              dG_min=1e-7, compact=True, tpd_tol=1e-6):
        """X (N,7) combined endmember fractions, T [K], P [bar].
        Returns dict with beta (phase-1 mole fraction of cpx, 0 if single),
        X1, X2 (N,7), is_split (N,)."""
        r0 = self.x_to_r(X)
        sg, gate = self.s_guess_and_gate(r0)
        rfree = r0.abs() > 1e-9                       # absent species stay absent
        ymask = self.y(torch.cat([r0, sg], -1)) > 1e-12
        s0 = self.relax_s(r0, sg, T, P, gate, s_iters, ymask)
        z0 = torch.cat([r0, s0], -1)
        g0, gr0, _, _ = self.derivs(z0, T, P)
        lam = gr0[:, :6]
        f0 = g0 - (r0 * lam).sum(-1)

        # seeds: move the bulk to a fixed M2-Ca target (via r5, clinoenstatite) or a
        # fixed r1 target (alumino-buffonite/Ti-Tschermak); each seed gets its own
        # TPD + two-phase solve and the lowest-G split wins
        N = X.shape[0]
        seeds = []
        for kind, tgt in seeds_spec:
            rt = r0.clone()
            if kind == 'ca':          # M2 Ca target (pigeonite / augite gap)
                rt[:, 5] = 1 - r0[:, 4] - tgt
            else:                     # 'r1' target (Al-Ti Tschermak gap)
                rt[:, 1] = tgt
            st, _ = self.s_guess_and_gate(rt)
            st = self.relax_s(rt, st, T, P, gate, 4, ymask)
            seeds.append(torch.cat([rt, st], -1))
        K = len(seeds)
        zs = torch.cat(seeds, 0)
        rep = lambda a: a.repeat(K, *([1] * (a.dim() - 1)))
        zt, ft = self.tpd_min(zs, rep(lam), rep(T), rep(P), rep(gate), rep(rfree), rep(ymask), tpd_iters)
        tpd = (ft - rep(f0)) / rep(R * T)

        # two-phase solve only for (seed,row) pairs whose TPD found an unstable
        # direction -- compaction costs one host sync, saves the dead stable-row work
        NK = N * K
        sel = torch.nonzero(tpd < -tpd_tol).squeeze(-1) if compact else torch.arange(NK, device=X.device)
        rowof = sel % N
        sub = lambda a: a[rowof]
        u_all = torch.zeros(NK, 11, device=X.device, dtype=X.dtype)
        phi_all = torch.full((NK,), float('inf'), device=X.device, dtype=X.dtype)
        if sel.numel():
            betas = torch.tensor([0.5, 0.3, 0.15, 0.05, 0.01], device=X.device, dtype=X.dtype)
            Kb = betas.numel(); M_ = sel.numel()
            b = betas.repeat(M_)
            rr = lambda a: sub(a).repeat_interleave(Kb, 0)
            zs_ = zt[sel]
            u = torch.cat([b[:, None], b[:, None] * zs_[:, :6].repeat_interleave(Kb, 0),
                           zs_[:, 6:].repeat_interleave(Kb, 0), rr(s0)], -1)
            fu, fe = self.two_phase_obj(u, rr(r0), rr(T), rr(P), rr(ymask))
            fu = torch.where(fe, fu, torch.full_like(fu, float('inf'))).reshape(M_, Kb)
            kb = fu.argmin(-1)
            u = u.reshape(M_, Kb, 11)[torch.arange(M_, device=X.device), kb]
            u, phi = self.two_phase(u, sub(r0), sub(T), sub(P), sub(gate), sub(rfree), sub(ymask), split_iters)
            phi = torch.where(tpd[sel] < -tpd_tol, phi, torch.full_like(phi, float('inf')))
            u_all[sel], phi_all[sel] = u, torch.nan_to_num(phi, nan=float('inf'))
        phi = phi_all.reshape(K, N)
        best = phi.argmin(0)
        ar = torch.arange(N, device=X.device)
        phi = phi[best, ar]
        u = u_all.reshape(K, N, 11)[best, ar]
        u[:, 0] = torch.where(torch.isfinite(phi), u[:, 0], torch.full_like(u[:, 0], 0.5))  # keep parts finite
        tpd_best = tpd.reshape(K, N).min(0).values

        n1, n2, z1, z2 = self._two_phase_parts(u, r0)
        RT = R * T
        dG = (g0 - phi) / RT
        dist = (z1[:, :6] - z2[:, :6]).abs().max(-1).values
        is_split = (torch.isfinite(phi) & (dG > dG_min) & (dist > 1e-3)
                    & (n1[:, 0] > 1e-5) & (n2[:, 0] > 1e-5))
        beta = torch.where(is_split, n1[:, 0], torch.zeros_like(n1[:, 0]))
        X1 = torch.where(is_split[:, None], self.r_to_x(z1[:, :6]), X)
        X2 = torch.where(is_split[:, None], self.r_to_x(z2[:, :6]), torch.zeros_like(X))
        return dict(beta=beta, X1=X1, X2=X2, is_split=is_split, tpd=tpd_best,
                    dG=torch.where(torch.isfinite(dG), dG, torch.zeros_like(dG)))


# =============================================================================
# Public entry point
# =============================================================================
_MODELS = {}


def _model(device, dtype):
    key = (str(device), dtype)
    if key not in _MODELS:
        _MODELS[key] = CpxTorch(device, dtype, 'hybrid')
    return _MODELS[key]


@torch.no_grad()
def exsolve_cpx(endmember_moles, T_C, P_bar, *, device=None, dtype=None,
                batch_size=2 ** 17, min_moles=1e-12, **split_kwargs):
    """Split pooled clinopyroxene into a major and a minor cpx (lowest-G pair).

    Parameters
    ----------
    endmember_moles : (B, 7) tensor, moles of the cpx endmembers in
        melts_vec.clinopyroxene.ENDMEMBERS order (diopside, clinoenstatite,
        hedenbergite, alumino-buffonite, buffonite, essenite, jadeite).
        Individual entries may be negative (MELTS's own basis allows it).
    T_C, P_bar : (B,) temperature [C] and pressure [bar].
    device, dtype : solve on this device / precision. Defaults: the input's
        device; float32 on CUDA (validated: same accuracy as float64,
        ~5x faster on consumer GPUs), float64 on CPU.
    batch_size : rows per solve call (only cpx-bearing rows are solved).
    min_moles : rows whose pooled cpx has fewer total moles stay unsplit.
    **split_kwargs : passed to CpxTorch.split (iterations, seeds, tolerances).

    Returns
    -------
    major, minor : (B, 7) endmember moles, on the input's device/dtype.
        major + minor == endmember_moles to round-off (the split is done in
        extensive variables, phase 2 = pooled - phase 1). Rows with no split
        (stable single cpx, no cpx, or an EOS-infeasible composition) return
        major = endmember_moles, minor = 0. "Major" is the larger by moles of
        formula units.
    is_split : (B,) bool.
    """
    em = torch.as_tensor(endmember_moles)
    out_dev = em.device
    out_dt = em.dtype if em.is_floating_point() else torch.float64
    dev = torch.device(device) if device is not None else out_dev
    if dtype is None:
        dtype = torch.float32 if dev.type == 'cuda' else torch.float64
    B = em.shape[0]
    # everything stays on the solve device; one transfer back at the end
    em_d = em.to(dev, torch.float64)
    major = em_d.clone()
    minor = torch.zeros_like(major)
    is_split = torch.zeros(B, dtype=torch.bool, device=dev)
    if B:
        n = em_d.sum(-1)
        rows = torch.nonzero(n > min_moles).squeeze(-1)
        if rows.numel():
            m = _model(dev, dtype)
            T_K = torch.as_tensor(T_C, dtype=torch.float64).reshape(-1).to(dev) + 273.15
            P = torch.as_tensor(P_bar, dtype=torch.float64).reshape(-1).to(dev)
            for i in range(0, rows.numel(), batch_size):
                r = rows[i:i + batch_size]
                nr, er = n[r], em_d[r]
                o = m.split((er / nr[:, None]).to(dtype), T_K[r].to(dtype), P[r].to(dtype), **split_kwargs)
                sp = o['is_split']
                n1 = o['beta'].to(torch.float64) * nr
                a = n1[:, None] * o['X1'].to(torch.float64)
                b = er - a                     # phase 2 = pooled - phase 1: exactly conservative
                a_major = (n1 >= nr - n1)[:, None]
                major[r] = torch.where(sp[:, None], torch.where(a_major, a, b), er)
                minor[r] = torch.where(sp[:, None], torch.where(a_major, b, a), torch.zeros_like(er))
                is_split[r] = sp
    return major.to(out_dev, out_dt), minor.to(out_dev, out_dt), is_split.to(out_dev)
    m = _model(dev, dtype)
    T_K = torch.as_tensor(T_C, dtype=torch.float64).reshape(-1) + 273.15
    P = torch.as_tensor(P_bar, dtype=torch.float64).reshape(-1)
    for i in range(0, rows.numel(), batch_size):
        r = rows[i:i + batch_size]
        nr = n[r].to(dev)
        X = (em[r].to(dev, torch.float64) / nr[:, None]).to(dtype)
        o = m.split(X, T_K[r.to(T_K.device)].to(dev, dtype), P[r.to(P.device)].to(dev, dtype), **split_kwargs)
        sp = o['is_split']
        beta = o['beta'].to(torch.float64)
        n1, n2 = beta * nr, (1 - beta) * nr
        # phase 2 = pooled - phase 1 keeps the split exactly conservative
        a = n1[:, None] * o['X1'].to(torch.float64)
        b = em[r].to(dev, torch.float64) - a
        a_major = n1 >= n2
        maj = torch.where((sp & a_major)[:, None], a, b)
        mnr = torch.where((sp & a_major)[:, None], b, a)
        maj = torch.where(sp[:, None], maj, em[r].to(dev, torch.float64))
        mnr = torch.where(sp[:, None], mnr, torch.zeros_like(mnr))
        major[r] = maj.to(out_dev, out_dt)
        minor[r] = mnr.to(out_dev, out_dt)
        is_split[r] = sp.to(out_dev)
    return major, minor, is_split

