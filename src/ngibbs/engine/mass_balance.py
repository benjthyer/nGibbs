"""Mass-balance correction for emulator component-mole predictions.

Both emulator architectures (`MidLevelNetwork`, `ContinuousModel`) emit raw, NON
mass-balanced component moles. `NN_MELTS` is the single owner of the correction that
projects those onto `n @ compToEl ~= b` (the requested bulk); this module holds the
iterative projector it uses. The one-shot pseudo-inverse alternative still lives on
`NN_MELTS.polish_masses`; `NN_MELTS.apply_mass_balance` dispatches between them and a
no-op by name (`"iterative"` / `"pinv"` / `"none"`).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MassBalanceProjector(nn.Module):
    """Support-restricted minimum-norm correction onto `n @ compToEl ~= b`.

    A zeroed phase must not participate in reactions, so the correction is restricted to
    the active support `Omega = (n > 0)`:

        delta = W A_Om^T (A_Om W^2 A_Om^T + lam I)^-1 r,   A_Om = compToEl^T masked to Omega

    Entries can only leave the support and never re-enter, so `Omega` is monotonically
    non-increasing and the iteration terminates rather than merely converging -- unlike
    generic alternating projection, whose rate degrades precisely where a phase is going
    out. The solve is (B, E, E) with E the element count, i.e. a batched ~8x8, and runs
    on whatever device `n` / `compToEl` are on (CUDA-safe: `torch.linalg.solve` + a
    double-precision regularizer).

    Weighting (`relative=True`, the default): `W = diag(n)` clamped to `weight_floor`, so
    the objective is `sum (delta_c / n_c)^2` -- a low-abundance phase is corrected in
    proportion to its own size instead of being pushed by an absolute-scale amount that
    is relatively enormous. This matches `NN_MELTS.polish_masses`' objective exactly (it
    scales `compToEl` rows by the component moles and rescales the solution the same
    way), so the two correctors differ only in support-restriction, non-negativity
    clamping and iteration -- `polish_masses` is the single-shot unclamped special case.
    A genuinely trace phase that is the *sole* source of an imbalance then cannot be the
    route by which it is fixed; in practice imbalance spreads across the major phases, so
    this is the intended behaviour (it keeps low-modal-abundance terms -- to which the
    EOS metamorphic quantities are very sensitive -- close to their predicted values).
    Set `relative=False` for the unweighted `min ||delta||_2` correction, for
    intercomparison.

    Only the composition *direction* is constrained. The target is rebuilt on the current
    scale every iteration; holding a scale fixed across iterations makes the residual
    non-monotone, because each clamp changes the element total.
    """

    def __init__(self, iters: int = 3, tikhonov: float = 1.0e-6, damping: float = 1.0,
                 relative: bool = True, weight_floor: float = 1.0e-9):
        super().__init__()
        self.iters = int(iters)
        self.tikhonov = float(tikhonov)
        self.damping = float(damping)
        self.relative = bool(relative)
        self.weight_floor = float(weight_floor)

    def forward(self, n, compToEl, b_dir):
        E = compToEl.shape[1]
        eye64 = torch.eye(E, dtype=torch.float64, device=n.device).unsqueeze(0)
        At = compToEl.T.unsqueeze(0)

        for _ in range(self.iters):
            bl = n @ compToEl
            tot = bl.sum(dim=1, keepdim=True).clamp(min=1e-6)
            r = b_dir * tot - bl
            omega = (n > 0).to(n.dtype)
            if self.relative:
                # w = 0 for an absent phase (omega), else n clamped up off zero so a
                # present-but-tiny phase still has a solvable column.
                w = torch.clamp(n, min=self.weight_floor) * omega
            else:
                w = omega
            A_w = At * w.unsqueeze(1)                              # (B, E, C)
            # Double precision from the regularizer onward, not just at solve time.
            # Two supported elements can be exactly degenerate given the current active
            # phases (two columns of A_w coincide), which + tikhonov*I is meant to
            # rescue -- but a diagonal entry here commonly runs into the tens, where
            # float32's ULP (~35 * 2^-23 ~= 4e-6) exceeds tikhonov=1e-6: adding the
            # regularizer in float32 rounds it away to nothing on exactly the rows that
            # need it, and torch.linalg.solve's LU path then reads M as truly singular.
            M = (A_w @ A_w.transpose(1, 2)).double() + self.tikhonov * eye64
            lam = torch.linalg.solve(M, r.unsqueeze(-1).double()).to(n.dtype)
            delta = w * (A_w.transpose(1, 2) @ lam).squeeze(-1)
            n = torch.clamp(n + self.damping * delta, min=0.0)

        bl = n @ compToEl
        resid = (bl / bl.sum(dim=1, keepdim=True).clamp(min=1e-6) - b_dir).norm(dim=1)
        return n, resid
