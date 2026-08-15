"""
Continuous phase-saturation model — standalone.

Independent of `MidLevelNetwork`: there is no lower model, so there is no separate
feature encoder feeding saturation logits and no prior-saturation head. Weights are not
interchangeable with the gated model and are not meant to be.

Why this removes the discontinuity
----------------------------------
In the gated design the gate acts on one variable (a sigmoid likelihood) and the
magnitude on another (the mole head). When `likelihoods > 0.5` flips, moles jump from a
finite value to zero and nothing ties the jump size to zero. Here gate and magnitude are
the same variable:

    m_phi = leaky_relu(g_phi)        signed saturation   (the loss operates on this)
    n_phi = clamp(m_phi, min=0)      molar abundance     (the physics operates on this)

`clamp(leaky(x), min=0)` is continuous at `x = 0` because the leaky branch passes
through zero, so a phase leaving the assemblage is a kink rather than a step. That
matches the physics: near a phase-out boundary `n_phi` goes to zero linearly in reaction
progress. `g_phi` reads as an affinity, and the complementarity condition
`n >= 0, A >= 0, n*A = 0` is exactly what a rectifier encodes.

Architecture
------------
    core = cat([encoder(x), x])          encoder sized by encoderLayerUp/Down
    chem = softmax heads on core         per multi-component phase
    g    = per-phase branches on cat([core, chem])       (head_order='chem_first')

Chem-first by default. Given the endmember fractions and the bulk, mass balance nearly
determines the moles -- `A(chem) n = b` -- so handing `chem` to the mole heads lets them
approximate that solve instead of rediscovering it. It is a concatenation, not a
replacement, so a mole head can ignore chem if it turns out unhelpful. Set
`head_order='parallel'` to run both off `core` alone.

Reuse
-----
`save`, `_set_indexer` and the type-conversion path are borrowed from `NN.py` by
assignment rather than copied, so there is one implementation to maintain.
`load_continuous_from_zip` delegates to `NN.load_model_from_zip(model_class=...)`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .NN import TunableModel, MidLevelNetwork, load_model_from_zip
from ..utils.string_utils import pull_letter, pull_number_range


# --------------------------------------------------------------------------- #
#  Mass balance
# --------------------------------------------------------------------------- #
class MassBalanceProjector(nn.Module):
    """Support-restricted minimum-norm correction onto `n @ compToEl ~ b`.

    A zeroed phase must not participate in reactions, so the correction is restricted to
    the active support `Omega = (n > 0)`:

        delta = A_Om^T (A_Om A_Om^T + lam I)^-1 r,   A_Om = compToEl^T masked to Omega

    Entries can only leave the support and never re-enter, so `Omega` is monotonically
    non-increasing and the iteration terminates rather than merely converging -- unlike
    generic alternating projection, whose rate degrades precisely where a phase is going
    out. The solve is (B, E, E) with E the element count, i.e. a batched 8x8.

    Only the composition *direction* is constrained. The target is rebuilt on the current
    scale every iteration; holding a scale fixed across iterations makes the residual
    non-monotone, because each clamp changes the element total.
    """

    def __init__(self, iters: int = 3, tikhonov: float = 1.0e-6, damping: float = 1.0):
        super().__init__()
        self.iters = int(iters)
        self.tikhonov = float(tikhonov)
        self.damping = float(damping)

    def forward(self, n, compToEl, b_dir):
        E = compToEl.shape[1]
        eye = torch.eye(E, dtype=n.dtype, device=n.device).unsqueeze(0)
        At = compToEl.T.unsqueeze(0)

        for _ in range(self.iters):
            bl = n @ compToEl
            tot = bl.sum(dim=1, keepdim=True).clamp(min=1e-6)
            r = b_dir * tot - bl
            omega = (n > 0).to(n.dtype)
            A_om = At * omega.unsqueeze(1)
            M = A_om @ A_om.transpose(1, 2) + self.tikhonov * eye
            lam = torch.linalg.solve(M, r.unsqueeze(-1))
            delta = (A_om.transpose(1, 2) @ lam).squeeze(-1)
            n = torch.clamp(n + self.damping * delta, min=0.0)

        bl = n @ compToEl
        resid = (bl / bl.sum(dim=1, keepdim=True).clamp(min=1e-6) - b_dir).norm(dim=1)
        return n, resid


# --------------------------------------------------------------------------- #
#  Chemistry head
# --------------------------------------------------------------------------- #
class ContinuousPhaseHead(nn.Module):
    """Softmax over a phase's endmembers, without in-place writes.

    `TunableModel.PhaseHead` does `proportions[torch.isnan(proportions)] = 0.0`, mutating
    a softmax output in place. A single backward tolerates it; the DOUBLE backward that
    Sobolev/derivative training needs (`create_graph=True`) raises

        one of the variables needed for gradient computation has been modified by an
        inplace operation ... output 0 of Softmax

    so this is load-bearing, not cosmetic: derivative training cannot run against the
    original head at all. Same masking semantics, expressed out-of-place.
    """

    def __init__(self, n_components: int, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_components)

    def forward(self, x, inf_mask=None, train_inf_mask=None):
        raw = self.fc(x)
        mask = train_inf_mask if train_inf_mask is not None else inf_mask
        if mask is None:
            return F.softmax(raw, dim=-1)
        raw = torch.where(mask, torch.full_like(raw, -1.0e9), raw)
        p = F.softmax(raw, dim=-1)
        # a fully masked row has no admissible component: emit zeros, as -inf + nan->0 did
        return torch.where(mask.all(dim=-1, keepdim=True), torch.zeros_like(p), p)


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class ContinuousModel(nn.Module):
    """Standalone continuous-saturation network.

    Config keys mirror the gated model's naming where the meaning is unchanged, and drop
    `middleLayerUp` / `middleLayerDown` entirely -- what used to be the middle brain is
    now simply the encoder.
    """

    # Reused verbatim from NN.py rather than reimplemented, so there is one copy to
    # maintain. `save` touches only state_dict(), .config and .ml_indexer, all of which
    # this class provides.
    save = MidLevelNetwork.save
    _set_indexer = TunableModel._set_indexer

    def __init__(self,
                 encoderLayerUp: int = 2,
                 encoderLayerDown: int = 1,
                 encoderBase: int = 64,
                 moleLayerUp: int = 2,
                 moleLayerDown: int = 1,
                 moleBranchBase: int = 32,
                 chemBranchBase: int = 0,
                 head_order: str = 'chem_first',
                 encoder_regularization: str = 'none',
                 mole_regularization: str = 'none',
                 activation_leak: float = 0.05,
                 mole_activation_leak: float = 0.05,
                 massbalance_iters: int = 3,
                 massbalance_tikhonov: float = 1.0e-6,
                 massbalance_damping: float = 1.0,
                 lowWD: float = 0, highWD: float = 0, noise: float = 0,
                 description: str = '',
                 ml_indexer=None,
                 device=torch.device('cpu')):
        super().__init__()

        # `MidLevelNetwork.save` runs the config through `apply_type_conversions(...,
        # default_dtype=str)`, so any key absent from TYPE_CONVERSION_MAP round-trips as
        # a string. Coercing here rather than registering each key keeps the class
        # loadable no matter what the map does or does not know about, and means adding
        # a config later cannot silently break `load_continuous_from_zip`.
        encoderLayerUp, encoderLayerDown = int(encoderLayerUp), int(encoderLayerDown)
        encoderBase = int(encoderBase)
        moleLayerUp, moleLayerDown = int(moleLayerUp), int(moleLayerDown)
        moleBranchBase, chemBranchBase = int(moleBranchBase), int(chemBranchBase)
        activation_leak, mole_activation_leak = float(activation_leak), float(mole_activation_leak)
        massbalance_iters = int(massbalance_iters)
        massbalance_tikhonov = float(massbalance_tikhonov)
        massbalance_damping = float(massbalance_damping)
        lowWD, highWD, noise = float(lowWD), float(highWD), float(noise)
        head_order = str(head_order)

        assert encoderLayerUp >= encoderLayerDown, "encoderLayerUp must be >= encoderLayerDown"
        assert moleLayerUp >= moleLayerDown, "moleLayerUp must be >= moleLayerDown"
        assert head_order in ('chem_first', 'parallel'), \
            "head_order must be 'chem_first' or 'parallel'"

        self._set_indexer(ml_indexer)
        self.n_phases = len(list(self.label_indices.keys()))
        self.input_dim = len(ml_indexer.featureNames) + len(self.Elkeys)

        self.encoderLayerUp, self.encoderLayerDown = encoderLayerUp, encoderLayerDown
        self.encoderBase = encoderBase
        self.moleLayerUp, self.moleLayerDown = moleLayerUp, moleLayerDown
        self.moleBranchBase, self.chemBranchBase = moleBranchBase, chemBranchBase
        self.head_order = head_order
        self.encoder_regularization = encoder_regularization
        self.mole_regularization = mole_regularization
        self.activation_leak = activation_leak
        self.mole_activation_leak = mole_activation_leak
        self.lowWD, self.highWD, self.noise, self.description = lowWD, highWD, noise, description

        self.config = dict(
            encoderLayerUp=encoderLayerUp, encoderLayerDown=encoderLayerDown,
            encoderBase=encoderBase,
            moleLayerUp=moleLayerUp, moleLayerDown=moleLayerDown,
            moleBranchBase=moleBranchBase, chemBranchBase=chemBranchBase,
            head_order=head_order,
            encoder_regularization=encoder_regularization,
            mole_regularization=mole_regularization,
            activation_leak=activation_leak,
            mole_activation_leak=mole_activation_leak,
            massbalance_iters=massbalance_iters,
            massbalance_tikhonov=massbalance_tikhonov,
            massbalance_damping=massbalance_damping,
            lowWD=lowWD, highWD=highWD, noise=noise, description=description,
            model_class='ContinuousModel',
        )

        # ---- encoder (what the gated model called middleBrain) ----
        layers, w = [], int(encoderBase)
        layers.append(nn.Linear(self.input_dim, w))
        layers += self._reg(encoder_regularization, w, activation_leak)
        for _ in range(encoderLayerUp):
            w2 = w * 2
            layers += [nn.Linear(w, w2)] + self._reg(encoder_regularization, w2, activation_leak)
            w = w2
        for _ in range(encoderLayerDown):
            w2 = max(1, w // 2)
            layers += [nn.Linear(w, w2)] + self._reg(encoder_regularization, w2, activation_leak)
            w = w2
        self.encoder = nn.Sequential(*layers)
        self._encoder_output_dim = w
        self.core_dim = w + self.input_dim          # core = cat([encoder(x), x])

        # ---- chemistry heads ----
        self.comp_mappingsL, self.comp_binariesL, chem_specs = [], [], []
        j = 0
        for i, (label, inds) in enumerate(self.label_indices.items()):
            if len(inds) > 1:
                self.comp_binariesL.append(i)
                self.comp_mappingsL += np.repeat(j, len(inds)).tolist()
                chem_specs.append((len(inds), i))
                j += 1
        self.chem_heads = nn.ModuleList(
            ContinuousPhaseHead(n, self.core_dim) for n, _ in chem_specs)
        self.n_chem_components = len(self.comp_mappingsL)

        # ---- mole heads ----
        mole_in = self.core_dim + (self.n_chem_components if head_order == 'chem_first' else 0)
        self.mole_head = nn.ModuleList(
            self._mole_branch(mole_in) for _ in range(self.n_phases))

        self.projector = MassBalanceProjector(massbalance_iters, massbalance_tikhonov,
                                              massbalance_damping)
        self._register_buffers(device)

    # ------------------------------------------------------------------ build
    def _reg(self, spec, neurons, leak):
        name = pull_letter(spec).lower()
        mods = []
        if 'batchnorm' in name:
            mods.append(nn.BatchNorm1d(int(neurons)))
        if 'layernorm' in name:
            mods.append(nn.LayerNorm(int(neurons)))
        mods.append(nn.LeakyReLU(leak))
        if 'dropout' in name:
            frac, _ = pull_number_range(spec)
            assert frac is not None and 0.0 <= frac < 1.0
            mods.append(nn.Dropout(frac))
        return mods

    def _mole_branch(self, in_dim):
        w = int(self.moleBranchBase)
        layers = [nn.Linear(in_dim, w)] + self._reg(self.mole_regularization, w,
                                                    self.mole_activation_leak)
        for _ in range(self.moleLayerUp):
            w2 = w * 2
            layers += [nn.Linear(w, w2)] + self._reg(self.mole_regularization, w2,
                                                     self.mole_activation_leak)
            w = w2
        for _ in range(self.moleLayerDown):
            w2 = max(1, w // 2)
            layers += [nn.Linear(w, w2)] + self._reg(self.mole_regularization, w2,
                                                     self.mole_activation_leak)
            w = w2
        layers.append(nn.Linear(w, 1))        # signed saturation, NO activation
        return nn.Sequential(*layers)

    def _register_buffers(self, device):
        num_cols, num_rows = len(self.comp_mappingsL), len(self.chem_heads)
        cm = torch.zeros((num_rows, num_cols), dtype=torch.float32)
        for col, row in enumerate(self.comp_mappingsL):
            cm[int(row), int(col)] = 1.0
        self.register_buffer('comp_mappings', cm)
        self.register_buffer('comp_binaries', torch.tensor(self.comp_binariesL, dtype=torch.long))

        self.compToOx = torch.tensor(self.compToOx_raw, dtype=torch.float, device=device)
        self.oxToEl = torch.tensor(self.oxToEl_raw, dtype=torch.float, device=device)
        self.register_buffer('boolTransCompToEl',
                             torch.tensor(self.boolTransCompToOx_raw @ self.oxToEl_raw, dtype=torch.int))
        self.register_buffer('compositionally_variable_subset',
                             torch.tensor(self.compositionally_variable_subset_raw, dtype=int))
        self.register_buffer('phaseToCompMap', torch.tensor(self.phaseToCompMap_raw, dtype=torch.float))
        self.register_buffer('variedToAllComp', torch.tensor(self.variedToAllComp_raw, dtype=torch.float))
        self.register_buffer('fixed_phaseToCompMap', torch.tensor(self.fixed_phaseToCompMap_raw, dtype=torch.float))
        self.register_buffer('compToEl', torch.tensor(self.compToOx_raw @ self.oxToEl_raw, dtype=torch.float))

    # ---------------------------------------------------------------- forward
    def forward(self, x, detailed: bool = False, **_ignored):
        n_feat = len(self.ml_indexer.featureNames)
        b_target = x[:, n_feat:]
        inf_mask = ((b_target == 0).to(torch.float32)
                    @ self.boolTransCompToEl[self.compositionally_variable_subset]
                    .T.to(torch.float32)) != 0

        core = torch.cat([self.encoder(x), x], dim=1)

        chem_out = torch.cat(
            [head(core, inf_mask=inf_mask[:, (self.comp_mappings[i]).to(torch.bool)])
             for i, head in enumerate(self.chem_heads)], dim=1)

        mole_in = torch.cat([core, chem_out], dim=1) if self.head_order == 'chem_first' else core
        g = torch.cat([h(mole_in) for h in self.mole_head], dim=1)

        m = F.leaky_relu(g, self.mole_activation_leak)     # signed, keeps gradient below 0
        phaseMoles = torch.clamp(m, min=0.0)               # exact zeros, exact sparsity

        compMultipliers = phaseMoles @ self.phaseToCompMap
        phaseProportions = chem_out @ self.variedToAllComp + self.fixed_phaseToCompMap
        componentMoles = phaseProportions * compMultipliers

        componentMoles, resid = self.projector(componentMoles, self.compToEl, b_target)
        reconBulkUnNormed = componentMoles @ self.compToEl
        totals = reconBulkUnNormed.sum(dim=1).clamp(min=1e-6)
        reconBulk = reconBulkUnNormed / totals.unsqueeze(-1)

        if detailed:
            return (chem_out, m, reconBulk, componentMoles / totals.unsqueeze(-1),
                    phaseProportions, phaseMoles, resid)
        return chem_out, m, reconBulk


def load_continuous_from_zip(zip_path, substitutions=None, epsilon=None,
                             load_prefixes=None):
    """Thin wrapper on `NN.load_model_from_zip` with `model_class=ContinuousModel`.

    Same zip layout as the gated model -- state_dict.pt, config.json, ml_indexer/, plus
    the yaml/stats/log extras -- so nothing about packaging or deployment changes.
    """
    return load_model_from_zip(zip_path, substitutions=substitutions,
                               epsilon=epsilon, load_prefixes=load_prefixes,
                               model_class=ContinuousModel)
