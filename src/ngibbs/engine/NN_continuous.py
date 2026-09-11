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

Notation vs. code: `g_phi`/`m_phi`/`n_phi` above are the paper-style names for one
per-phase scalar at three stages of the same computation, all inside
`network_component_moles`. In code they are `g` (the raw `mole_head` branch output,
never itself returned -- `upper_forward` recovers it exactly from `m` via leaky_relu's
inverse when it needs it as a BCE logit), `m` (`leaky_relu(g, leak)`, returned as-is and
used for the mole-regression loss), and `phaseMoles` (`clamp(m, min=0)`, returned as-is
and used for everything downstream: mass balance, phase tables, the presence check
`n_phi > 0`). `g`/`m`/`phaseMoles` is what the rest of this file and `upper_forward` use;
the `_phi` names appear only in prose, to keep the physics argument readable.

Architecture
------------
    core = cat([encoder(x), x])          encoder sized by encoderLayerUp/Down
    chem = softmax heads on core         per multi-component phase
    g    = per-phase branches on cat([core, chem])       (head_order='chem_first'; this is
                                                           `g_phi` above, before leaky_relu/clamp)

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

    `hidden_dim` (the model's `chemBranchBase`) optionally inserts ONE hidden layer
    before the final linear-to-softmax projection: `Linear(input_dim, hidden_dim) ->
    LeakyReLU(leak) -> Linear(hidden_dim, n_components)`. `hidden_dim=0` (the default,
    and every checkpoint trained before this existed) reproduces the original bare
    `Linear(input_dim, n_components)` exactly -- composition used to be a single linear
    readout off `core`, the one branch that never got `_mole_branch`'s configurable
    depth. Deliberately no `chemLayerUp`/`chemLayerDown`: one hidden layer is plenty for
    a per-phase softmax, and it keeps this from growing its own multi-parameter search
    space the way the mole branch has. A checkpoint trained with `chemBranchBase=0`
    will not transfer these weights on warm-start into a model with `chemBranchBase>0`
    (the state_dict shapes/keys differ) -- `chem_heads` reinitializes fresh, the same
    graceful-mismatch handling `moleBranchBase` changes already get via
    `_load_matching_state_dict`.
    """

    def __init__(self, n_components: int, input_dim: int, hidden_dim: int = 0, leak: float = 0.05):
        super().__init__()
        if hidden_dim > 0:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(leak),
                nn.Linear(hidden_dim, n_components),
            )
        else:
            self.fc = nn.Linear(input_dim, n_components)

    def forward(self, x, inf_mask=None, train_inf_mask=None):
        raw = self.fc(x)
        mask = train_inf_mask if train_inf_mask is not None else inf_mask
        if mask is None:
            return F.softmax(raw, dim=-1)
        # torch.finfo(raw.dtype).min rather than a fixed -1.0e9: that constant overflows
        # float16 (max magnitude ~65504) the instant autocast runs this head at half
        # precision -- `torch.full_like` raises rather than clamping. Reading the
        # sentinel off raw's own dtype makes it correct at whatever precision this head
        # actually ran at (float32/bfloat16/float16) instead of assuming float32.
        # Finite (not -inf), same as before: any row this large-but-finite value
        # dominates still gets a hard ~0 post-softmax (exp of the gap underflows to
        # exactly 0 well before reaching that magnitude, in every one of these dtypes),
        # without an actual -inf's risk of a nan gradient.
        raw = torch.where(mask, torch.full_like(raw, torch.finfo(raw.dtype).min), raw)
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

    # ---- interop with builder/training/trainer.py ------------------------------------
    # The training loop asks the model what its parts are called rather than assuming.
    # `None` means "this architecture has no such head, by design": freezing it is a
    # no-op, whereas an unrecognised name is still an error. That distinction is what
    # lets an existing recipe's `which_heads_to_freeze: [sat_head, encoder, mole_head]`
    # run unmodified against a model that has no saturation head.
    head_aliases = {
        'sat_head': None,
        'prior_sat_head': None,
        'middleBrain': 'encoder',
        'chem_head': 'chem_heads',
    }
    upper_regularization_config_key = 'mole_regularization'
    lower_regularization_config_key = 'encoder_regularization'

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
                 lowWD: float = 0, highWD: float = 0, noise: float = 0,
                 description: str = '',
                 ml_indexer=None,
                 device=torch.device('cpu'),
                 **_legacy_config):
        super().__init__()

        # Mass balance is no longer a model component -- `NN_MELTS` owns it entirely
        # (engine/mass_balance.py + NN_MELTS.apply_mass_balance). `**_legacy_config`
        # swallows `massbalance_iters` / `massbalance_tikhonov` / `massbalance_damping`
        # (and anything else) from bundles exported before that move, since
        # `load_model_from_zip` forwards the whole config dict by name.

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
            ContinuousPhaseHead(n, self.core_dim, hidden_dim=chemBranchBase, leak=activation_leak)
            for n, _ in chem_specs)
        self.n_chem_components = len(self.comp_mappingsL)

        # ---- mole heads ----
        mole_in = self.core_dim + (self.n_chem_components if head_order == 'chem_first' else 0)
        self.mole_head = nn.ModuleList(
            self._mole_branch(mole_in) for _ in range(self.n_phases))

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
    def network_component_moles(self, x):
        """The network body: heads -> raw (NON mass-balanced) component moles.

        Mass balance is no longer a model concern -- `NN_MELTS.apply_mass_balance`
        projects the raw moles onto the requested bulk. `forward` here only
        reconstructs the *implied* bulk from the raw moles, exactly as
        `MidLevelNetwork.forward_phase_moles` does. Kept as its own method because
        `upper_forward` (training) and the commented-out `derivative_outputs` block
        below consume the pre-anything quantities directly.

        Returns `(componentMoles_raw, chem_out, m, phaseMoles, phaseProportions)`.
        """
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
        # Forced to float32 regardless of an ambient torch.autocast context (see
        # constants.TRAIN_PRECISION): this is where g_phi -- the one quantity
        # upper_forward's annealed boundary-temperature scheme divides by an
        # arbitrarily small T -- actually comes into existence. upper_forward already
        # forces ITS OWN g_phi/T arithmetic to float32, but that guard only stops
        # computation *on* g_phi from overflowing; it cannot restore precision g_phi
        # never had in the first place. If this layer ran in bf16 (7 mantissa bits) or
        # fp16 (~10-11), g_phi's own quantization step can already exceed T well before
        # the anneal's a_end is reached -- concretely, at this project's real T0 range
        # (~1e-5 trace phases to ~0.06 dominant, see _evaluate_upper_model), bfloat16
        # stops resolving a dominant phase's g_phi past about a=1e-2 and a trace phase's
        # before the anneal even starts. Forcing float32 here, at the source, is what
        # actually protects it -- everything upstream (the encoder, chem_heads) still
        # gets the ambient precision's speed/memory benefit.
        with torch.autocast(device_type=mole_in.device.type, enabled=False):
            g = torch.cat([h(mole_in.float()) for h in self.mole_head], dim=1)   # g_phi (module docstring)

        m = F.leaky_relu(g, self.mole_activation_leak)     # m_phi; signed, keeps gradient below 0
        phaseMoles = torch.clamp(m, min=0.0)               # n_phi; exact zeros, exact sparsity

        compMultipliers = phaseMoles @ self.phaseToCompMap
        phaseProportions = chem_out @ self.variedToAllComp + self.fixed_phaseToCompMap
        return (phaseProportions * compMultipliers, chem_out, m, phaseMoles,
                phaseProportions)
    """
    def tangent_project(self, dn, n, tikhonov: Optional[float] = None):
        Project a component-mole tangent onto `null(A_Omega^T)`.

        The bulk `b` is fixed along a scan and normalised to one element mole, so the
        element totals are constants of the motion and every true tangent satisfies
        `A^T dn = 0` exactly -- verified against HeFESTo's own `fort.42` at 7e-8, which is
        its printed precision. Enforcing it here injects that structure rather than asking
        the network to rediscover it: on an untrained net the projection moves the raw
        tangent by 70% and drops `||A^T dn||` from 1.8e-2 to 1.1e-8.

        `Omega = (n > 0)` restricts the correction to phases that are actually present, so
        a zeroed phase cannot be revived by a mass-balance correction. It is piecewise
        constant and so contributes no gradient, which is correct: `clamp` already gives
        `dn/dP = 0` for an absent phase, matching HeFESTo.
        
        lam_reg = self.projector.tikhonov if tikhonov is None else float(tikhonov)
        A = self.compToEl                                   # (C, E)
        eye64 = torch.eye(A.shape[1], dtype=torch.float64, device=dn.device).unsqueeze(0)
        A_om = A.T.unsqueeze(0) * (n > 0).to(dn.dtype).unsqueeze(1)      # (B, E, C)
        # Double precision from the regularizer onward: see the matching note in
        # MassBalanceProjector.forward -- adding lam_reg in float32 rounds it away on
        # large-magnitude rows, letting a genuinely near-degenerate element pair read
        # as exactly singular to torch.linalg.solve's LU path.
        M = (A_om @ A_om.transpose(1, 2)).double() + lam_reg * eye64
        lam = torch.linalg.solve(M, (dn @ A).unsqueeze(-1).double()).to(dn.dtype)
        return dn - (A_om.transpose(1, 2) @ lam).squeeze(-1)

    def derivative_outputs(self, x, feature_indices, project: bool = True):
        Forward-mode tangents of the component moles w.r.t. chosen input features.

        One JVP per direction, each giving all C components at once -- a
        Jacobian-*vector* product, which is what forward mode is for. Reverse mode would
        need one backward per component: measured at B=128, C=62, forward mode for both
        P and T plus the backward costs 76 ms/step against ~383 ms for the reverse-mode
        Jacobian alone, and that gap grows linearly in C.

        Returns `(value_tuple, [tangent_per_direction])` where `value_tuple` is the primal
        `network_component_moles` output, taken from the first JVP so it costs nothing.

        Caveat worth knowing: each direction is a separate forward, so a stochastic layer
        (dropout) draws a fresh mask per direction and injects noise straight into the
        derivative target. Prefer `dropout0` and `layernorm` over `batchnorm` for
        derivative episodes -- batch norm also makes the tangent batch-coupled, which is
        not a property the physics has.
        
        primal, tangents = None, []
        for idx in feature_indices:
            v = torch.zeros_like(x)
            v[:, int(idx)] = 1.0
            out, dout = torch.func.jvp(lambda z: self.network_component_moles(z), (x,), (v,))
            if primal is None:
                primal = out
            tangents.append(dout[0])
        n = primal[0]
        if project:
            tangents = [self.tangent_project(t, n) for t in tangents]
        return primal, tangents
    """
    
    def forward(self, x, detailed: bool = False, **_ignored):
        componentMoles, chem_out, m, phaseMoles, phaseProportions = \
            self.network_component_moles(x)

        reconBulkUnNormed = componentMoles @ self.compToEl
        totals = reconBulkUnNormed.sum(dim=1).clamp(min=1e-6)
        reconBulk = reconBulkUnNormed / totals.unsqueeze(-1)

        # Output contract shared verbatim with `MidLevelNetwork.forward` so `NN_MELTS`
        # and every downstream consumer never branch on model type. This architecture
        # has no gate: `likelihoods` is the honest presence indicator `phaseMoles > 0`,
        # and the `logMoles` slot carries the signed saturation `m` (the only mole-like
        # quantity it produces). Both are pre-mass-balance, like the gated model's.
        likelihoods = (phaseMoles > 0).to(phaseMoles.dtype)
        if detailed:
            return (likelihoods, chem_out, m, reconBulk,
                    componentMoles / totals.unsqueeze(-1), phaseProportions, phaseMoles)
        return likelihoods, chem_out, m, reconBulk

    # ------------------------------------------------------- training-loop interface
    def transform_mole_targets(self, m_batch):
        """Datasets store moles as `log10(n + molar_epsilon)`, which is MidLevelNetwork's
        output space. This model's mole output is a *signed* saturation in linear moles,
        so the target is un-logged rather than the network being asked to reproduce a
        transform that has no defined value at n = 0 -- the one place the continuity this
        class exists for actually matters.

        `molar_epsilon = 0` means the dataset is already linear (see NN.py's
        `logMoles = phaseMoles` fallback), so pass it through.
        """
        eps = float(self.ml_indexer.molar_epsilon)
        if eps == 0:
            return m_batch
        return torch.clamp(torch.pow(10.0, m_batch) - eps, min=0.0)

    def upper_forward(self, x, binaries=None, T=None):
        """Adapter consumed by `builder.training.trainer._upper_forward`.

        `T`, when given, is the current annealed complementarity-smoothing temperature
        (`T = a*T0`, per-phase, shape broadcastable to `(1, n_phases)`; see
        `trainer.py`'s per-epoch schedule and `ngibbs.utils.file_utils.compute_T0` for
        where `T0` comes from). It replaces the hard `m -> clamp(m, min=0)` mole output
        and the plain `g_phi` logit with softened, complementary versions -- see the
        second half of this docstring. `T=None` (the default) is the ORIGINAL,
        unsmoothed behavior below; every existing recipe that doesn't opt into
        `boundary_temperature:` gets that unchanged.

        `binaries` never enters the network -- it only builds the chemistry loss mask, so
        supervision uses ground-truth support while the forward pass stays gate-free.
        `mole_mask` is all ones on purpose: the gated model could only be supervised
        where a phase was present, but here the value at and below zero is the whole
        point, so absent phases are supervised toward zero rather than ignored.

        `bulk` is the element-mole direction the heads imply, `normalize(n @ compToEl)`.
        `_upper_loss` scores it against the bulk slice of `x` (itself an element-mole
        direction), so a non-zero `bulk` loss weight penalises how much mass-balance
        repair `NN_MELTS.apply_mass_balance` has to do at inference, pulling the raw
        heads toward the feasible manifold rather than leaning on that correction.

        `logits` is NOT None, despite this architecture having no separate saturation
        head. (`g_phi`/`m_phi`/`n_phi` here are `g`/`m`/`phaseMoles` in
        `network_component_moles` -- see the module docstring's "Notation vs. code".)
        `m = leaky_relu(g_phi, leak)` already IS the module's own docstring calling
        `g_phi` "an affinity" whose sign the complementarity condition depends on -- it is
        a logit in every sense but name. `g_phi` is recovered exactly from `m` via
        leaky_relu's inverse (`g=m` where `m>=0`, `g=m/leak` where `m<0` -- leaky_relu is
        a monotonic bijection for `leak != 0`) rather than threading a second value out of
        `network_component_moles`, so this changes no other call site: `forward` is
        untouched, and the physical output `n_phi = clamp(leaky_relu(g_phi), min=0)` is
        unchanged either way.

        Why `T` exists (`T=None` path): the huber loss on `mole=m` alone gives
        essentially no gradient once an absent phase's `m` is merely close to 0 (huber
        is minimised AT 0, so it does not prefer confidently-negative `g_phi` over
        marginally-negative `g_phi`) -- a soft, easily noise-flipped decision boundary.
        Plain `logits=g_phi` (unscaled, unweighted BCE) helps, but is fighting the same
        hard kink in `clamp` at `g_phi=0` that motivated this whole architecture.

        `T != None`: both losses are replaced by smooth, complementary versions of
        themselves, worked out over several turns of design discussion (kept here
        rather than only in chat history, since the reasoning is load-bearing):

        - `mole` becomes `T*softplus(g_phi/T)` instead of raw `m`. This is the
          quantity actually regressed by the huber mole loss (not just the downstream
          physical output), so unlike softening only `clamp`, this changes what the
          mole loss's gradient looks like: `d(T*softplus(g_phi/T))/d(g_phi) =
          sigmoid(g_phi/T)`, well-conditioned near `g_phi=0` and vanishing (not merely
          small) for `g_phi << -T`, i.e. it no longer pulls a confidently-absent
          `g_phi` back toward 0 the way the old leaky_relu-based `m` did. Bias vs. the
          hard target: `T*softplus(g_phi/T) - max(g_phi,0) = T*softplus(-|g_phi|/T)`,
          symmetric, peaking at `T*ln(2)` at `g_phi=0`, negligible beyond `|g_phi|>3T`
          -- shrinks with `T` as the schedule anneals, by construction not a separate
          thing to fix.
        - `logits` becomes `5*g_phi/T0` (T0 = `self.ml_indexer.T0`, the FIXED per-phase
          reference scale computed at export time -- NOT the annealed `T` -- so the
          classifier's absolute confidence calibration means the same thing at every
          point in the schedule: `sigmoid(5) ~= 1` is reached at `g_phi = T0`).
        - The binary loss gets an extra per-sample weight `|tanh(g_phi/T)|` (returned
          as `sat_weight`, consumed by `trainer._upper_loss` via
          `_weighted_binary_loss_gt_positive_only`'s `extra_weight`), vanishing at
          `g_phi=0` and saturating to 1 within a few `T`. This uses the ANNEALED `T`,
          not fixed `T0` -- tied to the same schedule as the mole loss's own
          well-conditioned width, so the two losses' effective coverage hands off
          seamlessly throughout the anneal rather than opening a growing gap where
          neither has much gradient. The point is to let the (now well-behaved) mole
          loss own fine calibration right at the boundary, while BCE owns dragging
          back samples that are confidently wrong far from it, where the mole loss's
          own gradient has vanished.
        """
        componentMoles_raw, chem, m, _, _ = self.network_component_moles(x)
        raw_bl = componentMoles_raw @ self.compToEl
        bulk = raw_bl / raw_bl.sum(dim=1, keepdim=True).clamp(min=1e-6)
        chem_mask = (torch.ones_like(chem) if binaries is None
                     else binaries[:, self.comp_binaries] @ self.comp_mappings)
        # `or` rather than `min=`: guards leak=0 (plain ReLU, non-invertible) without
        # perturbing the normal nonzero case at all.
        leak = self.mole_activation_leak or 1e-6
        g_phi = torch.where(m >= 0, m, m / leak)

        sat_weight = None
        if T is None:
            mole_out = m
            sat_logits = g_phi
        else:
            T0 = getattr(self.ml_indexer, 'T0', None)
            if T0 is None:
                raise ValueError(
                    "upper_forward(T=...) needs ml_indexer.T0 (per-phase vanishing-"
                    "abundance scale) to build the fixed BCE reference scale, but this "
                    "bundle has none. Re-export it (MLexporter.py computes T0 "
                    "automatically) or backfill it with scripts/compute_T0.py, or "
                    "disable boundary_temperature for this episode.")
            # Forced to float32 regardless of an ambient torch.autocast context (see
            # constants.TRAIN_PRECISION): T anneals down to a small fraction of an
            # already-small per-phase T0, and this divides by both. autocast's built-in
            # "always run in fp32" op list covers known-sensitive ops like softmax and
            # reductions, but an arbitrary tensor division like this isn't on it -- left
            # unguarded it would run in whatever the ambient dtype is, and float16's
            # ~65504 max is well within reach of g_phi/T late in the anneal. This is
            # exactly the regime the whole smoothing scheme exists to get right, so it
            # must not silently lose range/precision to autocast.
            with torch.autocast(device_type=g_phi.device.type, enabled=False):
                g_phi32 = g_phi.float()
                T32 = T.float() if torch.is_tensor(T) else torch.as_tensor(T, dtype=torch.float32, device=g_phi.device)
                T0_t = torch.as_tensor(T0, dtype=torch.float32, device=g_phi.device).reshape(1, -1)
                mole_out = T32 * torch.nn.functional.softplus(g_phi32 / T32)
                sat_logits = 5.0 * g_phi32 / T0_t
                sat_weight = torch.tanh(g_phi32 / T32).abs()

        return {'logits': sat_logits,
                'chem': chem * chem_mask,
                'chem_mask': chem_mask,
                'mole': mole_out,
                'bulk': bulk,
                'mole_mask': torch.ones_like(m),
                'sat_weight': sat_weight,
                # Raw g_phi, always populated regardless of T/annealing (unlike
                # `logits`, which is scaled by T0 only when T is given) -- diagnostic
                # consumers (trainer.py's per-epoch boundary histograms) want this
                # exact quantity, not whichever of the two `logits` happens to be.
                'g_phi': g_phi}


def load_continuous_from_zip(zip_path, substitutions=None, epsilon=None,
                             load_prefixes=None):
    """Thin wrapper on `NN.load_model_from_zip` with `model_class=ContinuousModel`.

    Same zip layout as the gated model -- state_dict.pt, config.json, ml_indexer/, plus
    the yaml/stats/log extras -- so nothing about packaging or deployment changes.
    """
    return load_model_from_zip(zip_path, substitutions=substitutions,
                               epsilon=epsilon, load_prefixes=load_prefixes,
                               model_class=ContinuousModel)
