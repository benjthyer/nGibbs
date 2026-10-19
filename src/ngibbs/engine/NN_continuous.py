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
from ..utils.string_utils import pull_letter, pull_number_range, _number_pattern


# --------------------------------------------------------------------------- #
#  Regularization specs
# --------------------------------------------------------------------------- #
# `encoder_regularization` / `mole_regularization` are normalisation only ('none' |
# 'layernorm' | 'batchnorm'), per part -- model properties. Dropout is a TRAINING
# setting, one shared spec `dropout: <initial>dropout<max>` (e.g. '0.0dropout0.3') from
# the recipe: every hidden layer of the encoder and mole heads carries an nn.Dropout
# (p=0 unless the trainer sets it), so its rate is never part of the model.
#
# Older specs folded dropout into the per-part key ('0.0layernormdropout0.3'). Those
# are still accepted: the model keeps only the normalisation, and main.py/tuners take
# the dropout numbers as the training spec. Numbers in a per-part spec only ever come
# from something written by hand (a recipe block, a tune grid), so they override the
# shared spec rather than being ignored.

_NORMS = ('none', 'layernorm', 'batchnorm')


def _dropout_from_numbers(spec):
    """'0.0layernormdropout0.3' -> '0.0dropout0.3', keeping the numbers' own text so a
    round trip does not reformat them. One number means max only (initial 0)."""
    tokens = _number_pattern.findall(str(spec))
    if not tokens:
        raise ValueError(f"dropout spec {spec!r} has no rate; write e.g. '0.0dropout0.3'.")
    initial = tokens[0] if len(tokens) >= 2 else '0.0'
    spec = f"{initial}dropout{tokens[-1]}"
    lo, hi = pull_number_range(spec)
    if not (0.0 <= lo < 1.0 and 0.0 <= hi < 1.0):
        raise ValueError(f"dropout rates must be in [0, 1), got {spec!r}")
    return spec


def split_regularization(encoder_regularization='none', mole_regularization='none',
                         dropout=None):
    """Resolve (encoder_norm, mole_norm, dropout_spec) from new- or old-style config.

    Returns dropout_spec 'none' when no part asks for dropout (no dropout layers built).
    """
    def _norm(spec):
        name = pull_letter(str(spec)).lower().replace('dropout', '') or 'none'
        if name not in _NORMS:
            raise ValueError(f"regularization {spec!r}: normalisation must be one of {_NORMS}"
                             f" (dropout goes in the shared `dropout` key).")
        return name

    def _legacy_dropout(spec):
        return 'dropout' in pull_letter(str(spec)).lower()

    enc_norm, mole_norm = _norm(encoder_regularization), _norm(mole_regularization)
    if dropout is None or str(dropout).strip().lower() in ('', 'none', 'null'):
        shared = 'none'
    elif isinstance(dropout, (int, float)):
        shared = _dropout_from_numbers(f"0.0dropout{dropout}")
    else:
        shared = _dropout_from_numbers(dropout)

    enc_legacy, mole_legacy = _legacy_dropout(encoder_regularization), _legacy_dropout(mole_regularization)
    if enc_legacy or mole_legacy:
        # The mole spec drove the adaptive rate under the old scheme, so it wins a tie.
        source = mole_regularization if mole_legacy else encoder_regularization
        legacy = _dropout_from_numbers(source)
        if enc_legacy and mole_legacy and _dropout_from_numbers(encoder_regularization) != legacy:
            print(f"[regularization] encoder ({encoder_regularization}) and mole "
                  f"({mole_regularization}) dropout differ; dropout is now one shared rate, "
                  f"using the mole spec's: {legacy}")
        if shared != 'none' and shared != legacy:
            print(f"[regularization] per-part dropout {legacy} overrides dropout={shared}")
        shared = legacy
    return enc_norm, mole_norm, shared


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
    # MELTS pyroxene/spinel legality corrections (they need only detail_label_indices,
    # set by _set_indexer). Not applied in forward(): NN_MELTS runs them inside the
    # iterative mass balance, after every correction step.
    polish_negative_px = MidLevelNetwork.polish_negative_px
    polish_negative_cpx = MidLevelNetwork.polish_negative_cpx
    polish_negative_spFe = MidLevelNetwork.polish_negative_spFe
    polish_negative_spAl = MidLevelNetwork.polish_negative_spAl
    polish_negative_sp = MidLevelNetwork.polish_negative_sp

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
    # Dropout rate is a training setting (see split_regularization); the trainer adapts
    # every nn.Dropout in the model, mole heads included, from `train_params['dropout']`.
    dropout_is_training_setting = True
    adaptive_dropout_exclude = ()
    # Always trained with annealed boundary temperature T = a*T0 (needs ml_indexer.T0);
    # the trainer supplies a default schedule when the recipe gives none.
    requires_boundary_temperature = True

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
                 description: str = '',
                 ml_indexer=None,
                 device=torch.device('cpu'),
                 **_legacy_config):
        super().__init__()

        # Mass balance is no longer a model component -- `NN_MELTS` owns it entirely
        # (engine/mass_balance.py + NN_MELTS.apply_mass_balance). `**_legacy_config`
        # swallows `massbalance_iters` / `massbalance_tikhonov` / `massbalance_damping`
        # (and anything else) from bundles exported before that move, since
        # `load_model_from_zip` forwards the whole config dict by name. It likewise drops
        # the training settings older checkpoints recorded here (lowWD, highWD, noise and
        # AffinityModel's loss weights): those describe how a model is trained, not what
        # it is, and now come from the episode config (see trainer.TRAIN_PARAM_DEFAULTS).

        # `MidLevelNetwork.save` runs the config through `apply_type_conversions(...,
        # default_dtype=str)`, so any key absent from TYPE_CONVERSION_MAP round-trips as
        # a string. Coercing here rather than registering each key keeps the class
        # loadable no matter what the map does or does not know about, and means adding
        # a config later cannot silently break `load_continuous_from_zip`.
        encoderLayerUp, encoderLayerDown = int(encoderLayerUp), int(encoderLayerDown)
        encoderBase = int(encoderBase)
        moleLayerUp, moleLayerDown = int(moleLayerUp), int(moleLayerDown)
        moleBranchBase, chemBranchBase = int(moleBranchBase), int(chemBranchBase)
        activation_leak = float(activation_leak)
        # `mole_activation_leak` used to set the mole heads' hidden-layer slope separately;
        # every hidden layer now uses `activation_leak`. Every checkpoint written had the two
        # equal, so this loads them unchanged -- and refuses one that did not.
        legacy_mole_leak = _legacy_config.get('mole_activation_leak')
        if legacy_mole_leak is not None and float(legacy_mole_leak) != activation_leak:
            raise ValueError(
                f"Checkpoint has mole_activation_leak={legacy_mole_leak} != activation_leak="
                f"{activation_leak}; mole heads now share activation_leak, so this model "
                f"would not load as trained.")
        head_order = str(head_order)
        encoder_regularization, mole_regularization, _ = split_regularization(
            encoder_regularization, mole_regularization)

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
        self.description = description

        self.config = dict(
            encoderLayerUp=encoderLayerUp, encoderLayerDown=encoderLayerDown,
            encoderBase=encoderBase,
            moleLayerUp=moleLayerUp, moleLayerDown=moleLayerDown,
            moleBranchBase=moleBranchBase, chemBranchBase=chemBranchBase,
            head_order=head_order,
            encoder_regularization=encoder_regularization,
            mole_regularization=mole_regularization,
            activation_leak=activation_leak,
            description=description,
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
    def _reg(self, norm, neurons, leak):
        """[norm] -> LeakyReLU -> Dropout. The Dropout is always built, at p=0: its rate
        is a training setting the trainer sets each episode. Same module order as the old
        combined specs with dropout, so those checkpoints load index-for-index; ones
        built without dropout are remapped by `upgrade_state_dict`."""
        mods = []
        if norm == 'batchnorm':
            mods.append(nn.BatchNorm1d(int(neurons)))
        elif norm == 'layernorm':
            mods.append(nn.LayerNorm(int(neurons)))
        mods.append(nn.LeakyReLU(leak))
        mods.append(nn.Dropout(0.0))
        return mods

    def upgrade_state_dict(self, state_dict):
        """Remap a checkpoint built without Dropout layers onto this layout.

        A Dropout has no parameters but still takes a slot in its nn.Sequential, so a
        model built before dropout layers were always present numbers its Linear/norm
        layers differently. Per Sequential: if the saved keys put a parameter on a slot
        that is a Dropout or LeakyReLU here, the checkpoint is the old layout, and its
        k-th module maps to this layout's k-th non-Dropout module. Called by
        NN._load_matching_state_dict before keys are matched."""
        out = dict(state_dict)
        for name, seq in self.named_modules():
            if not isinstance(seq, nn.Sequential) or not any(
                    isinstance(m, nn.Dropout) for m in seq):
                continue
            prefix = f"{name}."
            keys = [k for k in out if k.startswith(prefix) and k[len(prefix):].split('.')[0].isdigit()]
            slots = {int(k[len(prefix):].split('.')[0]) for k in keys}
            has_params = lambda i: not isinstance(seq[i], (nn.Dropout, nn.LeakyReLU))
            if all(i < len(seq) and has_params(i) for i in slots):
                continue  # already this layout
            new_idx = [i for i, m in enumerate(seq) if not isinstance(m, nn.Dropout)]
            if not all(i < len(new_idx) and has_params(new_idx[i]) for i in slots):
                continue  # a different geometry, not the old layout: leave to shape matching
            moved = {}
            for k in keys:
                slot, rest = k[len(prefix):].split('.', 1)
                moved[f"{prefix}{new_idx[int(slot)]}.{rest}"] = out.pop(k)
            out.update(moved)
        return out

    def _mole_branch(self, in_dim):
        w = int(self.moleBranchBase)
        layers = [nn.Linear(in_dim, w)] + self._reg(self.mole_regularization, w,
                                                    self.activation_leak)
        for _ in range(self.moleLayerUp):
            w2 = w * 2
            layers += [nn.Linear(w, w2)] + self._reg(self.mole_regularization, w2,
                                                     self.activation_leak)
            w = w2
        for _ in range(self.moleLayerDown):
            w2 = max(1, w // 2)
            layers += [nn.Linear(w, w2)] + self._reg(self.mole_regularization, w2,
                                                     self.activation_leak)
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
    def signed_output(self, g):
        """The signed per-phase quantity `forward` reports in its third slot (unused by
        the emulator, which reads phaseMoles). Here: g itself."""
        return g

    def network_component_moles(self, x):
        """The network body: heads -> raw (NON mass-balanced) component moles.

        Mass balance is no longer a model concern -- `NN_MELTS.apply_mass_balance`
        projects the raw moles onto the requested bulk. `forward` here only
        reconstructs the *implied* bulk from the raw moles, exactly as
        `MidLevelNetwork.forward_phase_moles` does. Kept as its own method because
        `upper_forward` (training) and the commented-out `derivative_outputs` block
        below consume the pre-anything quantities directly.

        Returns `(componentMoles_raw, chem_out, g, phaseMoles, phaseProportions)`; g is
        the raw mole-head output, and phaseMoles = relu(g) needs no activation setting.
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

        phaseMoles = torch.clamp(g, min=0.0)               # n_phi = relu(g); exact zeros

        compMultipliers = phaseMoles @ self.phaseToCompMap
        phaseProportions = chem_out @ self.variedToAllComp + self.fixed_phaseToCompMap
        return (phaseProportions * compMultipliers, chem_out, g, phaseMoles,
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
        componentMoles, chem_out, g, phaseMoles, phaseProportions = \
            self.network_component_moles(x)
        m = self.signed_output(g)

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

        `T` is REQUIRED: the current annealed complementarity-smoothing temperature
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
        `g_phi` is what the module's own docstring calls "an affinity" whose sign the
        complementarity condition depends on -- a logit in every sense but name. It is the
        raw, un-activated mole-head output (`network_component_moles` returns it directly),
        so the logits, the softplus below and `|tanh(g_phi/T)|` all see `g_phi` itself, never
        an activated value. The physical output is `n_phi = relu(g_phi)`.

        Why `T` exists (the hard, un-annealed loss this replaced -- no longer available): the mole loss on `mole=m` (whichever
        `loss_config.moles.type` -- symmetric_rel_l2 by default, which reduces to |m| for
        an absent phase) is minimised AT m=0, so it pulls a confidently-absent phase back
        onto the boundary rather than preferring confidently-negative `g_phi` over
        marginally-negative `g_phi` -- a soft, easily noise-flipped decision boundary.
        Plain `logits=g_phi` (unscaled, unweighted BCE) helps, but is fighting the same
        hard kink in `clamp` at `g_phi=0` that motivated this whole architecture.

        `T != None`: both losses are replaced by smooth, complementary versions of
        themselves, worked out over several turns of design discussion (kept here
        rather than only in chat history, since the reasoning is load-bearing):

        - `mole` becomes `T*softplus(g_phi/T)` instead of raw `m`. This is the
          quantity actually regressed by the mole loss (not just the downstream
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
        componentMoles_raw, chem, g_phi, _, _ = self.network_component_moles(x)
        raw_bl = componentMoles_raw @ self.compToEl
        bulk = raw_bl / raw_bl.sum(dim=1, keepdim=True).clamp(min=1e-6)
        chem_mask = (torch.ones_like(chem) if binaries is None
                     else binaries[:, self.comp_binaries] @ self.comp_mappings)

        if T is None:
            raise ValueError(
                "ContinuousModel always trains with annealed boundary temperature: "
                "upper_forward needs T (trainer.train_Upper_MELTS supplies it, with a "
                "default schedule if the recipe gives none).")
        T0 = getattr(self.ml_indexer, 'T0', None)
        if T0 is None:
            raise ValueError(
                "upper_forward needs ml_indexer.T0 (per-phase vanishing-abundance scale). "
                "main.py computes it from the validation set when the bundles lack it; "
                "otherwise backfill with scripts/compute_T0.py.")
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
                'mole_mask': torch.ones_like(g_phi),
                'sat_weight': sat_weight,
                # Raw g_phi (unlike `logits`, which is scaled by T0) -- diagnostic
                # consumers (trainer.py's per-epoch boundary histograms) want this
                # exact quantity.
                'g_phi': g_phi}


def _coerce_float_list(v):
    """Config values round-trip through MidLevelNetwork.save as strings (a list becomes a
    list of strings, or occasionally one string); accept any of those."""
    if v is None:
        return None
    if isinstance(v, str):
        import ast
        v = ast.literal_eval(v)
    if torch.is_tensor(v):
        v = v.detach().cpu().tolist()
    return [float(x) for x in np.asarray(v, dtype=np.float64).ravel()]


# Training-only output activation settings. Inference uses relu(g) for both models;
# ContinuousModel trains through T*softplus(g/T) (upper_forward) and needs none.
#   affinity_eps     : AffinityModel's slope below the tanh plateau. Also used to build
#                      the affinity LABELS (AffinityTargets), so main.py takes it from the
#                      recipe's `affinity: eps:` block -- model and labels cannot disagree.
ACTIVATION_TRAIN_DEFAULTS = {'affinity_eps': 0.02}

# Loss weights for AffinityModel.affinity_mole_loss. Training settings, not model
# properties: the trainer passes them per episode and they are not saved with the model.
AFFINITY_LOSS_DEFAULTS = {
    'affinity_margin': 0.5,
    'affinity_hinge_weight': 1.0,
    'affinity_pull_weight': 0.05,
    'affinity_present_margin': 0.03,
    'affinity_present_weight': 0.0,
}


class AffinityModel(ContinuousModel):
    """ContinuousModel trained on signed affinity labels instead of moles.

    The mole head's raw output g is mapped through the same activation the labels were:

        y_hat = g                            g >= 0     (present: y_hat IS n)
              = c*tanh(g/c) + eps*g          g <  0     (absent: saturates near -c)

    with one scale c per phase (c = dT_window * median slope, from the Test bundle's
    cStats). n = clamp(y_hat, 0) = clamp(g, 0), so every downstream consumer of
    `phaseMoles` (mass balance, NN_MELTS) is unchanged. There is no saturation head, no
    T0 and no annealed softplus: the sign of y_hat is the presence call, and the label
    already carries how far a phase is from saturating.

    The loss lives here (`affinity_mole_loss`), not in the trainer, so the trainer only
    has to know that a model declares one. Per cell, in units of c (u = y_hat/c,
    t = y/c):

        trusted   : (u - t)^2  [+ present_weight * relu(min(t, present_margin) - u)^2  if t > 0]
        untrusted : hinge_weight * relu(u + margin)^2 + pull_weight * (u - t)^2

    where an untrusted cell is a NaN sentinel or a low-slope-filtered one, and its
    target is the -c placeholder (t = -1): the phase is known absent, only its distance
    from saturation is not.

    The present-side hinge mirrors the untrusted one. Where presence is ambiguous, MSE
    returns the conditional mean, and with absent labels reaching down to -c against
    present ones of a few hundredths of c that mean sits below zero: a systematic
    present->absent miss (precision/recall cross near u = -0.05 for most phases; see
    scripts/affinity_threshold_sweep.py). The hinge makes undershooting a present cell
    cost more than overshooting it, which lifts that mean. Its threshold is
    min(t, present_margin) rather than present_margin alone so it never pulls against
    the regression term: an isolated present cell is still minimised exactly at u = t,
    so trace masses are not inflated toward the margin. Off (weight 0) by default, so
    checkpoints trained before it existed load and score unchanged.
    """

    # Trained through its own tanh activation, not the annealed softplus; no T, no T0.
    requires_boundary_temperature = False

    def __init__(self, affinity_c=None, **kwargs):
        super().__init__(**kwargs)
        ml_indexer = kwargs.get('ml_indexer')
        c = _coerce_float_list(affinity_c)
        if c is None:
            c = _coerce_float_list(getattr(ml_indexer, 'affinity_c', None))
        if c is None:
            raise ValueError(
                "AffinityModel needs per-phase c: pass affinity_c, or set "
                "ml_indexer.affinity_c (main.py does this from the Test bundle's cStats).")
        if len(c) != self.n_phases:
            raise ValueError(f"affinity_c has {len(c)} entries; the model has "
                             f"{self.n_phases} mole heads")
        if not all(np.isfinite(c)) or min(c) <= 0:
            raise ValueError(f"affinity_c must be finite and positive, got {c}")
        self.register_buffer('aff_c', torch.tensor(c, dtype=torch.float32).reshape(1, -1))
        self.config.update(model_class='AffinityModel', affinity_c=c)

    # ------------------------------------------------------------------ activation
    def _mole_activation(self, g, params=None):
        """Training activation, matching how the labels were built (AffinityTargets):
        h(g) = g above 0, c*tanh(g/c) + eps*g below. eps only keeps gradient alive for a
        confidently-absent phase; it is a training setting (`affinity_eps`)."""
        eps = {**ACTIVATION_TRAIN_DEFAULTS, **(params or {})}['affinity_eps']
        c = self.aff_c.to(g.dtype)
        return torch.where(g >= 0, g, c * torch.tanh(g / c) + eps * g)

    def signed_output(self, g):
        """Affinity at eps = 0 for reporting: y/c bottoms out at -1 ("confidently
        absent", full stop). Positive values are the phase moles, as in relu(g)."""
        c = self.aff_c.to(g.dtype)
        return torch.where(g >= 0, g, c * torch.tanh(g / c))

    def transform_mole_targets(self, m_batch):
        # Affinity targets are already in output space; nothing to un-log.
        return m_batch

    # ------------------------------------------------------- training-loop interface
    def upper_forward(self, x, binaries=None, act_params=None):
        componentMoles_raw, chem, g, _, _ = self.network_component_moles(x)
        y_hat = self._mole_activation(g, act_params)
        raw_bl = componentMoles_raw @ self.compToEl
        bulk = raw_bl / raw_bl.sum(dim=1, keepdim=True).clamp(min=1e-6)
        chem_mask = (torch.ones_like(chem) if binaries is None
                     else binaries[:, self.comp_binaries] @ self.comp_mappings)
        return {'logits': None,
                'chem': chem * chem_mask,
                'chem_mask': chem_mask,
                'mole': y_hat,
                'bulk': bulk,
                'mole_mask': None,
                'sat_weight': None,
                # In units of c: comparable across phases, which is all the trainer's
                # diagnostic histogram needs.
                'g_phi': y_hat / self.aff_c.to(y_hat.dtype)}

    def affinity_mole_loss(self, y_hat, y, mask, phase_weights=None, params=None):
        """Scalar mole loss over every cell (see class docstring). Always float32.
        `params` holds the loss weights (keys of AFFINITY_LOSS_DEFAULTS; missing keys
        take the defaults) -- supplied by the trainer from the episode config."""
        p = {**AFFINITY_LOSS_DEFAULTS, **(params or {})}
        with torch.autocast(device_type=y_hat.device.type, enabled=False):
            c = self.aff_c.float()
            u = y_hat.float() / c
            t = y.float() / c
            mask = mask.to(torch.bool)
            sq = (u - t) ** 2
            hinge = torch.relu(u + p['affinity_margin']) ** 2
            trusted = sq
            if p['affinity_present_weight']:
                present = mask & (t > 0)
                floor = torch.clamp(t, max=p['affinity_present_margin'])
                p_hinge = torch.relu(floor - u) ** 2
                trusted = sq + p['affinity_present_weight'] * torch.where(
                    present, p_hinge, torch.zeros_like(p_hinge))
            per = torch.where(mask, trusted,
                              p['affinity_hinge_weight'] * hinge + p['affinity_pull_weight'] * sq)
            w = (torch.ones_like(per) if phase_weights is None
                 else phase_weights.float().expand_as(per))
            return (per * w).sum() / w.sum().clamp(min=1e-12)


def load_continuous_from_zip(zip_path, substitutions=None, epsilon=None,
                             load_prefixes=None):
    """Thin wrapper on `NN.load_model_from_zip` with `model_class=ContinuousModel`.

    Same zip layout as the gated model -- state_dict.pt, config.json, ml_indexer/, plus
    the yaml/stats/log extras -- so nothing about packaging or deployment changes.
    """
    return load_model_from_zip(zip_path, substitutions=substitutions,
                               epsilon=epsilon, load_prefixes=load_prefixes,
                               model_class=ContinuousModel)
