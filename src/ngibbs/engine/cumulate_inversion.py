"""
Cumulate inversion for continuous-saturation MELTS emulators (prototype).

Given a cumulate -- a set of cumulus phases and their compositions -- find the
liquid, pressure and temperature it could have crystallised from, by running the
emulator *backwards*: gradient descent on the network INPUTS (P, T, bulk
composition) with the trained weights frozen.

Formulation
-----------
The emulator's input is a bulk composition, not a liquid, so a single cumulate is
compatible with a whole family of bulks that differ only in modal abundance
(liquid + more or less of each cumulus phase). That family is collapsed by adding a
liquid-fraction term: among all bulks saturated in the imposed cumulus phases,
prefer the one with the most liquid. The optimum sits just inside the saturation
surface, so the bulk converges onto the liquid composition and the recovered
(P, T) is the near-liquidus condition. The reported liquid (`liq_*`) is the
network's liquid chemistry at that state -- the liquid in equilibrium with the
assemblage -- not the input bulk.

Per node (one independent optimisation) the loss is

    L = w_asm  * L_assemblage      affinity hinges (below)
      + w_comp * L_composition     cumulus-phase oxide wt% misfit
      + w_liq  * (1 - f_liq)       f_liq = element-mole fraction held by liquid (w_liq = 2)
      + w_bnd  * L_bounds          bulk oxide wt% outside the training range

`ContinuousModel` exposes one signed affinity per phase, `g_phi` (the raw mole-head
output: `n_phi = clamp(leaky_relu(g_phi), 0)`, so the phase is present iff
`g_phi > 0`). Scaling by the per-phase vanishing-abundance scale `ml_indexer.T0`
gives `a_phi = g_phi / T0_phi` (T0 is only a per-phase scale here; the network
itself uses T0 only during training), and the assemblage term is

    required (cumulus phases + liquid):  hinge((delta_p - a) / w)   delta_p = 0.05, w = 0.1
    any other phase:                     free by default (extra_phases='free'): a
                                         phase the user did not list may appear, and
                                         only costs what it takes from f_liq;
                                         extra_phases='forbid' adds hinge((a + delta_a) / w)

with `hinge` a one-sided Huber (zero once satisfied, quadratic, then linear).

Chemical space
--------------
The inversion only claims what the cumulate can constrain. An oxide is
*represented* if some cumulus phase carries >= `min_oxide_wt` of it; the *basis*
is the represented oxides plus the volatiles (H2O, and CO2 in 1.2.x). Oxides
outside the basis are exactly zero in every prior and every iterate. Volatiles
(unless represented, e.g. by hornblende) are carried at their prior value: the
bulk is parameterised as `x = x_fixed + (1 - sum x_fixed) * softmax(z)` with the
softmax over represented elements only, so no gradient ever reaches the rest.
Comparisons with a known liquid should renormalise it onto the same basis
(`InversionResult.truth_liquid`).

Prior
-----
Each node draws random cumulus-phase proportions (Dirichlet), mixes the phase
compositions into a bulk, adds H2O/CO2 from the training-data distribution
(`constants.CUMULATE_VOLATILE_PRIORS`), picks P and T (T <= 1600 C), and runs the
forward model; the predicted liquid becomes the starting bulk, a few degrees below
the T it came from (see `CumulateInverter.initial_nodes`). Every node therefore
starts on a liquidus without a liquidus search. Degeneracy is probed by optimising
many nodes (default 2**15) in parallel and keeping each node's best state;
`InversionResult` summarises the cloud (densities, covariance). That cloud is the
prior pushed through the optimiser's basins, not a posterior.

Liquidus finder
---------------
`find_liquidi` locates the liquidus of many (composition, pressure) pairs in one
vectorised pass: a descending temperature grid followed by bisection to a
tolerance `tol` on the highest crossing of `max_solid(g_phi) = 0`. It runs only
the network body (`network_component_moles`) -- no mass balance, no phase
tables -- and costs `n_grid + ceil(log2(step / tol))` passes.

Scope
-----
Closed, isothermal MELTS models (features exactly Pressure and Temperature, Fe3+
tracked as its own element) built as `ContinuousModel`. Gated (`MidLevelNetwork`)
checkpoints have a hard 0.5 threshold between the saturation head and the moles and
are not supported by the inverter.
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
import tarfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .emulator import NN_MELTS

__all__ = [
    'MLBundle', 'load_ml_bundle', 'load_georoc_pool',
    'find_liquidi', 'CumulateTarget', 'InversionConfig', 'CumulateInverter',
    'InversionResult', 'select_cumulate_rows',
]

LIQUID = 'melts-liquid'


# --------------------------------------------------------------------------- #
#  ML bundle loading
# --------------------------------------------------------------------------- #
@dataclass
class MLBundle:
    """In-memory contents of an ML bundle (`*.tar.gz` written by MLexporter).

    `features` are the network inputs in model units (conditions, then the closed
    element-mole composition); `labels` are the intensive chemistry labels in
    chem-head (`label_indices_comp`) order; `binary_labels`, `molar_labels`,
    `mass_labels` are per-phase (`mass_phasedict` order).
    """
    features: np.ndarray
    labels: np.ndarray
    binary_labels: np.ndarray
    molar_labels: np.ndarray
    mass_labels: np.ndarray
    free_outputs: Optional[np.ndarray] = None
    stats_text: str = ''
    source: str = ''

    def __len__(self):
        return self.features.shape[0]


def load_ml_bundle(path: Union[str, Path]) -> MLBundle:
    """Read an ML bundle (`.tar.gz`, as produced by MLexporter, e.g. the
    `*_Test_subset15000.tar.gz` deployment-test bundles) straight into memory."""
    path = Path(path)
    arrays, stats = {}, ''
    with tarfile.open(path, 'r:*') as tf:
        for member in tf.getmembers():
            name = Path(member.name).name
            if name.endswith('.npy'):
                arrays[name[:-4]] = np.load(io.BytesIO(tf.extractfile(member).read()),
                                            allow_pickle=False)
            elif name == 'stats.txt':
                stats = tf.extractfile(member).read().decode(errors='replace')
    missing = {'features', 'labels', 'binary_labels', 'molar_labels', 'mass_labels'} - set(arrays)
    if missing:
        raise ValueError(f"{path} is not a complete ML bundle (missing {sorted(missing)})")
    return MLBundle(features=arrays['features'], labels=arrays['labels'],
                    binary_labels=arrays['binary_labels'], molar_labels=arrays['molar_labels'],
                    mass_labels=arrays['mass_labels'], free_outputs=arrays.get('free_outputs'),
                    stats_text=stats, source=str(path))


# --------------------------------------------------------------------------- #
#  Emulator geometry (indices, conversions, bounds) shared by everything below
# --------------------------------------------------------------------------- #
class _Geometry:
    """Everything the inverter needs from an `NN_MELTS`, as tensors on one device."""

    def __init__(self, emulator: NN_MELTS):
        if type(emulator.model).__name__ != 'ContinuousModel':
            raise TypeError(
                "Cumulate inversion needs a ContinuousModel checkpoint (signed per-phase "
                f"affinities); got {type(emulator.model).__name__}.")
        ix = emulator.ml_indexer
        if LIQUID not in ix.mass_phasedict:
            raise ValueError("Model has no 'melts-liquid' phase; nothing to invert for.")
        names = list(ix.featureNames)
        p_idx = [i for i, n in enumerate(names) if n.startswith('Pressure')]
        t_idx = [i for i, n in enumerate(names) if n.startswith('Temperature')]
        if len(names) != 2 or len(p_idx) != 1 or len(t_idx) != 1:
            raise ValueError(f"Only closed isothermal models (features = Pressure, Temperature) "
                             f"are supported; this model's features are {names}.")
        if 'Fe3' not in ix.Elkeys:
            raise ValueError("Only closed models (Fe3 tracked as an element) are supported.")

        self.em = emulator
        self.model = emulator.model
        self.ix = ix
        self.dev = torch.device(emulator.dev)          # the model lives here; so does everything else
        dev, f32 = self.dev, torch.float32
        self.P_idx, self.T_idx = p_idx[0], t_idx[0]
        self.n_feat = len(names)
        self.Elkeys = list(ix.Elkeys)
        self.Oxides = list(ix.Oxides)
        self.E, self.O = len(self.Elkeys), len(self.Oxides)
        self.phases = list(ix.all_phases)
        self.phase_col = dict(ix.mass_phasedict)               # phase -> column of g
        self.liq = self.phase_col[LIQUID]
        self.comp_cols = {k: np.asarray(v, dtype=np.int64) for k, v in ix.label_indices.items()}
        self.chem_cols = {k: np.asarray(v, dtype=np.int64) for k, v in ix.label_indices_comp.items()}
        self.variable_phases = [p for p in self.phases if p in self.chem_cols and p != LIQUID]

        # normaliser (differentiable, no in-place writes)
        self.miner = emulator.norm_features.miner.detach().to(dev, f32)
        rng = emulator.norm_features.ranger.detach().to(dev, f32)
        self.ranger = torch.where(rng == 0, torch.ones_like(rng), rng)

        # oxide bookkeeping
        # Network chemistry heads emit pyroxene/spinel in the PxSp-transformed basis
        # (compToOx = inv(PxSpTransform) @ compToOxLoad); bundle LABELS are stored in the
        # native MELTS endmember basis, which projects with compToOxLoad.
        self.compToOx = torch.as_tensor(np.asarray(ix.compToOx), dtype=f32, device=dev)   # (C, O)
        load = getattr(ix, 'compToOxLoad', None)
        self.compToOxNative = (torch.as_tensor(np.asarray(load), dtype=f32, device=dev)
                               if load is not None else self.compToOx)
        self.MMdiag = torch.as_tensor(np.diag(np.asarray(ix.MM)).copy(), dtype=f32, device=dev)[:self.O]
        self.elToOx = emulator.elToOx.detach().to(dev, f32)                               # (E, E)
        self.Minv = emulator.Minv.detach().to(dev, f32)
        self.oxToEl = emulator.oxToEl.detach().to(dev, f32)
        self.compToEl = emulator.compToEl.detach().to(dev, f32)                           # (C, E)
        self.phaseToComp = torch.as_tensor(np.asarray(ix.phaseToCompMap), dtype=f32, device=dev)  # (P, C)
        self.comp_el_count = self.compToEl.sum(1)                                        # element moles / component
        # oxide j <-> element el_of_ox[j] (one cation per oxide in closed models)
        o2e = emulator.oxToEl.detach().cpu().numpy()[:self.O]
        if not ((o2e != 0).sum(1) == 1).all():
            raise ValueError("Expected exactly one cation element per oxide.")
        self.el_of_ox = np.abs(o2e).argmax(1)

        # per-phase vanishing-abundance scale
        T0 = getattr(ix, 'T0', None)
        T0 = np.full(len(self.phases), 1e-3, np.float32) if T0 is None else np.asarray(T0, np.float32)
        T0 = np.where(T0 > 0, T0, np.float32(1e-3))
        self.T0 = torch.as_tensor(T0, dtype=f32, device=dev)

        # training ranges
        tb = getattr(emulator, 'training_bounds', None)
        if tb is not None:
            cond = [tb['conditions'][n] for n in names]
            ox = [tb['oxides_wtpct'].get(o, (0.0, 100.0)) for o in self.Oxides]
        else:
            lo = self.miner[:self.n_feat].cpu().numpy()
            cond = list(zip(lo, lo + self.ranger[:self.n_feat].cpu().numpy()))
            ox = [(0.0, 100.0)] * self.O
        self.P_bounds = tuple(float(v) for v in cond[self.P_idx])
        self.T_bounds = tuple(float(v) for v in cond[self.T_idx])
        self.ox_lo = torch.tensor([b[0] for b in ox], dtype=f32, device=dev)
        self.ox_hi = torch.tensor([b[1] for b in ox], dtype=f32, device=dev)

        leak = float(getattr(self.model, 'mole_activation_leak', 0.05) or 1e-6)
        self.leak = leak
        self.model.eval()

    # ---- conversions ---------------------------------------------------------
    def el_to_oxwt(self, x):
        """Closed element-mole composition (B, E) -> oxide wt% (B, O)."""
        m = (x @ self.elToOx) * self.MMdiag[:self.E]
        return 100.0 * m / m.sum(dim=1, keepdim=True).clamp(min=1e-12)

    def oxwt_to_el(self, w):
        """Oxide wt% (B, O) in `Oxides` order -> closed element moles (B, E)."""
        u = (w @ self.Minv[:self.O, :self.O]) @ self.oxToEl[:self.O]
        return u / u.sum(dim=1, keepdim=True).clamp(min=1e-12)

    def phase_oxwt(self, chem, phase, native: bool = False):
        """Intensive chemistry of one phase -> its oxide wt%. `native=False`: network
        chem-head basis (PxSp-transformed); `native=True`: bundle-label basis."""
        c = chem[:, torch.as_tensor(self.chem_cols[phase], device=chem.device)]
        mat = self.compToOxNative if native else self.compToOx
        C = mat[torch.as_tensor(self.comp_cols[phase], device=chem.device)]
        m = (c @ C) * self.MMdiag
        return 100.0 * m / m.sum(dim=1, keepdim=True).clamp(min=1e-12)

    def pure_oxwt(self, phase):
        """Oxide wt% of a fixed-composition (single-component) phase."""
        cols = self.comp_cols[phase]
        m = self.compToOx[torch.as_tensor(cols, device=self.dev)].sum(0) * self.MMdiag
        return (100.0 * m / m.sum().clamp(min=1e-12)).cpu().numpy()

    def solid_cols(self, ignore=('fluid',)):
        return [self.phase_col[p] for p in self.phases if p != LIQUID and p not in set(ignore)]

    def norm_inputs(self, P, T, x):
        cond = torch.stack([P, T], dim=1) if (self.P_idx, self.T_idx) == (0, 1) else \
            torch.stack([T, P], dim=1)
        feats = torch.cat([cond, x], dim=1)
        return (feats - self.miner) / self.ranger

    def network(self, P, T, x):
        """Run the network body. Returns (componentMoles_raw, chem_out, g)."""
        cm, chem, m, _pm, _pp = self.model.network_component_moles(self.norm_inputs(P, T, x))
        g = torch.where(m >= 0, m, m / self.leak)
        return cm, chem, g

    def phase_fractions(self, cm):
        """Element-mole fraction of the (raw, not mass-balanced) system held by each
        phase, (B, P) in `mass_phasedict` order."""
        per_phase = (cm * self.comp_el_count) @ self.phaseToComp.T              # (B, P)
        return per_phase / per_phase.sum(1, keepdim=True).clamp(min=1e-12)

    def liquid_fraction(self, cm):
        """Element-mole fraction of the (raw, not mass-balanced) system held by liquid."""
        return self.phase_fractions(cm)[:, self.liq]


def _hinge(d):
    """One-sided Huber hinge: 0 for d <= 0, d**2 for 0 < d < 1, 2d - 1 beyond.
    Exactly zero once a constraint is met (so it never competes with the liquid
    term inside the feasible set) and only linear far from it (so one badly wrong
    phase cannot swamp every other gradient)."""
    d = F.relu(d)
    return torch.where(d < 1.0, d * d, 2.0 * d - 1.0)


@contextlib.contextmanager
def _frozen(model):
    flags = [p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, f in zip(model.parameters(), flags):
            p.requires_grad_(f)


# --------------------------------------------------------------------------- #
#  Parallel liquidus finder
# --------------------------------------------------------------------------- #
@torch.no_grad()
def find_liquidi(emulator: Union[NN_MELTS, _Geometry], compositions, pressures, *,
                 composition_space: str = 'elements', T_bounds: Optional[Tuple[float, float]] = None,
                 tol: float = 1.0, n_grid: int = 24, n_bisect: Optional[int] = None,
                 return_phase: bool = True, ignore_phases: Sequence[str] = ('fluid',),
                 batch_size: int = 2 ** 15) -> Dict[str, np.ndarray]:
    """Liquidus temperatures of many (composition, pressure) pairs at once.

    The liquidus is the highest temperature at which any solid phase has a
    positive affinity (`g_phi > 0`). Each row is scanned on a descending grid of
    `n_grid` temperatures to bracket the highest crossing, then bisected until the
    bracket is narrower than `tol` (C). Only the network body runs: no mass
    balance, no phase tables.

    Cost: `n_grid + n_bisect (+1 if return_phase)` network passes per row, with
    `n_bisect = ceil(log2(grid_step / tol))` unless given explicitly. Defaults
    (24-point grid over the 1.0.2 range = 51 C steps, tol 1 C) cost 30 passes.
    A coarser grid is cheaper but can step over a narrow liquid-only window in
    a non-monotonic affinity profile.

    Parameters
    ----------
    emulator : NN_MELTS (closed, isothermal ContinuousModel)
    compositions : (N, E) closed element moles in `ml_indexer.Elkeys` order
        (`composition_space='elements'`), or (N, O) oxide wt% in
        `ml_indexer.Oxides` order (`composition_space='oxides'`).
    pressures : (N,) or scalar, bar.
    T_bounds : (Tmin, Tmax) in C. Default: the model's training range.
    tol : final bracket width, C (the returned T is its midpoint, so within tol/2).
    n_grid : grid points (>= 2).
    n_bisect : override the number of bisection steps (then `tol` is ignored).
    return_phase : also identify the liquidus phase (one extra pass).
    ignore_phases : phases that do not count as crystallisation (default fluid).

    Returns
    -------
    dict of numpy arrays
        'T_liquidus' (N,) C; 'status' (N,) int: 0 = bracketed, +1 = a solid is
        already present at Tmax (liquidus >= Tmax, T_liquidus = Tmax), -1 = no
        solid down to Tmin (liquidus < Tmin, T_liquidus = Tmin);
        'liquidus_phase' (N,) str -- the solid with the largest affinity just
        below the liquidus ('' where status == -1; only if return_phase);
        'passes' -- network passes per row actually spent.
    """
    geo = emulator if isinstance(emulator, _Geometry) else _Geometry(emulator)
    dev = geo.dev
    comp = torch.as_tensor(np.asarray(compositions, dtype=np.float32), device=dev)
    if composition_space in ('oxides', 'oxide', 'wt', 'oxides_wt'):
        comp = geo.oxwt_to_el(comp)
    elif composition_space != 'elements':
        raise ValueError("composition_space must be 'elements' or 'oxides'")
    comp = comp / comp.sum(1, keepdim=True).clamp(min=1e-12)
    N = comp.shape[0]
    P = torch.as_tensor(np.broadcast_to(np.asarray(pressures, np.float32), (N,)).copy(), device=dev)
    Tmin, Tmax = T_bounds if T_bounds is not None else geo.T_bounds
    n_grid = max(int(n_grid), 2)
    step = (Tmax - Tmin) / (n_grid - 1)
    if n_bisect is None:
        n_bisect = max(0, int(math.ceil(math.log2(step / max(tol, 1e-9)))))

    solid = torch.as_tensor(geo.solid_cols(ignore_phases), device=dev)

    def hmax(Pv, Tv, xv):
        out_h = torch.empty(Pv.shape[0], device=dev)
        out_i = torch.empty(Pv.shape[0], dtype=torch.long, device=dev)
        for s in range(0, Pv.shape[0], batch_size):
            e = s + batch_size
            _, _, g = geo.network(Pv[s:e], Tv[s:e], xv[s:e])
            h, i = g[:, solid].max(1)
            out_h[s:e], out_i[s:e] = h, solid[i]
        return out_h, out_i

    grid = torch.linspace(Tmax, Tmin, n_grid, device=dev)            # descending
    Pg = P.repeat_interleave(n_grid)
    Tg = grid.repeat(N)
    xg = comp.repeat_interleave(n_grid, dim=0)
    h, _ = hmax(Pg, Tg, xg)
    present = (h > 0).view(N, n_grid)

    any_solid = present.any(1)
    first = torch.argmax(present.to(torch.int8), dim=1)              # first (hottest) solid-bearing
    status = torch.zeros(N, dtype=torch.long, device=dev)
    status[any_solid & (first == 0)] = 1
    status[~any_solid] = -1

    lo = grid[first].clone()                                         # solid present here
    hi = grid[(first - 1).clamp(min=0)].clone()                      # liquid only here
    br = status == 0
    if br.any():
        idx = torch.nonzero(br).squeeze(1)
        lo_b, hi_b = lo[idx], hi[idx]
        for _ in range(n_bisect):
            mid = 0.5 * (lo_b + hi_b)
            hm, _ = hmax(P[idx], mid, comp[idx])
            sol = hm > 0
            lo_b = torch.where(sol, mid, lo_b)
            hi_b = torch.where(sol, hi_b, mid)
        lo[idx], hi[idx] = lo_b, hi_b

    T_liq = torch.where(status == 0, 0.5 * (lo + hi),
                        torch.where(status == 1, torch.full_like(lo, Tmax), torch.full_like(lo, Tmin)))
    out = {'T_liquidus': T_liq.cpu().numpy(), 'status': status.cpu().numpy(),
           'passes': n_grid + n_bisect + int(return_phase)}
    if return_phase:
        # evaluated on the solid side of the bracket
        _, phase_i = hmax(P, torch.where(status == -1, torch.full_like(lo, Tmin), lo), comp)
        names = np.array([geo.phases[i] for i in phase_i.cpu().numpy()], dtype=object)
        names[out['status'] == -1] = ''
        out['liquidus_phase'] = names
    return out


# --------------------------------------------------------------------------- #
#  GEOROC starting pool
# --------------------------------------------------------------------------- #
def load_georoc_pool(csv_path, emulator: NN_MELTS, *, within_training_bounds: bool = True,
                     ferric_fraction: float = 0.1, max_rows: Optional[int] = None,
                     seed: int = 0) -> pd.DataFrame:
    """Read a GEOROC/PetDB whole-rock table (as in data/MELTStables/GEOROC) into
    oxide wt% in the model's `Oxides` order, closed to 100.

    FeO is taken as FeO-total and split with a fixed `ferric_fraction`
    (Fe3+/sum Fe, molar); the inverter re-randomises this split per node. Oxides the
    model does not carry are dropped; oxides the table lacks (e.g. CO2 for 1.2.x)
    are zero. Rows outside the model's bulk-composition training range are
    dropped when `within_training_bounds`.
    """
    geo = _Geometry(emulator)
    tab = pd.read_csv(csv_path)
    cols = {c.strip(): c for c in tab.columns}
    out = pd.DataFrame(0.0, index=tab.index, columns=geo.Oxides)
    for ox in geo.Oxides:
        if ox in cols and ox not in ('FeO', 'Fe2O3'):
            out[ox] = pd.to_numeric(tab[cols[ox]], errors='coerce')
    feot = pd.to_numeric(tab[cols['FeO']], errors='coerce') if 'FeO' in cols else 0.0
    if 'Fe2O3' in cols:
        feot = feot + 0.8998 * pd.to_numeric(tab[cols['Fe2O3']], errors='coerce').fillna(0)
    MM = dict(zip(geo.Oxides, geo.MMdiag.cpu().numpy()))
    out['FeO'] = feot * (1 - ferric_fraction)
    if 'Fe2O3' in out:
        out['Fe2O3'] = feot * ferric_fraction * MM['Fe2O3'] / (2 * MM['FeO'])
    out = out.fillna(0.0).clip(lower=0.0)
    tot = out.sum(1)
    out = out[tot > 50].div(out[tot > 50].sum(1), axis=0) * 100.0
    if within_training_bounds:
        lo, hi = geo.ox_lo.cpu().numpy(), geo.ox_hi.cpu().numpy()
        ok = ((out.values >= lo) & (out.values <= hi)).all(1)
        out = out[ok]
    if max_rows is not None and len(out) > max_rows:
        out = out.sample(max_rows, random_state=seed)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Target cumulate
# --------------------------------------------------------------------------- #
@dataclass
class CumulateTarget:
    """A cumulus assemblage and its phase compositions.

    phases      : cumulus solids that must be saturated (with liquid).
    oxide_wt    : {phase: (O,) oxide wt% in `ml_indexer.Oxides` order} for every
                  compositionally variable cumulus phase. Fixed-composition phases
                  (apatite, quartz, ...) only need to appear in `phases`.
    free_phases : phases left unconstrained (neither required nor forbidden).
    truth       : optional ground truth (from a bundle row) for validation.
    """
    phases: List[str]
    oxide_wt: Dict[str, np.ndarray]
    free_phases: Tuple[str, ...] = ('fluid',)
    truth: Optional[dict] = None
    name: str = ''

    @classmethod
    def from_bundle(cls, emulator: NN_MELTS, bundle: MLBundle, row: int, *,
                    free_phases: Sequence[str] = ('fluid',), min_mass_pct: float = 0.0,
                    name: Optional[str] = None) -> 'CumulateTarget':
        """Strip the liquid from an equilibrium bundle row and keep its solids.

        The truth (P, T, the liquid's element moles / oxide wt%, and the original
        bulk) is stored for validation. Solids below `min_mass_pct` of the system
        mass are dropped from the cumulate (and left free rather than forbidden).
        """
        geo = _Geometry(emulator)
        b = bundle.binary_labels[row]
        if b[geo.liq] < 0.5:
            raise ValueError(f"Bundle row {row} has no liquid.")
        mass = bundle.mass_labels[row]
        phases, free = [], list(free_phases)
        for p in geo.phases:
            if p == LIQUID or b[geo.phase_col[p]] < 0.5 or p in free:
                continue
            if mass[geo.phase_col[p]] < min_mass_pct:
                free.append(p)
                continue
            phases.append(p)
        if not phases:
            raise ValueError(f"Bundle row {row} has no cumulus solids.")
        chem = torch.as_tensor(bundle.labels[row:row + 1], dtype=torch.float32, device=geo.dev)
        oxide_wt = {p: geo.phase_oxwt(chem, p, native=True)[0].cpu().numpy()
                    for p in phases if p in geo.chem_cols}
        f = bundle.features[row]
        liq_el = bundle.labels[row, geo.chem_cols[LIQUID]].astype(np.float64)
        liq_el = liq_el / liq_el.sum()
        truth = {
            'row': int(row),
            'P_bar': float(f[geo.P_idx]), 'T_C': float(f[geo.T_idx]),
            'liquid_elements': liq_el,
            'liquid_oxide_wt': geo.el_to_oxwt(torch.as_tensor(liq_el[None], dtype=torch.float32,
                                                              device=geo.dev))[0].cpu().numpy(),
            'bulk_elements': f[geo.n_feat:].astype(np.float64),
            'bulk_oxide_wt': geo.el_to_oxwt(torch.as_tensor(f[None, geo.n_feat:], dtype=torch.float32,
                                                            device=geo.dev))[0].cpu().numpy(),
            'mass_pct': {p: float(mass[geo.phase_col[p]]) for p in phases + [LIQUID]},
        }
        truth['emulator'] = _emulator_baseline(geo, truth, phases, oxide_wt, b)
        nm = name or f"row{row}:" + '+'.join(_short(p) for p in phases)
        return cls(phases=phases, oxide_wt=oxide_wt, free_phases=tuple(free), truth=truth, name=nm)

    @classmethod
    def from_oxides(cls, emulator: NN_MELTS, compositions: Dict[str, Dict[str, float]], *,
                    phases: Optional[Sequence[str]] = None, free_phases: Sequence[str] = ('fluid',),
                    name: str = '') -> 'CumulateTarget':
        """Build a target from measured mineral analyses, e.g.
        `{'olivine': {'SiO2': 40.1, 'MgO': 47.9, 'FeO': 11.6, 'CaO': 0.2}, ...}`.
        Oxides the model does not carry are ignored (with a warning); missing ones
        are zero. `phases` defaults to the keys of `compositions`."""
        geo = _Geometry(emulator)
        oxide_wt = {}
        for p, comp in compositions.items():
            if p not in geo.phase_col:
                raise KeyError(f"Unknown phase {p!r}; model phases: {geo.phases}")
            extra = set(comp) - set(geo.Oxides)
            if extra:
                warnings.warn(f"{p}: oxides {sorted(extra)} are not model inputs and are ignored")
            v = np.array([float(comp.get(o, 0.0)) for o in geo.Oxides])
            oxide_wt[p] = 100.0 * v / v.sum()
        ph = list(phases) if phases is not None else list(compositions)
        return cls(phases=ph, oxide_wt=oxide_wt, free_phases=tuple(free_phases), name=name)


@torch.no_grad()
def _emulator_baseline(geo: '_Geometry', truth: dict, phases, oxide_wt, binary_row) -> dict:
    """The forward model at the bundle row's own (P, T, bulk): its liquid, assemblage,
    liquid fraction and cumulus-phase compositions. Separates emulator error (how
    far nGibbs is from MELTS at the true answer) from inversion error."""
    dev = geo.dev
    P = torch.tensor([truth['P_bar']], dtype=torch.float32, device=dev)
    T = torch.tensor([truth['T_C']], dtype=torch.float32, device=dev)
    x = torch.as_tensor(np.asarray(truth['bulk_elements'], np.float32)[None], device=dev)
    cm, chem, g = geo.network(P, T, x)
    present = (g[0] > 0).cpu().numpy()
    solids = [p for p in geo.phases if p not in (LIQUID, 'fluid')]
    pred = sorted(p for p in solids if present[geo.phase_col[p]])
    true = sorted(p for p in solids if binary_row[geo.phase_col[p]] > 0.5)
    cum = {p: geo.phase_oxwt(chem, p)[0].cpu().numpy() for p in phases if p in geo.chem_cols}
    rm = [float(np.sqrt(np.mean((cum[p] - np.asarray(oxide_wt[p])) ** 2))) for p in cum]
    out = {'liquid_oxide_wt': geo.phase_oxwt(chem, LIQUID)[0].cpu().numpy(),
           'liquid_present': bool(present[geo.liq]),
           'f_liquid': float(geo.phase_fractions(cm)[0, geo.liq]),
           'solids_predicted': pred, 'solids_true': true, 'assemblage_match': pred == true,
           'cumulus_oxide_wt': cum, 'cumulus_rmse_wt': float(np.mean(rm)) if rm else float('nan')}
    try:   # the same liquid after NN_MELTS mass balance
        feats = torch.zeros(1, geo.n_feat + geo.E, device=dev)
        feats[:, geo.P_idx], feats[:, geo.T_idx], feats[:, geo.n_feat:] = P, T, x
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mb = geo.em.forwardMB(feats, Normalize=True, WtPercent=False,
                                  outputs=['component_moles'])['component_moles'].to(dev)
        le = mb[:, torch.as_tensor(geo.comp_cols[LIQUID], device=dev)]
        out['liquid_oxide_wt_MB'] = geo.el_to_oxwt(le / le.sum(1, keepdim=True).clamp(min=1e-12))[0].cpu().numpy()
    except Exception:   # pragma: no cover - MB is a nicety here
        pass
    return out


def _short(p):
    return {'olivine': 'ol', 'orthopyroxene': 'opx', 'clinopyroxene': 'cpx', 'spinel': 'sp',
            'plagioclase': 'pl', 'k-feldspar': 'kfs', 'garnet': 'gt', 'nepheline': 'ne',
            'leucite': 'lc', 'biotite': 'bt', 'rhm-oxide': 'ilm', 'apatite': 'ap',
            'whitlockite': 'wht', 'quartz': 'qz', 'tridymite': 'trd', 'fluid': 'fl',
            'hornblende': 'hbl', 'muscovite': 'ms', 'graphite': 'gr', 'calcite': 'cc',
            LIQUID: 'liq'}.get(p, p)


def select_cumulate_rows(emulator: NN_MELTS, bundle: MLBundle, assemblage: Sequence[str], *,
                         exact: bool = True, ignore: Sequence[str] = ('fluid',)) -> np.ndarray:
    """Indices of liquid-bearing bundle rows whose solid assemblage is `assemblage`
    (exactly, or as a superset when `exact=False`), ignoring `ignore` phases."""
    geo = _Geometry(emulator)
    b = bundle.binary_labels > 0.5
    want = np.zeros(len(geo.phases), bool)
    for p in assemblage:
        want[geo.phase_col[p]] = True
    care = np.ones(len(geo.phases), bool)
    care[geo.liq] = False
    for p in ignore:
        if p in geo.phase_col:
            care[geo.phase_col[p]] = False
    has = b[:, geo.liq]
    if exact:
        ok = (b[:, care] == want[care]).all(1)
    else:
        ok = b[:, want].all(1)
    return np.nonzero(has & ok)[0]


# --------------------------------------------------------------------------- #
#  Inversion
# --------------------------------------------------------------------------- #
@dataclass
class InversionConfig:
    # loss weights
    w_assemblage: float = 0.05
    w_composition: float = 1.0
    w_liquid: float = 2.0
    w_bounds: float = 1.0
    # Assemblage hinges. These margins are this INVERTER's convention, not the
    # network's: inference decides presence by g_phi > 0 alone (T0 is used only
    # in training, for the annealed boundary smoothing). Here T0 merely gives each
    # phase a natural scale, so "present" is imposed as g_phi >= margin * T0_phi --
    # a small positive abundance, far below the 5th-percentile training abundance
    # that T0 is.
    present_margin: float = 0.05
    liquid_margin: float = 0.05
    hinge_width: float = 0.1            # hinge(d / width): the penalty reaches 1 at `width` (x T0)
                                        # past the margin, so a tiny margin still bites
    extra_phases: str = 'free'          # 'free': phases outside the cumulate are not constrained
                                        # (their abundance still costs liquid fraction); 'forbid'
    absent_margin: float = 0.05         # only with extra_phases='forbid'
    # composition misfit
    comp_sigma_wt: float = 1.0          # 1-sigma per oxide, wt%
    combine_iron: bool = False          # compare FeO + Fe2O3 as FeO-total
    # which oxides the inversion may claim to constrain
    min_oxide_wt: float = 0.01          # "represented" = >= this wt% in some cumulus phase
    volatiles: Tuple[str, ...] = ('H2O', 'CO2')   # carried in the prior/basis, never optimised
    zero_elements: Tuple[str, ...] = ()  # elements pinned exactly to zero (e.g. ('H',))
    # optimisation
    steps: int = 600
    lr_composition: float = 0.05
    lr_PT: float = 0.05
    lr_final_frac: float = 0.05         # cosine decay to this fraction of the lr
    liquid_warmup_frac: float = 0.0     # ramp w_liquid from 0 over this fraction of steps (the
                                        # prior already starts on a liquidus, so no ramp by default)
    batch_size: int = 8192              # nodes per forward/backward chunk
    # box constraint on (P, T); default: training range
    P_bounds: Optional[Tuple[float, float]] = None
    T_bounds: Optional[Tuple[float, float]] = None
    # prior (see CumulateInverter.initial_nodes)
    prior_T_max: float = 1600.0
    prior_T_step: float = 100.0         # superliquidus: T -= step (no liquid: T += step)
    prior_max_iter: int = 5
    prior_T_offset: float = 5.0         # node starts this far below the T its liquid came from
    prior_dirichlet_alpha: float = 1.0  # cumulus-phase mass proportions ~ Dirichlet(alpha)
    # Accessory phases get their mass fraction of the cumulus mix from U(0, prior_trace_max)
    # instead of the Dirichlet (which can hand them up to 100%); the rest share the remainder.
    prior_trace_phases: Tuple[str, ...] = ('apatite', 'whitlockite', 'muscovite', 'graphite', 'calcite')
    prior_trace_max: float = 0.10
    prior_oxide_noise: Optional[Tuple[float, float]] = (0.7, 1.3)  # each mixed oxide x U(lo, hi), then
                                        # re-closed, before the forward model (None = off)
    volatile_prior: Optional[str] = None  # key of constants.CUMULATE_VOLATILE_PRIORS (default by model)
    volatile_zero_frac: float = 1.0 / 3.0
    mass_balanced_liquid: bool = True   # also report the liquid after NN_MELTS mass balance
    seed: int = 0
    progress: Optional[bool] = None     # tqdm bar over steps; None = only when stderr is a terminal
    log_every: int = 50                 # step lines printed instead when there is no bar


class CumulateInverter:
    """Invert a closed isothermal MELTS `ContinuousModel` for the liquid, pressure
    and temperature a cumulate crystallised from (see module docstring)."""

    def __init__(self, emulator: NN_MELTS, target: CumulateTarget,
                 config: Optional[InversionConfig] = None):
        self.cfg = cfg = config or InversionConfig()
        self.geo = geo = _Geometry(emulator)
        self.target = target
        dev = geo.dev
        if cfg.extra_phases not in ('free', 'forbid'):
            raise ValueError("extra_phases must be 'free' or 'forbid'")

        unknown = [p for p in list(target.phases) + list(target.free_phases)
                   if p not in geo.phase_col and p != 'fluid']
        if unknown:
            raise KeyError(f"Phases {unknown} are not in this model: {geo.phases}")
        self.required = list(target.phases)
        free = set(target.free_phases)
        # phases that are neither cumulus nor liquid: reported, and forbidden only on request
        self.extra = [p for p in geo.phases if p != LIQUID and p not in self.required and p not in free]
        self.forbidden = list(self.extra) if cfg.extra_phases == 'forbid' else []
        self._req_idx = torch.as_tensor([geo.phase_col[p] for p in self.required], device=dev)
        self._forb_idx = torch.as_tensor([geo.phase_col[p] for p in self.forbidden], device=dev,
                                         dtype=torch.long)
        self._extra_idx = torch.as_tensor([geo.phase_col[p] for p in self.extra], device=dev,
                                          dtype=torch.long)
        self.comp_phases = [p for p in self.required if p in geo.chem_cols]
        missing = [p for p in self.comp_phases if p not in target.oxide_wt]
        if missing:
            raise ValueError(f"No target composition for compositionally variable phase(s) {missing}")
        self._tgt = {p: torch.as_tensor(np.asarray(target.oxide_wt[p], np.float32), device=dev)
                     for p in self.comp_phases}
        self._iron = [geo.Oxides.index(o) for o in ('FeO', 'Fe2O3') if o in geo.Oxides]
        self.P_bounds = tuple(cfg.P_bounds or geo.P_bounds)
        self.T_bounds = tuple(cfg.T_bounds or geo.T_bounds)

        # ---- which oxides the cumulate can speak for -----------------------------
        self.phase_oxwt_target = {}
        for p in self.required:
            w = target.oxide_wt.get(p) if p in target.oxide_wt else geo.pure_oxwt(p)
            self.phase_oxwt_target[p] = np.asarray(w, np.float64)
        zero_ox = {geo.Oxides[j] for j in range(geo.O) if geo.Elkeys[geo.el_of_ox[j]] in cfg.zero_elements}
        represented = {geo.Oxides[j] for p, w in self.phase_oxwt_target.items()
                       for j in range(geo.O) if w[j] >= cfg.min_oxide_wt}
        self.constrained_oxides = [o for o in geo.Oxides if o in represented and o not in zero_ox]
        self.volatile_oxides = [o for o in cfg.volatiles
                                if o in geo.Oxides and o not in represented and o not in zero_ox]
        self.basis_oxides = [o for o in geo.Oxides
                             if o in self.constrained_oxides or o in self.volatile_oxides]
        free_el = np.zeros(geo.E, bool)
        for o in self.constrained_oxides:
            free_el[geo.el_of_ox[geo.Oxides.index(o)]] = True
        self._free_mask = torch.as_tensor(free_el, device=dev)          # optimised elements
        basis_el = np.zeros(geo.E, bool)
        for o in self.basis_oxides:
            basis_el[geo.el_of_ox[geo.Oxides.index(o)]] = True
        self._basis_mask = torch.as_tensor(basis_el, device=dev)

    # ---- parameterisation -----------------------------------------------------
    # x = x_fixed + (1 - sum(x_fixed)) * softmax(z over the optimised elements).
    # x_fixed holds, per node, the elements the cumulate cannot constrain (volatiles
    # from the prior; everything outside the basis is 0). They never move: the
    # optimiser only redistributes the remaining mole fraction among the elements
    # that are represented in the cumulate.
    def _decode(self, z, u, xfix):
        Plo, Phi = self.P_bounds
        Tlo, Thi = self.T_bounds
        s = torch.sigmoid(u)
        P = Plo + (Phi - Plo) * s[:, 0]
        T = Tlo + (Thi - Tlo) * s[:, 1]
        zf = z.masked_fill(~self._free_mask, float('-inf'))
        x = xfix + (1.0 - xfix.sum(1, keepdim=True)) * torch.softmax(zf, dim=1)
        return P, T, x

    def _encode(self, P, T, x):
        Plo, Phi = self.P_bounds
        Tlo, Thi = self.T_bounds
        eps = 1e-4
        sp = ((P - Plo) / (Phi - Plo)).clamp(eps, 1 - eps)
        st = ((T - Tlo) / (Thi - Tlo)).clamp(eps, 1 - eps)
        u = torch.stack([torch.logit(sp), torch.logit(st)], dim=1)
        x = x * self._basis_mask
        x = x / x.sum(1, keepdim=True).clamp(min=1e-12)
        xfix = x * (~self._free_mask)
        xf = x * self._free_mask
        z = torch.log(xf.clamp(min=1e-7) / xf.sum(1, keepdim=True).clamp(min=1e-12))
        return z, u, xfix

    # ---- loss -----------------------------------------------------------------
    def _compare(self, pred, tgt):
        if self.cfg.combine_iron and len(self._iron) == 2:
            i, j = self._iron
            fe_p = pred[:, i] + 0.8998 * pred[:, j]
            fe_t = tgt[i] + 0.8998 * tgt[j]
            keep = [k for k in range(pred.shape[1]) if k not in (i, j)]
            pred = torch.cat([pred[:, keep], fe_p[:, None]], 1)
            tgt = torch.cat([tgt[keep], fe_t[None]])
        return pred - tgt

    def evaluate(self, P, T, x, w_liquid: Optional[float] = None, grad: bool = False) -> dict:
        """All per-node loss terms and diagnostics at physical (P [bar], T [C], x
        [closed element moles]). Tensors in, dict of tensors out."""
        cfg, geo = self.cfg, self.geo
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            cm, chem, g = geo.network(P, T, x)
            a = g / geo.T0
            wd = cfg.hinge_width
            L_asm = _hinge((cfg.present_margin - a[:, self._req_idx]) / wd).sum(1) \
                + _hinge((cfg.liquid_margin - a[:, geo.liq]) / wd)
            if len(self.forbidden):
                L_asm = L_asm + _hinge((a[:, self._forb_idx] + cfg.absent_margin) / wd).sum(1)

            if self.comp_phases:
                terms, rmse = [], []
                for p in self.comp_phases:
                    d = self._compare(geo.phase_oxwt(chem, p), self._tgt[p])
                    terms.append(((d / cfg.comp_sigma_wt) ** 2).mean(1))
                    rmse.append((d ** 2).mean(1))
                L_comp = torch.stack(terms, 1).mean(1)
                comp_rmse = torch.stack(rmse, 1).mean(1).sqrt()
            else:
                L_comp = torch.zeros_like(L_asm)
                comp_rmse = torch.zeros_like(L_asm)

            frac = geo.phase_fractions(cm)
            f_liq = frac[:, geo.liq]
            L_liq = 1.0 - f_liq

            bulk_wt = geo.el_to_oxwt(x)
            L_bnd = (F.relu(geo.ox_lo - bulk_wt) ** 2 + F.relu(bulk_wt - geo.ox_hi) ** 2).sum(1)

            wl = cfg.w_liquid if w_liquid is None else w_liquid
            total = (cfg.w_assemblage * L_asm + cfg.w_composition * L_comp
                     + wl * L_liq + cfg.w_bounds * L_bnd)
            present = g > 0
            ok = present[:, geo.liq] & present[:, self._req_idx].all(1)
            if cfg.extra_phases == 'forbid' and len(self.forbidden):
                ok = ok & ~present[:, self._forb_idx].any(1)
            if len(self.extra):
                extra_present = present[:, self._extra_idx]
                extra_frac = (frac[:, self._extra_idx] * extra_present).sum(1)
            else:
                extra_present = torch.zeros(len(P), 0, dtype=torch.bool, device=P.device)
                extra_frac = torch.zeros_like(f_liq)
        return dict(total=total, assemblage=L_asm, composition=L_comp, liquid=L_liq,
                    bounds=L_bnd, f_liquid=f_liq, comp_rmse=comp_rmse, assemblage_ok=ok,
                    extra_present=extra_present, extra_fraction=extra_frac,
                    g=g, chem=chem, bulk_wt=bulk_wt)

    # ---- prior ------------------------------------------------------------------
    def volatile_prior(self) -> Dict[str, Tuple[float, float]]:
        """{oxide: (mean, std)} wt% for the carried volatiles, from
        constants.CUMULATE_VOLATILE_PRIORS (key: config.volatile_prior, else
        'MELTS120' for models with carbon, 'MELTS102' otherwise)."""
        from ..config.constants import CUMULATE_VOLATILE_PRIORS
        key = self.cfg.volatile_prior or ('MELTS120' if 'C' in self.geo.Elkeys else 'MELTS102')
        table = CUMULATE_VOLATILE_PRIORS[key]
        return {o: (float(table[o]['mean']), float(table[o]['std']))
                for o in self.volatile_oxides if o in table}

    def _prior_phase_fractions(self, rng, n: int) -> np.ndarray:
        """Mass fractions of the cumulus phases in each prior mix, (n, k), rows sum
        to 1: accessory phases (`prior_trace_phases`) ~ U(0, prior_trace_max) each,
        the other phases split the remainder ~ Dirichlet(prior_dirichlet_alpha).
        A cumulate made only of accessory phases falls back to a plain Dirichlet."""
        cfg = self.cfg
        trace = np.array([p in set(cfg.prior_trace_phases) for p in self.required])
        k = len(self.required)
        if trace.all() or not trace.any():
            return rng.dirichlet(np.full(k, cfg.prior_dirichlet_alpha), n)
        w = np.zeros((n, k))
        w[:, trace] = rng.uniform(0.0, cfg.prior_trace_max, (n, int(trace.sum())))
        rest = 1.0 - w[:, trace].sum(1, keepdims=True)
        w[:, ~trace] = rest * rng.dirichlet(np.full(int((~trace).sum()), cfg.prior_dirichlet_alpha), n)
        return w

    @torch.no_grad()
    def initial_nodes(self, n_nodes: int) -> Dict[str, np.ndarray]:
        """Cumulate-derived prior: every node starts ON a liquidus, by construction.

        1. Random cumulus-phase mass proportions (Dirichlet; accessory phases --
           apatite, whitlockite, muscovite, graphite, calcite -- only U(0, 10%)
           each, see `_prior_phase_fractions`) mix the target phase
           compositions into a bulk that contains only represented oxides; each
           oxide is then multiplied by an independent U(0.7, 1.3) factor
           (`prior_oxide_noise`) and the bulk re-closed, so priors are not
           confined to the plane spanned by the cumulus phases.
        2. Volatiles (H2O, and CO2 for 1.2.x) are drawn from N(mean, std) of the
           training data (constants.CUMULATE_VOLATILE_PRIORS), clamped to
           [0, training max], and zeroed for a random `volatile_zero_frac` of nodes.
        3. Random P in the training range and T in [Tmin, min(Tmax, prior_T_max)].
        4. Forward model on that bulk. Superliquidus (no solid): T -= prior_T_step;
           no liquid: T += prior_T_step; repeat up to prior_max_iter times.
        5. The predicted LIQUID at (P, T) is the node's starting bulk; the node
           starts at T - prior_T_offset. That liquid is saturated in whatever
           crystallised from the mix, so it sits on its own liquidus. Nodes still
           superliquidus after the iterations keep their (all-liquid) bulk.
           The liquid's volatile contents are then reset to the drawn values
           (other oxides rescaled): a crystal-rich mix would otherwise hand the
           liquid ~1/F times the drawn H2O/CO2, and volatiles are never optimised.
        """
        cfg, geo, dev = self.cfg, self.geo, self.geo.dev
        rng = np.random.default_rng(cfg.seed)
        n = int(n_nodes)
        # 1. cumulus mixture (oxide wt%), represented oxides only
        keep = np.array([o in self.constrained_oxides for o in geo.Oxides])
        comps = np.stack([self.phase_oxwt_target[p] * keep for p in self.required])      # (k, O)
        comps = 100 * comps / comps.sum(1, keepdims=True)
        w = self._prior_phase_fractions(rng, n)                                           # (n, k)
        bulk = w @ comps                                                                  # (n, O)
        if cfg.prior_oxide_noise is not None:   # independent multiplicative noise per oxide
            bulk = bulk * rng.uniform(*cfg.prior_oxide_noise, bulk.shape)
            bulk = 100 * bulk / bulk.sum(1, keepdims=True)
        # 2. volatiles
        vol = np.zeros((n, geo.O))
        hi = geo.ox_hi.cpu().numpy()
        for o, (mu, sd) in self.volatile_prior().items():
            j = geo.Oxides.index(o)
            v = np.clip(rng.normal(mu, sd, n), 0.0, hi[j])
            v[rng.random(n) < cfg.volatile_zero_frac] = 0.0
            vol[:, j] = v
        bulk = bulk * (100.0 - vol.sum(1, keepdims=True)) / 100.0 + vol
        xb = geo.oxwt_to_el(torch.as_tensor(bulk, dtype=torch.float32, device=dev))
        # 3. P, T
        Tlo, Thi = self.T_bounds
        P = torch.as_tensor(rng.uniform(*self.P_bounds, n), dtype=torch.float32, device=dev)
        T = torch.as_tensor(rng.uniform(Tlo, min(Thi, cfg.prior_T_max), n), dtype=torch.float32, device=dev)
        solid = torch.as_tensor(geo.solid_cols(), device=dev)
        # 4. walk each node onto the two-phase field
        n_moves = torch.zeros(n, dtype=torch.long, device=dev)
        for it in range(cfg.prior_max_iter + 1):
            liq = torch.empty(n, dtype=torch.bool, device=dev)
            sol = torch.empty(n, dtype=torch.bool, device=dev)
            for s in range(0, n, cfg.batch_size):
                _, _, g = geo.network(P[s:s + cfg.batch_size], T[s:s + cfg.batch_size], xb[s:s + cfg.batch_size])
                liq[s:s + cfg.batch_size] = g[:, geo.liq] > 0
                sol[s:s + cfg.batch_size] = (g[:, solid] > 0).any(1)
            superliq, noliq = liq & ~sol, ~liq
            if it == cfg.prior_max_iter or not bool((superliq | noliq).any()):
                break
            T = torch.where(superliq, (T - cfg.prior_T_step).clamp(min=Tlo), T)
            T = torch.where(noliq, (T + cfg.prior_T_step).clamp(max=Thi), T)
            n_moves += (superliq | noliq).long()
        # 5. liquid at (P, T) -> starting bulk
        x0 = torch.empty_like(xb)
        lc = torch.as_tensor(geo.chem_cols[LIQUID], device=dev)
        for s in range(0, n, cfg.batch_size):
            _, chem, _ = geo.network(P[s:s + cfg.batch_size], T[s:s + cfg.batch_size], xb[s:s + cfg.batch_size])
            x0[s:s + cfg.batch_size] = chem[:, lc]
        x0 = x0 / x0.sum(1, keepdim=True).clamp(min=1e-12)
        x0 = torch.where((liq & sol)[:, None], x0, xb)       # superliquidus / no liquid: keep the bulk
        # Volatiles: the liquid of a mostly-crystalline mix concentrates H2O/CO2 by
        # ~1/F, far outside the training range, and they are not optimised. The
        # node's (liquid) volatile content is therefore set to the drawn value itself.
        if self.volatile_oxides:
            lw = geo.el_to_oxwt(x0).cpu().numpy()
            jv = [geo.Oxides.index(o) for o in self.volatile_oxides]
            other = np.ones(geo.O, bool)
            other[jv] = False
            lw[:, other] *= (100.0 - vol[:, jv].sum(1, keepdims=True)) / \
                np.clip(lw[:, other].sum(1, keepdims=True), 1e-12, None)
            lw[:, jv] = vol[:, jv]
            x0 = geo.oxwt_to_el(torch.as_tensor(lw, dtype=torch.float32, device=dev))
        status = np.where((liq & sol).cpu().numpy(), 'liquidus',
                          np.where(superliq.cpu().numpy(), 'superliquidus', 'no_liquid'))
        T0 = (T - cfg.prior_T_offset).clamp(Tlo, Thi)
        return {'x': x0.cpu().numpy(), 'P': P.cpu().numpy(), 'T': T0.cpu().numpy(),
                'prior_T': T.cpu().numpy(), 'prior_status': status,
                'prior_moves': n_moves.cpu().numpy(), 'prior_bulk_wt': bulk,
                'prior_phase_fractions': w}

    # ---- main loop --------------------------------------------------------------
    def run(self, n_nodes: int = 2 ** 15, init: Optional[dict] = None,
            verbose: bool = True) -> 'InversionResult':
        """Optimise `n_nodes` independent starts and keep each node's best state.

        By default the starts come from `initial_nodes` (cumulate-derived prior);
        pass `init` (dict with 'x' element moles, 'P' bar, 'T' C) to override.
        """
        cfg, geo, dev = self.cfg, self.geo, self.geo.dev
        torch.manual_seed(cfg.seed)
        t0 = time.time()
        if init is None:
            init = self.initial_nodes(n_nodes)
        N = len(init['P'])
        if verbose:
            st = init.get('prior_status')
            msg = '' if st is None else ' (' + ', '.join(
                f"{k} {np.mean(st == k):.0%}" for k in ('liquidus', 'superliquidus', 'no_liquid')) + ')'
            print(f"[cumulate] {self.target.name}: {N} nodes on {geo.dev}, prior in "
                  f"{time.time() - t0:.1f}s{msg}\n           optimising "
                  f"{'+'.join(self.constrained_oxides)}; carried {'+'.join(self.volatile_oxides) or '-'}; "
                  f"extra phases {cfg.extra_phases}", flush=True)

        P0 = torch.as_tensor(np.asarray(init['P']), dtype=torch.float32, device=dev)
        T0 = torch.as_tensor(np.asarray(init['T']), dtype=torch.float32, device=dev)
        x0 = torch.as_tensor(np.asarray(init['x']), dtype=torch.float32, device=dev)
        z0, u0, xfix = self._encode(P0, T0, x0)
        _, _, x0 = self._decode(z0, u0, xfix)                 # the start as the optimiser sees it
        z = z0.clone().requires_grad_(True)
        u = u0.clone().requires_grad_(True)
        opt = torch.optim.Adam([{'params': [z], 'lr': cfg.lr_composition},
                                {'params': [u], 'lr': cfg.lr_PT}])
        base = [cfg.lr_composition, cfg.lr_PT]

        best = {'loss': torch.full((N,), float('inf'), device=dev),
                'P': P0.clone(), 'T': T0.clone(), 'x': x0.clone(),
                'step': torch.full((N,), -1, dtype=torch.long, device=dev)}
        history = []

        use_bar = verbose and (cfg.progress if cfg.progress is not None else sys.stderr.isatty())
        bar = tqdm(total=cfg.steps + 1, desc=f"[cumulate] {self.target.name}", unit='step',
                   dynamic_ncols=True, mininterval=0.5) if use_bar else None

        with _frozen(geo.model):
            for step in range(cfg.steps + 1):
                frac = step / max(cfg.steps, 1)
                cos = cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * frac))
                for gi, grp in enumerate(opt.param_groups):
                    grp['lr'] = base[gi] * cos
                wl = cfg.w_liquid * min(1.0, frac / cfg.liquid_warmup_frac) if cfg.liquid_warmup_frac > 0 \
                    else cfg.w_liquid
                opt.zero_grad(set_to_none=True)
                tot_sum, ok_sum, fl_sum = 0.0, 0, 0.0
                term_sum = {'assemblage': 0.0, 'composition': 0.0, 'liquid': 0.0, 'bounds': 0.0}
                cur = torch.empty(N, device=dev)
                for s in range(0, N, cfg.batch_size):
                    e = min(s + cfg.batch_size, N)
                    P, T, x = self._decode(z[s:e], u[s:e], xfix[s:e])
                    ev = self.evaluate(P, T, x, w_liquid=wl, grad=True)
                    if step < cfg.steps:
                        ev['total'].sum().backward()
                    # selection uses the full-weight loss so every step is comparable
                    with torch.no_grad():
                        sel = (cfg.w_assemblage * ev['assemblage'] + cfg.w_composition * ev['composition']
                               + cfg.w_liquid * ev['liquid'] + cfg.w_bounds * ev['bounds'])
                        better = sel < best['loss'][s:e]
                        idx = torch.nonzero(better).squeeze(1) + s
                        best['loss'][idx] = sel[better]
                        best['P'][idx], best['T'][idx], best['x'][idx] = \
                            P.detach()[better], T.detach()[better], x.detach()[better]
                        best['step'][idx] = step
                        tot_sum += float(sel.sum())
                        ok_sum += int(ev['assemblage_ok'].sum())
                        fl_sum += float(ev['f_liquid'].sum())
                        cur[s:e] = sel
                        term_sum['assemblage'] += cfg.w_assemblage * float(ev['assemblage'].sum())
                        term_sum['composition'] += cfg.w_composition * float(ev['composition'].sum())
                        term_sum['liquid'] += cfg.w_liquid * float(ev['liquid'].sum())
                        term_sum['bounds'] += cfg.w_bounds * float(ev['bounds'].sum())
                q = torch.quantile(cur.float(), torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()
                history.append({'step': step, 'mean_loss': tot_sum / N, 'sum_loss': tot_sum,
                                **{f'sum_{k}': v for k, v in term_sum.items()},
                                'loss_p10': q[0], 'loss_p50': q[1], 'loss_p90': q[2],
                                'frac_assemblage_ok': ok_sum / N, 'mean_f_liquid': fl_sum / N,
                                'best_median': float(best['loss'].median()),
                                'best_min': float(best['loss'].min())})
                h = history[-1]
                if bar is not None:
                    bar.set_postfix(loss=f"{h['mean_loss']:.3g}", ok=f"{h['frac_assemblage_ok']:.0%}",
                                    f_liq=f"{h['mean_f_liquid']:.3f}", best=f"{h['best_median']:.3g}",
                                    refresh=False)
                    bar.update(1)
                elif verbose and (step % cfg.log_every == 0 or step == cfg.steps):
                    print(f"  step {step:4d}  mean loss {h['mean_loss']:.4g}  "
                          f"assemblage ok {h['frac_assemblage_ok']:.1%}  mean f_liq {h['mean_f_liquid']:.3f}  "
                          f"best median {h['best_median']:.4g}  ({time.time() - t0:.0f}s)", flush=True)
                if step < cfg.steps:
                    opt.step()
        if bar is not None:
            bar.close()

        # ---- final diagnostics at each node's best state -------------------------
        cols = {}
        lc = geo.comp_cols[LIQUID]
        for s in range(0, N, cfg.batch_size):
            e = min(s + cfg.batch_size, N)
            bP, bT, bx = best['P'][s:e], best['T'][s:e], best['x'][s:e]
            ev = self.evaluate(bP, bT, bx)
            for k in ('total', 'assemblage', 'composition', 'liquid', 'bounds', 'f_liquid',
                      'comp_rmse', 'assemblage_ok', 'extra_fraction', 'extra_present'):
                cols.setdefault(k, []).append(ev[k].cpu().numpy())
            cols.setdefault('liq_wt', []).append(geo.phase_oxwt(ev['chem'], LIQUID).cpu().numpy())
            cols.setdefault('bulk_wt', []).append(ev['bulk_wt'].cpu().numpy())
            per = [(self._compare(geo.phase_oxwt(ev['chem'], p), self._tgt[p]) ** 2).mean(1).sqrt().cpu().numpy()
                   for p in self.comp_phases]
            cols.setdefault('per_phase', []).append(np.stack(per, 1) if per else np.zeros((e - s, 0)))
            if cfg.mass_balanced_liquid:
                feats = torch.zeros(e - s, geo.n_feat + geo.E, device=dev)
                feats[:, geo.P_idx], feats[:, geo.T_idx], feats[:, geo.n_feat:] = bP, bT, bx
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mb = geo.em.forwardMB(feats, Normalize=True, WtPercent=False,
                                          outputs=['component_moles'])['component_moles']
                liq_el = mb[:, lc].to(dev)
                liq_el = liq_el / liq_el.sum(1, keepdim=True).clamp(min=1e-12)
                cols.setdefault('liqMB_wt', []).append(geo.el_to_oxwt(liq_el).cpu().numpy())
        cat = {k: np.concatenate(v) for k, v in cols.items()}
        init_wt = geo.el_to_oxwt(x0).cpu().numpy()

        df = pd.DataFrame({
            'node': np.arange(N),
            'P_bar': best['P'].cpu().numpy(), 'T_C': best['T'].cpu().numpy(),
            'loss': cat['total'], 'loss_assemblage': cat['assemblage'],
            'loss_composition': cat['composition'], 'loss_liquid': cat['liquid'],
            'loss_bounds': cat['bounds'], 'f_liquid': cat['f_liquid'],
            'comp_rmse_wt': cat['comp_rmse'], 'assemblage_ok': cat['assemblage_ok'],
            'extra_fraction': cat['extra_fraction'],
            'n_extra_phases': cat['extra_present'].sum(1),
            'extra_phases': ['+'.join(_short(p) for p, on in zip(self.extra, row) if on)
                             for row in cat['extra_present']],
            'best_step': best['step'].cpu().numpy(),
            'init_P_bar': np.asarray(init['P']), 'init_T_C': np.asarray(init['T']),
        })
        for k in ('prior_T', 'prior_status', 'prior_moves'):
            if k in init:
                df[k] = init[k]
        for j, p in enumerate(self.comp_phases):
            df[f'rmse_{_short(p)}'] = cat['per_phase'][:, j]
        for pre, key in (('liq', 'liq_wt'), ('liqMB', 'liqMB_wt'), ('bulk', 'bulk_wt')):
            if key in cat:
                for j, ox in enumerate(geo.Oxides):
                    df[f'{pre}_{ox}'] = cat[key][:, j]
        for j, ox in enumerate(geo.Oxides):
            df[f'init_{ox}'] = init_wt[:, j]
        if verbose:
            print(f"[cumulate] done in {time.time() - t0:.1f}s; assemblage satisfied at best state: "
                  f"{df['assemblage_ok'].mean():.1%}; median comp RMSE {df['comp_rmse_wt'].median():.2f} wt%; "
                  f"median f_liq {df['f_liquid'].median():.3f}; extra phases in {np.mean(df['n_extra_phases'] > 0):.1%}",
                  flush=True)
        return InversionResult(nodes=df, target=self.target, config=self.cfg,
                               history=pd.DataFrame(history), oxides=list(geo.Oxides),
                               basis_oxides=list(self.basis_oxides),
                               constrained_oxides=list(self.constrained_oxides),
                               model_source=str((getattr(geo.em, 'training_bounds', None) or {})
                                                .get('source', '')))


# --------------------------------------------------------------------------- #
#  Results
# --------------------------------------------------------------------------- #
@dataclass
class InversionResult:
    """Best state per node plus summaries. `nodes` has one row per node:
    P_bar, T_C, loss terms, f_liquid, comp_rmse_wt, assemblage_ok, extra-phase
    diagnostics, the liquid in equilibrium with the assemblage at the node's best
    state (`liq_<oxide>`: the network's liquid chemistry head, wt%;
    `liqMB_<oxide>`: the same after NN_MELTS mass balance), the optimised bulk
    (`bulk_<oxide>`) and the start (`init_*`).

    `basis_oxides` = oxides represented in the cumulate + carried volatiles. Every
    liquid and bulk here is zero outside the basis, so its wt% are already on that
    basis; `truth_vector` renormalises the true liquid onto the same basis.
    `constrained_oxides` (the represented ones) are the only ones the inversion
    claims to constrain."""
    nodes: pd.DataFrame
    target: CumulateTarget
    config: InversionConfig
    history: pd.DataFrame
    oxides: List[str]
    basis_oxides: Optional[List[str]] = None
    constrained_oxides: Optional[List[str]] = None
    model_source: str = ''

    def __post_init__(self):
        # derived index for evolved liquids: molar (Na2O + K2O + 2 CaO) / (2 Al2O3)
        for pre in ('liq', 'liqMB', 'bulk', 'init'):
            if f'{pre}_Al2O3' in self.nodes and f'{pre}_NKC2A' not in self.nodes:
                self.nodes[f'{pre}_NKC2A'] = _nkc2a({o: self.nodes[f'{pre}_{o}'] for o in self.oxides
                                                      if f'{pre}_{o}' in self.nodes})

    def composition_axes(self, mgo_threshold: float = 4.0) -> Tuple[str, str]:
        """('SiO2', 'MgO') normally; ('SiO2', 'NKC2A') -- molar (Na2O+K2O+2CaO)/(2Al2O3) --
        when the liquid has < `mgo_threshold` wt% MgO (the true liquid on the basis if
        known, else the accepted median) or MgO is not constrained."""
        basis = self.basis_oxides or self.oxides
        tl = self.truth_liquid()
        if tl is not None:
            mgo = tl.get('MgO', 0.0)
        else:
            acc = self.accepted()
            mgo = float(acc['liq_MgO'].median()) if len(acc) and 'liq_MgO' in acc else 0.0
        if 'MgO' in basis and mgo >= mgo_threshold:
            return 'SiO2', 'MgO'
        if 'Al2O3' in basis:
            return 'SiO2', 'NKC2A'
        other = [o for o in (self.constrained_oxides or basis) if o != 'SiO2']
        return 'SiO2', (other[0] if other else 'MgO')

    # ---- selection -------------------------------------------------------------
    def accepted(self, *, require_assemblage: bool = True, max_comp_rmse: Optional[float] = None,
                 min_f_liquid: Optional[float] = None, loss_quantile: Optional[float] = None,
                 max_loss: Optional[float] = None) -> pd.DataFrame:
        """Nodes whose best state satisfies the imposed assemblage (default) and any of:
        composition RMSE <= `max_comp_rmse` (wt%), liquid fraction >= `min_f_liquid`,
        loss <= `max_loss`, loss within the lowest `loss_quantile` of those remaining."""
        d = self.nodes
        m = np.ones(len(d), bool)
        if require_assemblage:
            m &= d['assemblage_ok'].values
        if min_f_liquid is not None:
            m &= d['f_liquid'].values >= min_f_liquid
        if max_loss is not None:
            m &= d['loss'].values <= max_loss
        if max_comp_rmse is not None:
            m &= d['comp_rmse_wt'].values <= max_comp_rmse
        if loss_quantile is not None and m.any():
            m &= d['loss'].values <= np.quantile(d['loss'].values[m], loss_quantile)
        return d[m]

    def weights(self, subset: Optional[pd.DataFrame] = None, temperature: Optional[float] = None):
        """Uniform weights, or Boltzmann weights exp(-(loss - min)/temperature)."""
        d = self.nodes if subset is None else subset
        if temperature is None:
            return np.full(len(d), 1.0 / max(len(d), 1))
        w = np.exp(-(d['loss'].values - d['loss'].values.min()) / temperature)
        return w / w.sum()

    # ---- statistics ------------------------------------------------------------
    def default_columns(self, which: str = 'liq') -> List[str]:
        return ['P_bar', 'T_C'] + [f'{which}_{o}' for o in (self.basis_oxides or self.oxides)]

    def truth_liquid(self, source: str = 'truth') -> Optional[Dict[str, float]]:
        """True liquid (oxide wt%) renormalised to 100 over `basis_oxides`.
        `source='emulator'`: instead, the liquid nGibbs predicts at the row's true
        (P, T, bulk) -- the emulator's own baseline -- on the same basis."""
        tr = self.target.truth
        if not tr:
            return None
        if source == 'emulator':
            emu = tr.get('emulator')
            if not emu:
                return None
            w = dict(zip(self.oxides, np.asarray(emu['liquid_oxide_wt'], float)))
        else:
            w = dict(zip(self.oxides, np.asarray(tr['liquid_oxide_wt'], float)))
        basis = self.basis_oxides or self.oxides
        tot = sum(w[o] for o in basis)
        return {o: (100.0 * w[o] / tot if o in basis else 0.0) for o in self.oxides}

    def covariance(self, columns: Optional[Sequence[str]] = None, subset: Optional[pd.DataFrame] = None,
                   weights=None, correlation: bool = False) -> pd.DataFrame:
        d = self.accepted() if subset is None else subset
        cols = list(columns) if columns is not None else self.default_columns()
        X = d[cols].values.astype(np.float64)
        w = np.full(len(X), 1.0 / max(len(X), 1)) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        mu = w @ X
        Xc = X - mu
        C = (Xc * w[:, None]).T @ Xc / max(1.0 - (w ** 2).sum(), 1e-12)
        if correlation:
            s = np.sqrt(np.clip(np.diag(C), 1e-300, None))
            C = C / np.outer(s, s)
        return pd.DataFrame(C, index=cols, columns=cols)

    def summary(self, subset: Optional[pd.DataFrame] = None, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Weighted mean / std / 5-50-95% of the accepted solutions (and truth, if known)."""
        d = self.accepted() if subset is None else subset
        cols = list(columns) if columns is not None else self.default_columns()
        out = pd.DataFrame({'mean': d[cols].mean(), 'std': d[cols].std(),
                            'p05': d[cols].quantile(0.05), 'p50': d[cols].median(),
                            'p95': d[cols].quantile(0.95)})
        t = self.truth_vector(cols)
        if t is not None:
            out['truth'] = t
        e = self.truth_vector(cols, source='emulator')
        if e is not None:
            out['nGibbs_at_truth'] = e
        best = self.nodes.loc[self.nodes['loss'].idxmin(), cols]
        out['best_node'] = best.values
        return out

    def truth_vector(self, columns, source: str = 'truth'):
        """Reference values for `columns`: the truth, or (`source='emulator'`) the
        nGibbs forward model at the true (P, T, bulk) -- same P and T, its own liquid."""
        tr = self.target.truth
        if not tr or (source == 'emulator' and not tr.get('emulator')):
            return None
        vals = []
        for c in columns:
            if c in ('P_bar', 'init_P_bar'):
                vals.append(tr['P_bar'])
            elif c in ('T_C', 'init_T_C'):
                vals.append(tr['T_C'])
            elif c.startswith(('liq_', 'liqMB_', 'bulk_', 'init_')):
                ox = c.split('_', 1)[1]
                tl = self.truth_liquid(source)
                vals.append(float(_nkc2a(tl)) if ox == 'NKC2A' else tl[ox])
            else:
                vals.append(np.nan)
        return np.array(vals)

    # ---- densities ------------------------------------------------------------
    def density2d(self, x: str = 'liq_SiO2', y: str = 'liq_MgO', subset: Optional[pd.DataFrame] = None,
                  weights=None, gridsize: int = 120, pad: float = 0.15, bw_scale: float = 1.0):
        """Weighted 2-D Gaussian KDE (Scott bandwidth, full covariance) of two columns.
        Returns (X, Y, Z) grids with Z a probability density (integrates to 1)."""
        d = self.accepted() if subset is None else subset
        pts = d[[x, y]].values.astype(np.float64)
        w = np.full(len(pts), 1.0 / max(len(pts), 1)) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        return _kde2d(pts, w, gridsize=gridsize, pad=pad, bw_scale=bw_scale)

    def plot_density(self, x: Optional[str] = None, y: Optional[str] = None, subset: Optional[pd.DataFrame] = None,
                     weights=None, ax_pt: bool = True, path: Optional[Union[str, Path]] = None,
                     title: Optional[str] = None, pt: Tuple[str, str] = ('T_C', 'P_bar'),
                     mark_best: bool = True):
        """KDE of the accepted solutions in (x, y) with marginals, plus a P-T panel.
        Truth (if known) is starred; the best node is a cross. `pt` names the
        (temperature, pressure) columns, e.g. ('init_T_C', 'init_P_bar') for starts."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        if x is None or y is None:
            ax_x, ax_y = self.composition_axes()
            x, y = x or f'liq_{ax_x}', y or f'liq_{ax_y}'
        d = self.accepted() if subset is None else subset
        if len(d) < 3:
            raise ValueError(f"Only {len(d)} accepted nodes; relax the acceptance criteria.")
        w = np.full(len(d), 1.0 / len(d)) if weights is None else np.asarray(weights, float) / np.sum(weights)
        X, Y, Z = self.density2d(x, y, subset=d, weights=w)
        tr = self.truth_vector([x, y, 'P_bar', 'T_C'])
        em = self.truth_vector([x, y, 'P_bar', 'T_C'], source='emulator')
        best = self.nodes.loc[self.nodes['loss'].idxmin()]
        tcol, pcol = pt

        fig = plt.figure(figsize=(11.5 if ax_pt else 6.5, 5.6))
        gs = GridSpec(2, 3 if ax_pt else 2, width_ratios=[4, 1, 4.2][:3 if ax_pt else 2],
                      height_ratios=[1, 4], wspace=0.08, hspace=0.08, figure=fig)
        ax = fig.add_subplot(gs[1, 0])
        axx = fig.add_subplot(gs[0, 0], sharex=ax)
        axy = fig.add_subplot(gs[1, 1], sharey=ax)
        ax.contourf(X, Y, Z, levels=14, cmap='Blues')
        ax.contour(X, Y, Z, levels=_hdr_levels(Z, [0.95, 0.68]), colors=['#1f3b73', '#0b1a33'],
                   linewidths=[0.8, 1.2])
        ax.scatter(d[x], d[y], s=2, c='k', alpha=min(0.5, 200 / len(d)), lw=0)
        if mark_best:
            ax.plot(best[x], best[y], 'x', color='#d95f02', ms=9, mew=2, label='best node')
        if tr is not None:
            ax.plot(tr[0], tr[1], '*', color='#e7298a', ms=15, mec='k', label='truth (MELTS)')
        if em is not None:
            ax.plot(em[0], em[1], **_EMU_STYLE)
        ax.set_xlabel(_label(x)); ax.set_ylabel(_label(y)); ax.legend(loc='best', fontsize=8)
        _marg(axx, d[x].values, w, orient='x', truth=None if tr is None else tr[0])
        _marg(axy, d[y].values, w, orient='y', truth=None if tr is None else tr[1])
        axx.tick_params(labelbottom=False); axy.tick_params(labelleft=False)
        axx.set_yticks([]); axy.set_xticks([])
        for a in (axx, axy):
            for s in ('top', 'right'):
                a.spines[s].set_visible(False)
        if ax_pt:
            ap = fig.add_subplot(gs[:, 2])
            Xp, Yp, Zp = _kde2d(d[[tcol, pcol]].values.astype(float) / np.array([1, 1000.0]), w)
            ap.contourf(Xp, Yp, Zp, levels=14, cmap='Oranges')
            ap.contour(Xp, Yp, Zp, levels=_hdr_levels(Zp, [0.95, 0.68]), colors=['#7f2704', '#3d1302'],
                       linewidths=[0.8, 1.2])
            ap.scatter(d[tcol], d[pcol] / 1000, s=2, c='k', alpha=min(0.5, 200 / len(d)), lw=0)
            if mark_best:
                ap.plot(best['T_C'], best['P_bar'] / 1000, 'x', color='#d95f02', ms=9, mew=2)
            if tr is not None:
                ap.plot(tr[3], tr[2] / 1000, '*', color='#e7298a', ms=15, mec='k')
            if em is not None:
                ap.plot(em[3], em[2] / 1000, **{**_EMU_STYLE, 'label': None})
            ap.set_xlabel('T (°C)'); ap.set_ylabel('P (kbar)'); ap.yaxis.set_label_position('right')
            ap.yaxis.tick_right()
        fig.suptitle(title or f"{self.target.name}  —  {len(d)} accepted of {len(self.nodes)} nodes",
                     fontsize=11)
        if path is not None:
            fig.savefig(path, dpi=150, bbox_inches='tight')
        return fig

    def plot_prior(self, x: Optional[str] = None, y: Optional[str] = None, which: str = 'all',
                   path: Optional[Union[str, Path]] = None, **accept_kw):
        """The same figure as `plot_density`, for the STARTING states (`init_*`,
        i.e. the prior liquids and their P, T). `which='all'` shows every node's
        start (the prior itself); `which='accepted'` shows where the accepted
        nodes started (pass `accepted()` keywords, e.g. loss_quantile=0.25)."""
        ax_x, ax_y = self.composition_axes()
        x, y = x or ax_x, y or ax_y
        d = self.nodes if which == 'all' else self.accepted(**accept_kw)
        ttl = (f"{self.target.name}  —  PRIOR: starts of all {len(d)} nodes" if which == 'all' else
               f"{self.target.name}  —  starts of the {len(d)} accepted nodes")
        return self.plot_density(f'init_{x}', f'init_{y}', subset=d, pt=('init_T_C', 'init_P_bar'),
                                 mark_best=False, title=ttl, path=path)

    def drift(self, subset: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Per-node change from start to best state: dP (bar), dT (C), and the RMS
        change of the liquid over the constrained oxides (start bulk = prior liquid
        vs final liquid), both renormalised onto the basis."""
        d = self.nodes if subset is None else subset
        con = self.constrained_oxides or self.oxides
        basis = self.basis_oxides or self.oxides
        a = d[[f'init_{o}' for o in basis]].values
        b = d[[f'liq_{o}' for o in basis]].values
        a = 100 * a / a.sum(1, keepdims=True)
        b = 100 * b / b.sum(1, keepdims=True)
        j = [basis.index(o) for o in con]
        out = pd.DataFrame({'dP_bar': d['P_bar'].values - d['init_P_bar'].values,
                            'dT_C': d['T_C'].values - d['init_T_C'].values,
                            'dliq_rmse_wt': np.sqrt(((b[:, j] - a[:, j]) ** 2).mean(1))}, index=d.index)
        for o in ('SiO2', 'MgO'):
            if o in basis:
                out[f'd{o}'] = b[:, basis.index(o)] - a[:, basis.index(o)]
        return out

    def plot_drift(self, x: str = 'SiO2', y: str = 'MgO', path: Optional[Union[str, Path]] = None,
                   n_lines: int = 400, seed: int = 0, **accept_kw):
        """Start -> best state for the accepted nodes: prior density (grey) and
        accepted final density (blue) with start->end segments for a random
        subsample, in (x, y) and in P-T, plus histograms of the drift of every
        node (grey) and of the accepted ones (blue)."""
        import matplotlib.pyplot as plt

        acc = self.accepted(**accept_kw)
        allv = self.nodes
        tr = self.truth_vector([f'liq_{x}', f'liq_{y}', 'P_bar', 'T_C'])
        rng = np.random.default_rng(seed)
        sub = acc.iloc[rng.permutation(len(acc))[:n_lines]]
        fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))

        def panel(ax, xs, ys, scale=(1.0, 1.0), labels=('', '')):
            for dd, cmap, lab in ((allv, 'Greys', 'prior (all starts)'), (acc, 'Blues', 'accepted (final)')):
                cols = xs if lab.startswith('prior') else ys
                pts = dd[list(cols)].values.astype(float) / np.array(scale)
                X, Y, Z = _kde2d(pts, gridsize=100)
                ax.contour(X, Y, Z, levels=_hdr_levels(Z, [0.95, 0.68, 0.38]), cmap=cmap, linewidths=1.2)
                ax.plot([], [], color='0.4' if cmap == 'Greys' else '#2171b5', label=lab)
            a = sub[list(xs)].values / np.array(scale)
            b = sub[list(ys)].values / np.array(scale)
            for (x0, y0), (x1, y1) in zip(a, b):
                ax.plot([x0, x1], [y0, y1], color='#fd8d3c', lw=0.4, alpha=0.35)
            ax.scatter(a[:, 0], a[:, 1], s=3, c='0.3', lw=0, label='start')
            ax.scatter(b[:, 0], b[:, 1], s=3, c='#08519c', lw=0, label='end')
            ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])

        panel(axs[0, 0], (f'init_{x}', f'init_{y}'), (f'liq_{x}', f'liq_{y}'),
              labels=(f'liquid {x} (wt%)', f'liquid {y} (wt%)'))
        panel(axs[0, 1], ('init_T_C', 'init_P_bar'), ('T_C', 'P_bar'), scale=(1.0, 1000.0),
              labels=('T (°C)', 'P (kbar)'))
        if tr is not None:
            axs[0, 0].plot(tr[0], tr[1], '*', color='#e7298a', ms=15, mec='k', label='truth')
            axs[0, 1].plot(tr[3], tr[2] / 1000, '*', color='#e7298a', ms=15, mec='k')
        axs[0, 0].legend(fontsize=8, loc='best')

        # where did accepted nodes START, relative to the prior?
        ax = axs[0, 2]
        for dd, c, lab in ((allv, '0.6', 'all starts'), (acc, '#2171b5', 'accepted starts')):
            ax.hist(dd['init_T_C'], bins=50, density=True, histtype='step', color=c, lw=1.5, label=lab)
        ax.set_xlabel('start T (°C)'); ax.set_yticks([]); ax.legend(fontsize=8)
        if tr is not None:
            ax.axvline(tr[3], color='#e7298a', lw=1.5)

        da, dc = self.drift(allv), self.drift(acc)
        for ax, col, lab in ((axs[1, 0], 'dliq_rmse_wt', 'liquid change, RMS over constrained oxides (wt%)'),
                             (axs[1, 1], 'dT_C', 'T change (°C)'), (axs[1, 2], 'dP_bar', 'P change (bar)')):
            lo, hi = np.quantile(da[col], [0.005, 0.995])
            bins = np.linspace(lo, hi, 60)
            ax.hist(da[col], bins=bins, density=True, histtype='step', color='0.5', lw=1.5, label='all nodes')
            ax.hist(dc[col], bins=bins, density=True, histtype='stepfilled', color='#6baed6', alpha=0.7,
                    label='accepted')
            ax.axvline(0, color='k', lw=0.6)
            ax.set_xlabel(lab); ax.set_yticks([])
            ax.text(0.98, 0.95, f"accepted median {np.median(dc[col]):.3g}\nall median {np.median(da[col]):.3g}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=8)
        axs[1, 0].legend(fontsize=8, loc='center right')
        fig.suptitle(f"{self.target.name}  —  start → best state ({len(acc)} accepted of {len(allv)}; "
                     f"segments for {len(sub)})", fontsize=11)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150, bbox_inches='tight')
        return fig

    def plot_prior_vs_final(self, x: Optional[str] = None, y: Optional[str] = None,
                            path: Optional[Union[str, Path]] = None, **accept_kw):
        """Prior (every node's start: forward-modelled liquid, P, T -- no inversion)
        above the accepted final states, on identical axes, with the truth (star)
        and the best node (cross). Each panel reports the median absolute error of
        its cloud against the truth, so the value added by the inversion over the
        prior alone can be read off directly."""
        import matplotlib.pyplot as plt

        ax_x, ax_y = self.composition_axes()
        x, y = x or ax_x, y or ax_y
        acc = self.accepted(**accept_kw)
        best = self.nodes.loc[self.nodes['loss'].idxmin()]
        rows = (('prior: all starts', self.nodes, 'init', ('init_T_C', 'init_P_bar'), 'Greys'),
                (f'final: {len(acc)} accepted', acc, 'liq', ('T_C', 'P_bar'), 'Blues'))
        tv = self.truth_vector([f'liq_{x}', f'liq_{y}', 'P_bar', 'T_C'])
        ev = self.truth_vector([f'liq_{x}', f'liq_{y}', 'P_bar', 'T_C'], source='emulator')

        def lims(cols_a, cols_b, scale=1.0):
            v = np.concatenate([self.nodes[cols_a].values, acc[cols_b].values]) / scale
            lo, hi = np.nanquantile(v, [0.005, 0.995])
            pad = 0.08 * (hi - lo if hi > lo else 1.0)
            return lo - pad, hi + pad

        xl = lims(f'init_{x}', f'liq_{x}'); yl = lims(f'init_{y}', f'liq_{y}')
        Tl = lims('init_T_C', 'T_C'); Pl = lims('init_P_bar', 'P_bar', 1000.0)
        for ref in (tv, ev):   # always show the truth and the emulator baseline
            if ref is not None:
                xl = (min(xl[0], ref[0]), max(xl[1], ref[0])); yl = (min(yl[0], ref[1]), max(yl[1], ref[1]))
                Tl = (min(Tl[0], ref[3]), max(Tl[1], ref[3]))
                Pl = (min(Pl[0], ref[2] / 1000), max(Pl[1], ref[2] / 1000))

        fig, axs = plt.subplots(2, 2, figsize=(12, 10.5))
        for r, (name, d, pre, (tc, pc), cmap) in enumerate(rows):
            for c, (cx, cy, sc, xlim, ylim, tx, ty) in enumerate((
                    (f'{pre}_{x}', f'{pre}_{y}', 1.0, xl, yl, 0, 1),
                    (tc, pc, 1000.0, Tl, Pl, 3, 2))):
                ax = axs[r, c]
                pts = d[[cx, cy]].values.astype(float) / np.array([1.0, sc])
                if len(pts) > 3:
                    X, Y, Z = _kde2d(pts, gridsize=110, bounds=(xlim, ylim))
                    ax.contourf(X, Y, Z, levels=12, cmap=cmap)
                    ax.contour(X, Y, Z, levels=_hdr_levels(Z, [0.95, 0.68]), colors='k',
                               linewidths=[0.6, 1.0])
                ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c='k', alpha=min(0.4, 300 / max(len(pts), 1)), lw=0)
                if r == 1:
                    ax.plot(best[cx], best[cy] / sc, 'x', color='#d95f02', ms=10, mew=2.2, label='best node')
                txt = []
                if ev is not None:
                    ax.plot(ev[tx], ev[ty] / sc, **_EMU_STYLE)
                if tv is not None:
                    ax.plot(tv[tx], tv[ty] / sc, '*', color='#e7298a', ms=16, mec='k', label='truth (MELTS)')
                    ex = np.median(np.abs(d[cx].values - tv[tx]))
                    ey = np.median(np.abs(d[cy].values - tv[ty])) / sc
                    fx, fy = ('{:.2f}', '{:.3f}' if y == 'NKC2A' else '{:.2f}') if c == 0 else ('{:.0f}', '{:.2f}')
                    txt = [f"median |error|: {fx.format(ex)}, {fy.format(ey)}"]
                ax.set_xlim(*xlim); ax.set_ylim(*ylim)
                if c == 0:
                    ax.set_xlabel(_label(f'liq_{x}').replace('liquid', 'liquid' if r else 'start liquid'))
                    ax.set_ylabel(_label(f'liq_{y}').replace('liquid', 'liquid' if r else 'start liquid'))
                else:
                    ax.set_xlabel('T (°C)'); ax.set_ylabel('P (kbar)')
                ax.set_title(name + ('' if not txt else '   ·   ' + txt[0]), fontsize=10)
                if c == 0:
                    ax.legend(fontsize=8, loc='best')
        fig.suptitle(f"{self.target.name}  —  prior (no inversion) vs inversion", fontsize=12)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150, bbox_inches='tight')
        return fig

    def plot_convergence(self, path: Optional[Union[str, Path]] = None):
        """Loss against iteration: summed over all nodes (with its terms), per-node
        quantiles and the running median of each node's best, and the fraction of
        nodes in the imposed assemblage / mean liquid fraction."""
        import matplotlib.pyplot as plt

        h = self.history
        if h is None or len(h) == 0:
            raise ValueError("No optimisation history stored with this result.")
        fig, axs = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
        ax = axs[0]
        ax.plot(h['step'], h['sum_loss'], color='k', lw=2, label='total')
        for k, c in (('assemblage', '#1b9e77'), ('composition', '#d95f02'), ('liquid', '#7570b3'),
                     ('bounds', '#e7298a')):
            if f'sum_{k}' in h and (h[f'sum_{k}'] > 0).any():
                ax.plot(h['step'], h[f'sum_{k}'], lw=1.2, color=c, label=k)
        ax.set_yscale('log'); ax.set_ylabel(f'loss summed over {len(self.nodes)} nodes')
        ax.legend(fontsize=8, ncol=5, loc='upper right')
        ax = axs[1]
        ax.fill_between(h['step'], h['loss_p10'], h['loss_p90'], color='#9ecae1', alpha=0.6,
                        label='current loss, 10-90%')
        ax.plot(h['step'], h['loss_p50'], color='#2171b5', lw=1.5, label='current loss, median')
        ax.plot(h['step'], h['best_median'], color='k', lw=1.5, ls='--', label="median of nodes' best")
        ax.plot(h['step'], h['best_min'], color='#d95f02', lw=1.2, label='best node')
        ax.set_yscale('log'); ax.set_ylabel('per-node loss'); ax.legend(fontsize=8)
        ax = axs[2]
        ax.plot(h['step'], h['frac_assemblage_ok'], color='#1b9e77', lw=1.5, label='cumulus + liquid present')
        ax.plot(h['step'], h['mean_f_liquid'], color='#7570b3', lw=1.5, label='mean liquid fraction')
        ax.set_ylim(0, 1.02); ax.set_xlabel('iteration'); ax.legend(fontsize=8, loc='lower right')
        fig.suptitle(f"{self.target.name}  —  convergence", fontsize=11)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150, bbox_inches='tight')
        return fig

    def to_parquet_or_csv(self, path: Union[str, Path]):
        path = Path(path)
        if path.suffix == '.parquet':
            self.nodes.to_parquet(path)
        else:
            self.nodes.to_csv(path, index=False)
        return path


# nGibbs forward model at the bundle row's true (P, T, bulk): hollow so the truth star
# beneath it stays visible (in P-T the two coincide by construction).
_EMU_STYLE = dict(marker='D', ls='none', ms=13, mfc='none', mec='#1b9e77', mew=2.2,
                  label='nGibbs at true P, T, bulk')


def _nkc2a(wt):
    """Molar (Na2O + K2O + 2 CaO) / (2 Al2O3) from a {oxide: wt%} mapping (arrays ok)."""
    from ..config.constants import OXIDE_MOLAR_MASSES as MM
    mol = {o: wt[o] / MM[o] for o in ('Na2O', 'K2O', 'CaO', 'Al2O3') if o in wt}
    num = mol.get('Na2O', 0.0) + mol.get('K2O', 0.0) + 2.0 * mol.get('CaO', 0.0)
    den = 2.0 * mol.get('Al2O3', 0.0)
    return num / np.maximum(den, 1e-12)


def _label(c):
    if c.endswith('NKC2A'):
        pre = c.split('_', 1)[0]
        who = {'liq': 'liquid', 'liqMB': 'liquid (MB)', 'bulk': 'bulk', 'init': 'start liquid'}.get(pre, pre)
        return f"{who} molar (Na$_2$O+K$_2$O+2CaO)/(2Al$_2$O$_3$)"
    if c in ('P_bar',):
        return 'P (bar)'
    if c == 'T_C':
        return 'T (°C)'
    if '_' in c:
        pre, ox = c.split('_', 1)
        if c == 'init_T_C':
            return 'start T (°C)'
        if c == 'init_P_bar':
            return 'start P (bar)'
        return f"{ {'liq': 'liquid', 'liqMB': 'liquid (MB)', 'bulk': 'bulk', 'init': 'start liquid'}.get(pre, pre)} {ox} (wt%)"
    return c


def _kde2d(pts, w=None, gridsize=120, pad=0.15, bw_scale=1.0, chunk=1024, bounds=None):
    pts = np.asarray(pts, float)
    n = len(pts)
    w = np.full(n, 1.0 / n) if w is None else np.asarray(w, float) / np.sum(w)
    neff = 1.0 / np.sum(w ** 2)
    mu = w @ pts
    cov = ((pts - mu) * w[:, None]).T @ (pts - mu) / max(1 - np.sum(w ** 2), 1e-12)
    cov = cov + np.eye(2) * 1e-9 * max(np.trace(cov), 1e-12)
    H = cov * (neff ** (-1.0 / 3.0)) * bw_scale ** 2          # Scott, d = 2
    lo, hi = pts.min(0), pts.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    if bounds is not None:
        gx = np.linspace(bounds[0][0], bounds[0][1], gridsize)
        gy = np.linspace(bounds[1][0], bounds[1][1], gridsize)
    else:
        gx = np.linspace(lo[0] - pad * span[0], hi[0] + pad * span[0], gridsize)
        gy = np.linspace(lo[1] - pad * span[1], hi[1] + pad * span[1], gridsize)
    X, Y = np.meshgrid(gx, gy)
    G = np.stack([X.ravel(), Y.ravel()], 1)
    Lw = np.linalg.cholesky(np.linalg.inv(H))        # whitening: q = |(g - p) @ Lw|^2
    norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(H)))
    Gw = G @ Lw
    Pw = pts @ Lw
    g2 = (Gw ** 2).sum(1)[:, None]
    Z = np.zeros(len(G))
    for s in range(0, n, chunk):
        pw = Pw[s:s + chunk]
        q = g2 + (pw ** 2).sum(1)[None, :] - 2.0 * Gw @ pw.T
        Z += np.exp(-0.5 * np.maximum(q, 0.0)) @ w[s:s + chunk]
    return X, Y, (Z * norm).reshape(X.shape)


def _hdr_levels(Z, masses):
    """Density thresholds enclosing the given probability masses (sorted ascending)."""
    z = np.sort(Z.ravel())[::-1]
    c = np.cumsum(z)
    c = c / c[-1]
    lv = sorted({float(z[min(np.searchsorted(c, m), len(z) - 1)]) for m in masses})
    if len(lv) < 2:
        lv = [lv[0], lv[0] * 1.0001]
    return lv


def _marg(ax, v, w, orient='x', truth=None, bins=50):
    hist, edges = np.histogram(v, bins=bins, weights=w, density=True)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    if orient == 'x':
        ax.fill_between(ctr, hist, step='mid', color='#6baed6', alpha=0.8)
        if truth is not None:
            ax.axvline(truth, color='#e7298a', lw=1.5)
    else:
        ax.fill_betweenx(ctr, hist, step='mid', color='#6baed6', alpha=0.8)
        if truth is not None:
            ax.axhline(truth, color='#e7298a', lw=1.5)
