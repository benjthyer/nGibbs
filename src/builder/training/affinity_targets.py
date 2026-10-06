"""
Training-side half of the affinity-label scheme (see
`builder.processing.affinity_labels` for the labelling half).

Bundles carry the raw signed label g (present: g = n; absent: g = -s*|T - T*|; NaN where
no crossing exists) and its slope s. Nothing here rewrites a bundle. At workspace build
time each row is turned into two cached arrays:

    affinity_y    = h(max(g, -clip*c))            where the cell is trusted
                  = -c                            where it is not (placeholder, not a label)
    affinity_mask = True for trusted cells

    h(g) = g                        g >= 0
         = c*tanh(g/c) + eps*g      g <  0

A cell is trusted when the phase is present (g > 0), or when it is absent with a finite g
and a slope that passes the low-side lognormal filter

    s >= 10**(mean_log10_s - k*max(std_log10_s, sigma_floor)).

cStats (4 x P: c, median log10 s, trimmed mean log10 s, trimmed std log10 s) is read from
the Test bundle, where `prepareML_affinity.py` computed it. Phases with too few crossings
for the statistics to mean anything (`min_crossings`) get a fallback c (the log-median of
the reliable phases' c) and all their absent cells are untrusted -- they are still pushed
below zero by the one-sided term, they just are not regressed onto a slope estimated from
a handful of crossings.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np

DEFAULT_AFFINITY_PARAMS = dict(
    eps=0.02,            # slope of h below the tanh plateau
    clip=3.0,            # g is floored at -clip*c before activation
    k=2.0,               # low-side filter width in sigmas
    sigma_floor=0.25,    # dex; floor on the trimmed std of log10 s
    min_crossings=20,    # below this, a phase's cStats are not trusted
)


def normalize_affinity_params(cfg):
    """Recipe `affinity:` block -> plain dict with every key present, floats cast."""
    cfg = dict(cfg or {})
    out = dict(DEFAULT_AFFINITY_PARAMS)
    for key in DEFAULT_AFFINITY_PARAMS:
        if cfg.get(key) is not None:
            out[key] = type(DEFAULT_AFFINITY_PARAMS[key])(float(cfg[key]))
    return out


def _bundle_file(bundle_path):
    p = Path(bundle_path)
    if not str(p).endswith('.tar.gz'):
        p = Path(str(p) + '.tar.gz')
    return p


def load_cstats_from_bundle(bundle_path):
    """Read cStats.npy (+ its json sidecar) out of a bundle without extracting the rest.

    Returns (cstats [4, P] float64, info dict with 'phases' and 'crossings_per_phase')."""
    p = _bundle_file(bundle_path)
    with tarfile.open(p, 'r:gz') as tar:
        names = set(tar.getnames())
        if 'cStats.npy' not in names:
            raise FileNotFoundError(
                f"{p.name} has no cStats.npy. cStats is written into the validation (Test) "
                f"bundle by prepareML_affinity.py; point the affinity block at that bundle.")
        cstats = np.load(io.BytesIO(tar.extractfile('cStats.npy').read()))
        info = {}
        if 'cStats.json' in names:
            info = json.loads(tar.extractfile('cStats.json').read().decode())
    cstats = np.asarray(cstats, dtype=np.float64)
    if cstats.ndim != 2 or cstats.shape[0] != 4:
        raise ValueError(f"cStats in {p.name} has shape {cstats.shape}; expected (4, P)")
    return cstats, info


def bundle_has_affinity_labels(bundle_path):
    with tarfile.open(_bundle_file(bundle_path), 'r:gz') as tar:
        names = set(tar.getnames())
    return 'g_labels.npy' in names and 's_labels.npy' in names


class AffinityTargets:
    """Per-phase constants derived once from cStats + params, and the row transform."""

    def __init__(self, cstats, params=None, crossings_per_phase=None, phases=None):
        self.params = normalize_affinity_params(params)
        self.cstats = np.asarray(cstats, dtype=np.float64)
        self.phases = list(phases) if phases is not None else None
        P = self.cstats.shape[1]
        c_raw, med, mean, std = self.cstats

        n_cross = (np.asarray(crossings_per_phase, dtype=np.int64)
                   if crossings_per_phase is not None else np.full(P, np.iinfo(np.int64).max))
        reliable = (np.isfinite(c_raw) & (c_raw > 0) & np.isfinite(mean) & np.isfinite(std)
                    & (n_cross >= self.params['min_crossings']))
        if not reliable.any():
            raise ValueError("No phase has usable cStats; cannot set the affinity scale c.")
        c_fallback = float(10 ** np.median(np.log10(c_raw[reliable])))
        self.reliable = reliable
        self.c = np.where(reliable, c_raw, c_fallback).astype(np.float64)
        sigma = np.maximum(np.nan_to_num(std, nan=0.0), self.params['sigma_floor'])
        # Unreliable phases: threshold +inf, so every absent cell is untrusted.
        self.s_min = np.where(reliable, 10 ** (mean - self.params['k'] * sigma), np.inf)
        self.c_fallback = c_fallback

    # ------------------------------------------------------------------ transform
    def activate(self, g):
        c, eps = self.c.astype(np.float32), np.float32(self.params['eps'])
        return np.where(g >= 0, g, c * np.tanh(g / c) + eps * g)

    def transform_chunk(self, g, s):
        """(rows, P) g and s -> (affinity_y float32, affinity_mask bool)."""
        g = np.asarray(g, dtype=np.float32)
        s = np.asarray(s, dtype=np.float32)
        c = self.c.astype(np.float32)
        finite_g = np.isfinite(g)
        present = finite_g & (g > 0)
        with np.errstate(invalid='ignore'):
            trusted_absent = (finite_g & (g <= 0) & np.isfinite(s)
                              & (s >= self.s_min.astype(np.float32)))
        mask = present | trusted_absent
        g_clip = np.maximum(np.where(finite_g, g, 0.0), -self.params['clip'] * c)
        y = np.where(mask, self.activate(g_clip), -c).astype(np.float32)
        return y, mask

    # ------------------------------------------------------------------ bookkeeping
    def fingerprint(self):
        h = hashlib.sha1(np.ascontiguousarray(self.cstats).tobytes()).hexdigest()
        return {'cstats_sha1': h, 'params': self.params,
                'reliable': self.reliable.astype(int).tolist()}

    def summary(self, phase_names=None):
        names = phase_names or self.phases or [str(i) for i in range(len(self.c))]
        lines = [f"[affinity] eps={self.params['eps']} clip={self.params['clip']} "
                 f"k={self.params['k']} sigma_floor={self.params['sigma_floor']} "
                 f"min_crossings={self.params['min_crossings']}"]
        for j, n in enumerate(names):
            tag = '' if self.reliable[j] else f'  UNRELIABLE -> c fallback {self.c_fallback:.3g}, absent cells untrusted'
            lines.append(f"  {n:<14} c={self.c[j]:.4g}  s_min={self.s_min[j]:.3g}{tag}")
        return '\n'.join(lines)

    @classmethod
    def from_bundle(cls, bundle_path, params=None):
        cstats, info = load_cstats_from_bundle(bundle_path)
        phases = info.get('phases')
        cpp = info.get('crossings_per_phase')
        if isinstance(cpp, dict) and phases:
            cpp = [cpp.get(p, 0) for p in phases]
        return cls(cstats, params=params, crossings_per_phase=cpp, phases=phases)


def check_phase_order(targets, ml_indexer):
    """cStats columns, molar-label columns and the model's mole heads must agree."""
    if targets.phases is None:
        return
    order = sorted(ml_indexer.mass_phasedict, key=ml_indexer.mass_phasedict.get)
    if list(order) != list(targets.phases):
        raise ValueError(
            "cStats phase order does not match the bundle's molar-label columns:\n"
            f"  cStats: {targets.phases}\n  bundle: {order}")
