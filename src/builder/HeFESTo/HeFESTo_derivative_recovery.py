"""
Recover dn/dT|_P and dn/dP|_T for HeFESTo runs whose build did not write `fort.42`.

The derivatives are not extra information -- they are a deterministic function of the
assemblage and the P-T state, obtained by solving the same Hessian system `physub.f`
solves. `hefesto_vec.metamorphic.molar_derivatives` already implements it. So any run
that kept `fort.99` (species moles) and `fort.56` (P, T) can have its derivatives
reconstructed offline, with no need to re-run HeFESTo.

Verified against a run that *does* carry `fort.42` (Htz_transition, 501 steps,
20-25 GPa at 10 MPa): mean correlation 0.997 over the 28 species that move, and an
overall `dn/dP` rms error of 3e-5 against a signal rms of 0.859 -- agreement to ~4
decimal places.

On the active set
-----------------
`add_metamorphic` can prune trace species out of the active set before the solve. That
option exists because emulator-predicted assemblages carry small spurious phases whose
`dndt` blows up `cpmet`; HeFESTo's own minimiser does not have that problem, so pass
`nsmall_rel=0` when the moles come from `fort.99`. Measured on Htz_transition:

    nsmall_rel = 0        mean corr 0.9972   rms err 2.6e-5
    nsmall_rel = 1e-6     mean corr 0.9787   rms err 7.5e-5

Validate with the **correlation**, not a median ratio. A wrong active set leaves the
median ratio sitting reassuringly near 1.0 while the point-by-point shape is destroyed,
so a ratio check passes a reconstruction that is badly wrong.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

_NUMERIC = re.compile(r'^[\s\-\d\.eEdD\+]+$')


def read_fort99_moles(path: str, snames: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (n (nP, S) in EOS species order, P (nP,), T (nP,)).

    Drops interleaved WARNING lines, and drops the trailing `Gibbs` / `Quality`
    columns, which are not species.
    """
    with open(path) as handle:
        lines = handle.readlines()
    cols = lines[0].split()
    rows = [l for l in lines[1:]
            if _NUMERIC.match(l.rstrip('\n')) and len(l.split()) == len(cols)]
    arr = np.array([[float(x) for x in l.split()] for l in rows])
    species = [c for c in cols[3:] if c not in ('Gibbs', 'Quality')]

    pos = {str(s).strip(): i for i, s in enumerate(snames)}
    n = np.zeros((len(arr), len(snames)), dtype=float)
    for c in species:
        if c in pos:
            n[:, pos[c]] = arr[:, cols.index(c)]
    return n, arr[:, 0], arr[:, 2]


def recover_derivatives(sim_dir: str, params, param_dir: Optional[str] = None,
                        nsmall_rel: float = 0.0,
                        tables=None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """-> (dndt (nP, S), dndp (nP, S), snames) in mol/K and mol/GPa, unnormalised.

    params    : `HeFESToParams`, e.g. from `hefesto_vec.load_control(control_path)`
    param_dir : HeFESTo parameter directory, for the regular-solution tables.
                Ignored when `tables` is supplied.
    tables    : prebuilt `MetamorphicTables`. `build_tables` parses the whole
                parameter set, so pass it in when looping over many simulations.
    """
    from ngibbs.engine.EOS_arithmetic.hefesto_vec import metamorphic as M
    from ngibbs.engine.EOS_arithmetic.hefesto_vec.compute import compute

    snames = [str(s).strip() for s in params.snames]
    n, P, T = read_fort99_moles(os.path.join(sim_dir, 'fort.99'), snames)
    result = compute(P=P, T=T, X=n, params=params)
    if tables is None:
        if param_dir is None:
            raise ValueError('recover_derivatives needs either param_dir or tables')
        tables = M.build_tables(params, param_dir)
    out = M.add_metamorphic(result, n, T, P, tables, nsmall_rel=nsmall_rel)
    return out['dndt'], out['dndp'], snames


def write_fort42(path: str, snames: Sequence[str],
                 dndt: np.ndarray, dndp: np.ndarray) -> None:
    """Emit a `fort.42`-format file so recovered runs feed the normal importer."""
    with open(path, 'w') as handle:
        for b in range(dndt.shape[0]):
            handle.write(' dndt and dndp by species:\n')
            for k, name in enumerate(snames):
                handle.write(f'{k+1:5d} {name:<9s}{dndt[b,k]:15.8f}{dndp[b,k]:15.8f}\n')


def validate_against_fort42(sim_dir: str, dndt: np.ndarray, dndp: np.ndarray,
                            snames: Sequence[str]) -> dict:
    """Correlation-based check where a genuine `fort.42` exists. Use correlation, not
    median ratio -- a bad active set preserves the ratio while destroying the shape."""
    from .HeFESTo_derivative_import import load_fort42
    names42, dndt42, dndp42 = load_fort42(os.path.join(sim_dir, 'fort.42'))
    pos = {s: i for i, s in enumerate(snames)}
    m = min(len(dndp42), len(dndp))
    corrs, errs = [], []
    for j, name in enumerate(names42):
        if name not in pos:
            continue
        a, b = dndp42[:m, j], dndp[:m, pos[name]]
        if np.abs(a).max() < 1e-8:
            continue
        sel = np.abs(a) > 0.02 * np.abs(a).max()
        if sel.sum() < 5:
            continue
        corrs.append(float(np.corrcoef(a[sel], b[sel])[0, 1]))
        errs.append(float(np.sqrt(np.mean((a - b) ** 2))))
    return dict(n_species=len(corrs), mean_corr=float(np.nanmean(corrs)),
                rms_error=float(np.nanmean(errs)),
                signal_rms=float(np.sqrt(np.nanmean(dndp42 ** 2))))
