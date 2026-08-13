"""
Load HeFESTo's own composition derivatives.

`fort.42` is titled `dndt and dndp by species`: one block per pressure step, one line
per species (`index name dndt dndp`), blocks aligned 1:1 with `fort.56` rows. These are
the TRUE partials at fixed T and fixed P, verified by chain rule against `fort.99`:

    dn/dP|_isentrope = dn/dP|_T + dn/dT|_P * dT/dP      (median ratio 1.000)

so no P-T stencil generation and no cross-run linkage are needed -- every row carries
both partials and stays independent.
"""
from __future__ import annotations
import re
import numpy as np

_NUMERIC = re.compile(r'^[\s\-\d\.eEdD\+]+$')


def load_fort42(path):
    """-> (names (S,), dndt (nP, S), dndp (nP, S)) in mol/K and mol/GPa."""
    lines = open(path).readlines()
    blocks = [i for i, l in enumerate(lines) if 'dndt and dndp' in l]
    if not blocks:
        raise ValueError(f"{path}: no 'dndt and dndp' blocks found")
    nsp = blocks[1] - blocks[0] - 1 if len(blocks) > 1 else len(lines) - 1
    names = [lines[blocks[0] + 1 + k].split()[1] for k in range(nsp)]
    dndt = np.zeros((len(blocks), nsp)); dndp = np.zeros_like(dndt)
    for b, i in enumerate(blocks):
        for k in range(nsp):
            p = lines[i + 1 + k].split()
            dndt[b, k] = float(p[2]); dndp[b, k] = float(p[3])
    return names, dndt, dndp


def load_fort99(path):
    """-> (species_names, moles (nP, S), PdT (nP, 3)). Drops interleaved WARNING lines
    and the trailing `Gibbs` / `Quality` columns, which are NOT species."""
    lines = open(path).readlines()
    cols = lines[0].split()
    rows = [l for l in lines[1:]
            if _NUMERIC.match(l.rstrip('\n')) and len(l.split()) == len(cols)]
    A = np.array([[float(x) for x in l.split()] for l in rows])
    names = [c for c in cols[3:] if c not in ('Gibbs', 'Quality')]
    idx = [cols.index(c) for c in names]
    return names, A[:, idx], A[:, :3]


def element_total(control_path):
    """Total element moles for a run, read from its control file.

    Constant along a profile (the bulk is fixed), but NOT constant across runs:
    Htz_transition is 23.63948, BENCHMARK is 24.00450. The invariant in both is
    cations = 10.00000, with oxygen -- and therefore the total -- floating with Fe3+
    content and the Si/Mg ratio. Hardcoding 24 gives a scale error correlated with
    composition, which on an Mg/Si sweep reads as a trend rather than as noise.
    """
    lines = [l.rstrip('\n') for l in open(control_path)]
    i = next(j for j, l in enumerate(lines) if l.strip() == 'oxides') + 1
    total = 0.0
    for l in lines[i:]:
        p = l.split()
        if len(p) < 3 or not re.match(r'^[A-Za-z]+$', p[0]):
            break
        total += float(p[1])
    return total


def normalised_derivatives(sim_dir, target_names):
    """Derivatives on the model's normalisation, aligned to `target_names`.

    `forward_phase_moles` divides component moles by `reconBulkUnNormed.sum(dim=1)`,
    the ELEMENT total -- which is fixed along a profile. So `dN/dP = 0` and the quotient
    rule collapses to a constant scale factor:

        d(n_i / N_el)/dP = (1 / N_el) * (dn_i/dP)

    Had the denominator been the SPECIES total, the quotient rule would be mandatory:
    that total swings 33.6% across the Htz transition, because `ri -> pv + pe` does not
    conserve mole count.

    Species are matched BY NAME. `fort.42` uses parameter-file order
    (an, ab, sp, hc, smag, picr, en, ...) which does not match `fort.99` positionally.
    """
    import os
    names, dndt, dndp = load_fort42(os.path.join(sim_dir, 'fort.42'))
    N_el = element_total(os.path.join(sim_dir, 'control'))
    pos = {n: k for k, n in enumerate(names)}
    missing = [n for n in target_names if n not in pos]
    if missing:
        raise KeyError(f"{sim_dir}: species absent from fort.42: {missing[:5]}")
    take = [pos[n] for n in target_names]
    return dndt[:, take] / N_el, dndp[:, take] / N_el, N_el
