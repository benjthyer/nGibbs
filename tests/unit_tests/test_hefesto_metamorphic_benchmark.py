"""Benchmark the metamorphic (latent-heat / phase-change) terms against HeFESTo.

Validation ladder, in the order you should debug it -- each rung isolates one
new piece of machinery, so a failure tells you exactly where to look:

  1. fort.59  volagg, alpiso, cpiso, btiso, bsiso, gamiso
     Isomorphic aggregate derivative properties.  Needs NO dn/dT.  Tests only
     the new aggregate accumulators in aggregate.py.

  2. mu . dn/dT  and  mu . dn/dP  ~ 0
     physub.f:185-186's own self-check.  Free, and catches null-space /
     projection bugs immediately without any reference file.

  3. fort.69  alpagg, alpmet, alptot, cvtot, bstot
     First rung that exercises dn/dT and dn/dP.  Written at f25.16.

  4. fort.56  1e5*alptot, cptot, bstot, btot
     End-to-end, the values a downstream consumer actually reads.

  (A further rung, per-species dndt/dndp vs fort.42, is available if you rerun
   HeFESTo with unit 42 enabled -- see HeFESToRepository/BENCHMARK/fort.42.
   Pass --fort42 to use it.)

Usage
-----
    python -m tests.unit_tests.test_hefesto_metamorphic_benchmark \
        --sim-dir data/HeFESToWorkspace/DMM_PS \
        --param-dir src/ngibbs/engine/EOS_arithmetic/HeFESTo_Parameters_010123
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ngibbs.engine.EOS_arithmetic.hefesto_vec import (   # noqa: E402
    load_control, compute, build_tables, add_metamorphic,
)


# ── fort.* column maps, read off the write statements in physub.f ────────────
# physub.f:660  Pi depth Ti rho Vbh Vsh Vph Vsh*vsred Vph*vpred H S
#               1e5*alptot cptot bstot btot qp rhoo
F56 = dict(P=0, T=2, rho=3, Vb=4, Vs=5, Vp=6, H=9, S=10,
           alptot=11, cptot=12, bstot=13, btot=14)

# physub.f:670  Pi depth Ti volagg bsiso btiso 1e5*alpiso cpiso thet gamiso
#               qq Vdeb ph pzp tmelt
F59 = dict(P=0, T=2, volagg=3, bsiso=4, btiso=5, alpiso=6, cpiso=7, gamiso=9)

# physub.f:628  Pi depth Ti pbp ClapeyronSlope dfdptotal deltaent deltavol
#               alpagg alpmet alpmet/alpagg alptot alptot/alpagg cvtot bstot
F69 = dict(P=0, T=2, ClapeyronSlope=4, deltaent=6, deltavol=7,
           alpagg=8, alpmet=9, alptot=11, cvtot=13, bstot=14)


def _read_numeric(path: Path, skip_header: bool) -> np.ndarray:
    """Read the numeric block of a HeFESTo fort.* file.

    Two quirks have to be handled:

    * fort.56 data lines end with the *names* of the stable phases
      (``... 3.2743652 ol wa opx``), so only the leading numeric prefix of each
      line is data.
    * fort.56 carries two header lines (a 'Parameter set:' banner and a column
      title row); fort.59/fort.99 carry one; fort.69 carries none.  Any line
      whose first token is not a float is skipped outright, so ``skip_header``
      only has to cover the banner.

    Rows are never dropped -- short rows are padded with NaN -- so row indices
    stay aligned across fort.56/59/69/99.
    """
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if skip_header and i == 0:
                continue
            vals = []
            for tok in line.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    break                       # trailing phase-name tokens
            if vals:
                rows.append(vals)
    if not rows:
        raise ValueError(f'no numeric rows parsed from {path}')

    width = int(np.bincount([len(r) for r in rows]).argmax())
    short = sum(1 for r in rows if len(r) < width)
    if short:
        print(f'  [warn] {path.name}: {short} row(s) shorter than {width} '
              f'columns -- padded with NaN')
    out = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        k = min(len(r), width)
        out[i, :k] = r[:k]
    return out


def load_fort99_species(sim_dir: Path, snames) -> np.ndarray:
    """Species moles straight from fort.99, whose columns ARE the species."""
    with open(sim_dir / 'fort.99') as fh:
        header = fh.readline().split()
    # header = Pi depth Ti <species...> Gibbs Quality
    spec_cols = header[3:-2]
    data = _read_numeric(sim_dir / 'fort.99', skip_header=True)

    X = np.zeros((data.shape[0], len(snames)), dtype=np.float64)
    name_to_idx = {s.strip(): i for i, s in enumerate(snames)}
    missing = []
    for j, name in enumerate(spec_cols):
        i = name_to_idx.get(name.strip())
        if i is None:
            missing.append(name)
            continue
        X[:, i] = data[:, 3 + j]
    if missing:
        print(f'  [warn] fort.99 columns not matched to species: {missing}')
    return np.maximum(X, 0.0)


def report(title, pairs, tol_pct):
    print(f'\n{title}')
    print(f"{'quantity':16s} {'MeanRelErr%':>12s} {'MaxRelErr%':>12s} "
          f"{'MAE':>12s} {'N':>6s}  status")
    print('-' * 72)
    worst = 0.0
    for name, pred, gt in pairs:
        pred = np.asarray(pred, dtype=float)
        gt = np.asarray(gt, dtype=float)
        m = np.isfinite(pred) & np.isfinite(gt) & (np.abs(gt) > 1e-12)
        if not m.any():
            print(f'{name:16s} {"--":>12s} {"--":>12s} {"--":>12s} {0:6d}  skip')
            continue
        rel = np.abs(pred[m] - gt[m]) / np.abs(gt[m]) * 100.0
        mae = np.mean(np.abs(pred[m] - gt[m]))
        status = 'PASS' if np.mean(rel) < tol_pct else 'FAIL'
        worst = max(worst, float(np.mean(rel)))
        print(f'{name:16s} {np.mean(rel):12.5f} {np.max(rel):12.5f} '
              f'{mae:12.4g} {m.sum():6d}  {status}')
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-dir', default='data/HeFESToWorkspace/DMM_PS')
    ap.add_argument('--param-dir', default=None)
    ap.add_argument('--npz', default=None, help='dos_tables.npz path')
    ap.add_argument('--fort42', action='store_true',
                    help='also compare per-species dndt/dndp against fort.42')
    ap.add_argument('--no-fast', action='store_true',
                    help='skip the order-disorder K_T softening of Vp/Vb '
                         '(leaves the velocities isomorphic)')
    args = ap.parse_args()

    sim_dir = (REPO_ROOT / args.sim_dir) if not Path(args.sim_dir).is_absolute() \
        else Path(args.sim_dir)
    param_dir = args.param_dir or str(
        REPO_ROOT / 'src' / 'ngibbs' / 'engine' / 'EOS_arithmetic'
        / 'HeFESTo_Parameters_010123'
    )

    print(f'sim_dir   : {sim_dir}')
    print(f'param_dir : {param_dir}')

    params = load_control(str(sim_dir / 'control'), param_dir_override=param_dir)
    tables = build_tables(params, param_dir)
    print(f'species={params.nspec}  elements={tables.elements}')
    od = {params.phase_names[i]: [tables.spec_names[g] for g in b]
          for i, b in enumerate(tables.od_members) if b}
    print(f'order-disorder blocks: {od}')

    X = load_fort99_species(sim_dir, params.snames)
    f56 = _read_numeric(sim_dir / 'fort.56', skip_header=True)
    f59 = _read_numeric(sim_dir / 'fort.59', skip_header=True)
    f69 = _read_numeric(sim_dir / 'fort.69', skip_header=False)

    nrow = min(X.shape[0], f56.shape[0], f59.shape[0], f69.shape[0])
    X, f56, f59, f69 = X[:nrow], f56[:nrow], f59[:nrow], f69[:nrow]
    P = f56[:, F56['P']].copy()
    T = f56[:, F56['T']].copy()
    print(f'rows={nrow}  P=[{P.min():.2f}, {P.max():.2f}] GPa  '
          f'T=[{T.min():.1f}, {T.max():.1f}] K')

    print('\nrunning isomorphic EOS ...')
    res = compute(P, T, X, params, npz_path=args.npz)

    print('running metamorphic terms ...')
    Vp_iso, Vb_iso, Vs_iso = res['Vp'].copy(), res['Vb'].copy(), res['Vs'].copy()
    met = add_metamorphic(res, X, T, P, tables, include_fast=not args.no_fast)
    res.update(met)

    # ── Rung 1: isomorphic aggregate (no dn/dT involved) ────────────────────
    w1 = report('RUNG 1 — isomorphic aggregate vs fort.59 (no dn/dT used)', [
        ('volagg',  res['volagg'],        f59[:, F59['volagg']]),
        ('alpiso',  res['alpiso'] * 1e5,  f59[:, F59['alpiso']]),
        ('cpiso',   res['cpiso'],         f59[:, F59['cpiso']]),
        ('KT iso',  res['KTiso'],         f59[:, F59['btiso']]),
        ('KS iso',  res['KSiso'],         f59[:, F59['bsiso']]),
        ('gamma iso', res['gamiso'],      f59[:, F59['gamiso']]),
    ], tol_pct=0.5)

    # ── Rung 2: HeFESTo's own self-check, needs no reference file ───────────
    # physub.f:185-186 -- mu_i dn_i/dT and mu_i dn_i/dP must vanish.  We use the
    # partial molar volume and entropy as proxies for the chemical potential
    # gradient directions; the exact check needs cpa (= mu_i), which the
    # isomorphic EOS does not currently expose.  Instead we verify the two
    # properties that ARE structurally guaranteed: bulk composition is
    # conserved along both derivative directions.
    print('\nRUNG 2 — structural self-checks')
    s_mat = tables.s_mat
    for label in ('dndt', 'dndp'):
        resid = np.abs(res[label] @ s_mat.T).max(axis=1)
        scale = np.abs(res[label]).max(axis=1) + 1e-30
        rel = resid / scale
        ok = 'PASS' if np.nanmax(rel) < 1e-8 else 'FAIL'
        print(f'  s . {label:5s}  max|residual|/max|{label}| = '
              f'{np.nanmax(rel):.3e}   {ok}')
    inact = X <= 0.0
    leak = max(np.abs(res['dndt'][inact]).max(), np.abs(res['dndp'][inact]).max())
    print(f'  absent-species leakage        = {leak:.3e}   '
          f'{"PASS" if leak < 1e-14 else "FAIL"}')

    # ── Rung 3: metamorphic terms ───────────────────────────────────────────
    w3 = report('RUNG 3 — metamorphic terms vs fort.69', [
        ('alpagg',   res['alpagg'],  f69[:, F69['alpagg']]),
        ('alpmet',   res['alpmet'],  f69[:, F69['alpmet']]),
        ('alptot',   res['alptot'],  f69[:, F69['alptot']]),
        ('cvtot',    res['cvtot'],   f69[:, F69['cvtot']]),
        ('bstot',    res['KStot'],   f69[:, F69['bstot']]),
        ('deltaent', res['deltaent'], f69[:, F69['deltaent']]),
        ('deltavol', res['deltavol'], f69[:, F69['deltavol']]),
    ], tol_pct=2.0)

    # ── Rung 4: end-to-end fort.56 ──────────────────────────────────────────
    w4 = report('RUNG 4 — end-to-end vs fort.56', [
        ('rho',     res['rho'],           f56[:, F56['rho']]),
        ('Vs',      res['Vs'],            f56[:, F56['Vs']]),
        ('Vp',      res['Vp'],            f56[:, F56['Vp']]),
        ('S',       res['S'],             f56[:, F56['S']]),
        ('alptot',  res['alptot'] * 1e5,  f56[:, F56['alptot']]),
        ('cptot',   res['cptot'],         f56[:, F56['cptot']]),
        ('KS tot',  res['KStot'],         f56[:, F56['bstot']]),
        ('KT tot',  res['KTtot'],         f56[:, F56['btot']]),
    ], tol_pct=2.0)

    # ── Rung 5: the fast (order-disorder) velocity correction ───────────────
    # The diagnostic that matters is not the mean error but its *pressure
    # dependence*: the isomorphic velocities are biased by an amount that grows
    # with P, because the order-disorder species (pv, mw) only exist in the
    # lower mantle.  A correct fast term flattens that trend.
    print('\nRUNG 5 — order-disorder softening of Vp/Vb (Vs must not move)')
    bands = [(0, 10), (10, 23), (23, 40), (40, 80), (80, 141)]
    print(f"  {'quantity':10s} " + ' '.join(f'{lo:>3d}-{hi:<3d}GPa' for lo, hi in bands))
    for label, iso, now, col in (
        ('Vb', Vb_iso, res['Vb'], F56['Vb']),
        ('Vp', Vp_iso, res['Vp'], F56['Vp']),
        ('Vs', Vs_iso, res['Vs'], F56['Vs']),
    ):
        for tag, cur in (('iso', iso), ('fast', now)):
            bias = (cur - f56[:, col]) / f56[:, col] * 100.0
            cells = []
            for lo, hi in bands:
                m = (P >= lo) & (P < hi) & np.isfinite(bias)
                cells.append(f'{np.mean(bias[m]):+8.4f}%' if m.any() else '      --')
            print(f'  {label + " " + tag:10s} ' + ' '.join(cells))
    spread = {}
    for label, cur, col in (('Vb', res['Vb'], F56['Vb']),
                            ('Vp', res['Vp'], F56['Vp'])):
        b = [np.mean(((cur - f56[:, col]) / f56[:, col] * 100.0)[(P >= lo) & (P < hi)])
             for lo, hi in bands]
        spread[label] = max(b) - min(b)
    print(f"  P-dependence of bias (max-min across bands):  "
          f"Vb={spread['Vb']:.4f}%  Vp={spread['Vp']:.4f}%")
    dvs = np.max(np.abs(res['Vs'] - Vs_iso))
    print(f'  max |Vs_fast - Vs_iso| = {dvs:.3e} km/s   '
          f'{"PASS" if dvs < 1e-12 else "FAIL — shear modulus must not soften"}')

    # ── Optional rung 0: per-species dndt/dndp vs fort.42 ───────────────────
    if args.fort42:
        f42 = sim_dir / 'fort.42'
        if not f42.exists():
            print(f'\n[skip] {f42} not found — rerun HeFESTo with unit 42 open.')
        else:
            compare_fort42(f42, res, params)

    print('\n' + '=' * 72)
    print(f'worst mean-relative-error:  rung1={w1:.4f}%  '
          f'rung3={w3:.4f}%  rung4={w4:.4f}%')


def compare_fort42(path: Path, res, params):
    """fort.42 blocks: 'dndt and dndp by species:' then nspec lines of
    ``ispec sname dndt dndp`` (physub.f:180-183)."""
    blocks, cur = [], None
    with open(path) as fh:
        for line in fh:
            if line.strip().startswith('dndt and dndp'):
                cur = []
                blocks.append(cur)
                continue
            if cur is None:
                continue
            p = line.split()
            if len(p) >= 4:
                cur.append((int(p[0]), float(p[2]), float(p[3])))
    n = min(len(blocks), res['dndt'].shape[0])
    gt_t = np.array([[v[1] for v in b] for b in blocks[:n]])
    gt_p = np.array([[v[2] for v in b] for b in blocks[:n]])
    et = np.abs(res['dndt'][:n, :gt_t.shape[1]] - gt_t)
    ep = np.abs(res['dndp'][:n, :gt_p.shape[1]] - gt_p)
    print('\nRUNG 0 — per-species dn/dT, dn/dP vs fort.42')
    print(f'  max|dndt err| = {et.max():.3e}   (fort.42 printed to 8 dp)')
    print(f'  max|dndp err| = {ep.max():.3e}')
    k = np.unravel_index(np.argmax(et), et.shape)
    print(f'  worst dndt: row {k[0]}, species {params.snames[k[1]]}  '
          f'py={res["dndt"][k]:.8f} vs fort={gt_t[k]:.8f}')


if __name__ == '__main__':
    main()
