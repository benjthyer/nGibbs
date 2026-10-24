"""Histogram of the entries and eigenvalues of G in the mass-balance Lagrange solve.

Thesis figure justifying the regularisation kappa (`MassBalanceProjector.tikhonov`).
In the notation of docs/continuous_model_training.tex (Section "Mass balance"),
each iteration of the projector solves

    nu (G + kappa I) = delta_b,     G = D^T D = A^T diag(m_hat)^2 A   (E x E),

and kappa only matters relative to the scale of G. This script runs an nGibbs
forward model (default mass_balance='iterative') on a chosen dataset, records
every G the projector forms, and plots

  (a) the non-zero diagonal entries G_ee (off-diagonal entries are not plotted:
      kappa is compared with the eigenvalues, and tiny off-diagonals are mostly
      the 1e-9 weight floor squared times integer stoichiometry);
  (b) the non-zero eigenvalues of G, which are the squared singular values
      sigma_i^2 of D, the quantities kappa competes with in the filter
      sigma_i^2 / (sigma_i^2 + kappa),

on log10 axes, with kappa marked. Panel (b) also marks the float64 resolution
floor eps*lambda_max (median over matrices) and shades the range lambda < 99 kappa
where the filter passes less than 99% of the correction. Exact zeros (elements
with no carrier among the present components) are excluded and counted. The
summary also compares the final residual ||b_hat - b|| of rows whose smallest
eigenvalue is below 999 kappa with that of all other rows.

How G is captured: torch.linalg.solve is wrapped only while
MassBalanceProjector.forward is running, and kappa*I is subtracted from the
matrix it receives (exact for the zero entries: (0 + kappa) - kappa == 0 in
float64). Nothing in the model is modified.

Inputs. For reproducibility the defaults are read from the installed ngibbs
package, not from paths in this repo:
  checkpoint  <ngibbs>/engine/TrainedModels/HeFESToEarthAdiabat/HeFESToEarthAdiabat_NPS.tar
  bundle      <ngibbs>/engine/TrainedModels/HeFESToEarthAdiabat/HeFESToEarthAdiabat_NPS_Test_subset15000.tar.gz
The API is built from the checkpoint's directory (it auto-discovers checkpoints
there), and the script checks that it loaded exactly this checkpoint.

Usage
  python scripts/plot_mass_balance_G_hist.py                      # defaults above, all 15000 rows
  python scripts/plot_mass_balance_G_hist.py --checkpoint <dir>/<...>_NPS.tar --bundle <...>_Test.tar.gz
  python scripts/plot_mass_balance_G_hist.py --iteration 1        # first solve of each row only
  python scripts/plot_mass_balance_G_hist.py --from-npz plots/mass_balance_G_hist.npz  # re-plot only

Outputs (default under <repo>/plots/): mass_balance_G_hist.pdf, two single-panel
histograms of the diagonal alone, labelled (D^T D)_ee, with log and linear count
axes (mass_balance_G_hist_DtD_diag_{log,linear}.pdf), and a .npz of the plotted
values (re-plot later with --from-npz, no model needed).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent                    # for _plot_output (default --out only)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Defaults resolved inside the installed ngibbs package (find_spec does not import it).
_spec = importlib.util.find_spec('ngibbs')
if _spec is None or _spec.origin is None:
    sys.exit("The ngibbs package is not importable; install it (e.g. pip install -e <nGibbs>).")
MODEL_DIR = Path(_spec.origin).parent / "engine" / "TrainedModels" / "HeFESToEarthAdiabat"
DEFAULT_CHECKPOINT = MODEL_DIR / "HeFESToEarthAdiabat_NPS.tar"
DEFAULT_BUNDLE = MODEL_DIR / "HeFESToEarthAdiabat_NPS_Test_subset15000.tar.gz"

# Colours: validated categorical slot 1 (blue) on a white surface;
# ink and hairline greys for chrome.
C_DIAG = C_EIG = "#2a78d6"
INK, INK_2, MUTED, HAIRLINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# ----------------------------------------------------------------------------- inputs
def bundle_inputs(bundle_path, batch, seed):
    """Raw feature rows from an ML-ready bundle (labels never loaded); `batch`
    rows drawn at random without replacement, or all of them if fewer."""
    from ngibbs.utils.file_utils import load_ml_bundle

    print(f"Reading features from {bundle_path}")
    bundle = load_ml_bundle(bundle_path, arrays=['features'])
    features = np.asarray(bundle.features)
    ix = bundle.ml_indexer
    headers = [str(h) for h in list(ix.featureNames) + list(ix.Elkeys)]
    if len(headers) != features.shape[1]:
        raise ValueError(f"{len(headers)} feature names {headers} for {features.shape[1]} columns")
    n = features.shape[0]
    if batch and batch < n:
        rows = np.sort(np.random.default_rng(seed).choice(n, size=batch, replace=False))
        features = features[rows]
    print(f"  using {features.shape[0]} of {n} rows; features: {headers}")
    return features.astype(np.float32), headers


# ----------------------------------------------------------------------------- capture
def capture_G(checkpoint, api_kind, device, table, headers, mass_balance):
    """Run ForwardMB; return G for every solve, its iteration number and row id,
    the final mass-balance residual per row, and kappa."""
    import torch
    from ngibbs.engine import mass_balance as mb
    from ngibbs.engine.API import HeFESToAPI, MELTSAPI

    checkpoint = Path(checkpoint).resolve()
    model_dir = checkpoint.parent
    if api_kind == 'auto':
        api_kind = 'hefesto' if 'hefesto' in model_dir.name.lower() else 'melts'
    cls = HeFESToAPI if api_kind == 'hefesto' else MELTSAPI
    print(f"Loading {cls.__name__} from {model_dir} (device={device}, mass_balance={mass_balance!r})")
    api = cls(model_dir=str(model_dir), device=device, mass_balance=mass_balance)
    loaded = [Path(p).resolve() for p in (getattr(api, a, None) for a in
              ('_iso_path', '_isen_path', '_open_path')) if p]
    if loaded and checkpoint not in loaded:
        raise RuntimeError(f"{cls.__name__} loaded {[p.name for p in loaded]}, not {checkpoint.name}")
    print(f"  checkpoint: {checkpoint}")

    # One projector call per internal sub-batch; `offset` gives its rows global ids.
    state = {'active': False, 'it': 0, 'kappa': None, 'offset': 0, 'b': 0}
    Gs, its, rows, resids = [], [], [], []
    orig_forward = mb.MassBalanceProjector.forward
    orig_solve = torch.linalg.solve

    def forward(self, n, *a, **k):
        state.update(active=True, it=0, kappa=float(self.tikhonov), b=int(n.shape[0]))
        try:
            out = orig_forward(self, n, *a, **k)
        finally:
            state['active'] = False
        resids.append(out[1].detach().double().cpu())
        state['offset'] += state['b']
        return out

    def solve(M, B, *a, **k):
        if state['active']:
            state['it'] += 1
            E, b = M.shape[-1], M.shape[0]
            G = M.detach().double() - state['kappa'] * torch.eye(E, dtype=torch.float64, device=M.device)
            Gs.append(G.cpu())
            its.append(torch.full((b,), state['it'], dtype=torch.int16))
            rows.append(torch.arange(state['offset'], state['offset'] + b))
        return orig_solve(M, B, *a, **k)

    mb.MassBalanceProjector.forward = forward
    torch.linalg.solve = solve
    try:
        api.ForwardMB(torch.tensor(table, device=device), headers=headers, outputs=['component_moles'])
    finally:
        mb.MassBalanceProjector.forward = orig_forward
        torch.linalg.solve = orig_solve

    if not Gs:
        raise RuntimeError("No mass-balance solves were captured. Is mass_balance='iterative' "
                           "(MassBalanceProjector) on this code path?")
    G, it, row = torch.cat(Gs).numpy(), torch.cat(its).numpy(), torch.cat(rows).numpy()
    resid = torch.cat(resids).numpy()
    print(f"Captured {G.shape[0]} solves of size {G.shape[1]}x{G.shape[2]} over {resid.size} rows "
          f"(iterations per row: {int(it.max())}), kappa = {state['kappa']:g}")
    return G, it, row, resid, state['kappa']


# ----------------------------------------------------------------------------- reduce
EPS64 = np.finfo(np.float64).eps


def reduce_G(G, row=None):
    """Non-zero diagonal entries and eigenvalues of G, the float64 resolution
    floor eps*lambda_max of each matrix, and each row's smallest eigenvalue."""
    B, E, _ = G.shape
    diag = np.diagonal(G, axis1=1, axis2=2)
    eig = np.linalg.eigvalsh(G)                                   # ascending, (B, E)
    lam_max = np.abs(eig).max(axis=1)
    nz = eig > E * EPS64 * lam_max[:, None]                       # numerically non-zero
    min_eig_solve = np.where(nz, eig, np.inf).min(axis=1)         # per solve

    min_eig_row = None
    if row is not None:                                           # smallest over a row's iterations
        min_eig_row = np.full(row.max() + 1, np.inf)
        np.minimum.at(min_eig_row, row, min_eig_solve)

    vals = dict(diag=diag[diag != 0], eig=eig[nz], floor=EPS64 * lam_max[lam_max > 0],
                lam_max=lam_max, min_eig_solve=min_eig_solve[np.isfinite(min_eig_solve)])
    counts = dict(n_matrices=B, E=E, diag_zero=int((diag == 0).sum()), eig_zero=int((~nz).sum()))
    return vals, counts, min_eig_row


def summarise(vals, counts, kappa, min_eig_row=None, resid=None):
    eig = vals['eig']
    print(f"\nMatrices: {counts['n_matrices']} (E = {counts['E']}); kappa = {kappa:g}")
    print(f"Excluded zeros: diagonal {counts['diag_zero']}, eigenvalues {counts['eig_zero']} "
          f"(elements with no carrier)")
    for name in ('diag', 'eig', 'min_eig_solve', 'lam_max'):
        v = vals[name]
        p = np.percentile(v, [0, 0.1, 1, 50, 100])
        print(f"  {name:13s} n={v.size:8d}  min {p[0]:.2e}  p0.1 {p[1]:.2e}  p1 {p[2]:.2e}  "
              f"median {p[3]:.2e}  max {p[4]:.2e}")
    floor = np.median(vals['floor'])
    print(f"\nfloat64 resolution floor eps*lambda_max: median {floor:.1e} "
          f"(kappa sits {np.log10(kappa / floor):.1f} decades above it)")
    print("Eigenvalues by filter factor lambda/(lambda+kappa):")
    for thr, lab in ((kappa, '< kappa        (filter < 0.5)'),
                     (99 * kappa, '< 99 kappa     (filter < 0.99)'),
                     (999 * kappa, '< 999 kappa    (filter < 0.999)')):
        print(f"  {lab}: {int((eig < thr).sum()):7d} of {eig.size} ({np.mean(eig < thr):.3%})")

    if min_eig_row is not None and resid is not None:
        n = min(min_eig_row.size, resid.size)
        m, r = min_eig_row[:n], resid[:n]
        low = m < 999 * kappa
        print(f"\nFinal mass-balance residual ||b_hat - b|| by each row's smallest eigenvalue:")
        for mask, lab in ((~low, 'all eigenvalues >= 999 kappa'), (low, 'some eigenvalue < 999 kappa')):
            if mask.any():
                q = np.percentile(r[mask], [50, 99, 100])
                print(f"  {lab:30s} rows {int(mask.sum()):6d}  median {q[0]:.1e}  p99 {q[1]:.1e}  max {q[2]:.1e}")


# ----------------------------------------------------------------------------- plot
# Font sizes (points) copied from DissertationBJT/Chapter3/Figures/plot_style.py.
# Figures are built at matplotlib's default 6.4 x 4.8 in and printed at a fixed
# 8.3 cm width; a figure W in wide uses scale = W / BASE_FIGWIDTH_IN so its text
# comes out the same size on the page.
BASE_FIGWIDTH_IN = 6.4
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, SUPTITLE_SIZE = 20, 17, 15, 11, 20


def _pyplot(scale=1.0):
    import matplotlib
    # matplotlib.use('Agg')  # disabled: a global backend switch breaks interactive plotting (e.g. PetThermoTools)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': LABEL_SIZE * scale, 'axes.titlesize': TITLE_SIZE * scale,
        'axes.labelsize': LABEL_SIZE * scale, 'legend.fontsize': LEGEND_SIZE * scale,
        'xtick.labelsize': TICK_SIZE * scale, 'ytick.labelsize': TICK_SIZE * scale,
        'figure.titlesize': SUPTITLE_SIZE * scale,
        'pdf.fonttype': 42, 'ps.fonttype': 42,           # embed TrueType (editable text)
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK_2, 'ytick.color': INK_2,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    return plt


def _save(fig, out_pdf, label):
    import matplotlib.pyplot as plt

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Provenance goes in the PDF metadata, not on the figure.
    fig.savefig(out_pdf, metadata={'Subject': label or '', 'Creator': 'scripts/plot_mass_balance_G_hist.py'},
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def plot(vals, kappa, out_pdf, log_y=True, label=None):
    figsize = (BASE_FIGWIDTH_IN, 9.0)                               # panels stacked at default width
    plt = _pyplot(scale=figsize[0] / BASE_FIGWIDTH_IN)

    lk = np.log10(kappa)
    lfloor = np.log10(np.median(vals['floor']))
    ldiag, leig = np.log10(vals['diag']), np.log10(vals['eig'])
    allv = np.concatenate([ldiag, leig, [lk, lfloor]])
    lo, hi = np.floor(allv.min()) - 0.5, np.ceil(allv.max()) + 0.5
    bins = np.linspace(lo, hi, int(round((hi - lo) * 4)) + 1)       # 4 bins per decade

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=True)
    panels = [(axes[0], ldiag, r"(a) diagonal entries $G_{ee}$", False),
              (axes[1], leig, r"(b) eigenvalues of $G$ ($=\sigma_i^2$)", True)]
    for ax, v, title, annotate in panels:
        ax.hist(v, bins=bins, histtype='step', linewidth=1.6, color=C_EIG if annotate else C_DIAG)
        kline = ax.axvline(lk, color=INK, linestyle='--', linewidth=1.0,
                           label=rf"$\kappa = 10^{{{lk:.0f}}}$")
        if annotate:
            # Directions the regularisation damps by more than 1%: lambda < 99 kappa.
            shade = ax.axvspan(lo, np.log10(99 * kappa), color=HAIRLINE, alpha=0.6, linewidth=0,
                               zorder=0, label=r"filter $\lambda/(\lambda+\kappa) < 0.99$")
            fline = ax.axvline(lfloor, color=INK_2, linestyle=':', linewidth=1.2,
                               label=r"float64 floor $\epsilon\,\lambda_{\max}$")
            legend_handles = [kline, fline, shade]
        ax.set_title(title, loc='left', color=INK)
        ax.set_ylabel("count")
        ax.set_xlim(lo, hi)
        if log_y:
            ax.set_yscale('log')
        ax.grid(axis='y', color=HAIRLINE, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[1].set_xlabel(r"$\log_{10}$ value")
    fig.legend(handles=legend_handles, loc='outside upper center', ncol=2, frameon=False,
               handlelength=2.0)
    print()
    _save(fig, out_pdf, label)


def plot_diag(vals, kappa, out_pdf, log_y=True, label=None):
    """Single-panel histogram of the non-zero diagonal entries of D^T D, with kappa marked."""
    plt = _pyplot()

    lk = np.log10(kappa)
    ldiag = np.log10(vals['diag'])
    allv = np.concatenate([ldiag, [lk]])
    lo, hi = np.floor(allv.min()) - 0.5, np.ceil(allv.max()) + 0.5
    bins = np.linspace(lo, hi, int(round((hi - lo) * 4)) + 1)       # 4 bins per decade

    fig, ax = plt.subplots()                                        # default 6.4 x 4.8 in
    ax.hist(ldiag, bins=bins, histtype='step', linewidth=1.6, color=C_DIAG)
    ax.axvline(lk, color=INK, linestyle='--', linewidth=1.0)
    # Label the kappa line directly instead of with a legend entry.
    ax.text(lk + 0.25, 0.97, rf"$\kappa = 10^{{{lk:.0f}}}$", transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontsize=TICK_SIZE, color=INK)
    ax.set_xlabel(r"$\log_{10}\,(D^{\mathsf{T}}D)_{ee}$", color=INK)
    ax.set_ylabel("count", color=INK)
    ax.set_xlim(lo, hi)
    if log_y:
        ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis='y', color=HAIRLINE, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(r"Nonzero Diagonal entries of $D^{\mathsf{T}}D$", color=INK)
    fig.tight_layout()
    _save(fig, out_pdf, label)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT,
                    help='emulator checkpoint (.tar); the API is built from its directory')
    ap.add_argument('--api', choices=['auto', 'hefesto', 'melts'], default='auto')
    ap.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE,
                    help='ML-ready .tar.gz bundle whose features are the inputs')
    ap.add_argument('--batch', type=int, default=15000, help='rows sampled from the bundle (0 = all)')
    ap.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: cuda if available)")
    ap.add_argument('--mass-balance', default='iterative')
    ap.add_argument('--iteration', default='all', help="'all' or an iteration number (1 = first solve)")
    ap.add_argument('--linear-y', action='store_true', help='linear count axis (default log)')
    ap.add_argument('--out', type=Path, help='output PDF (default plots/mass_balance_G_hist.pdf)')
    ap.add_argument('--from-npz', type=Path, help='re-plot from a saved .npz; skips the model')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.out is None:
        from _plot_output import plot_path
        args.out = plot_path('mass_balance_G_hist.pdf')
    npz_out = Path(args.out).with_suffix('.npz')

    if args.from_npz:
        d = np.load(args.from_npz, allow_pickle=True)
        G, it, kappa, stamp = d['G'], d['iteration'], float(d['kappa']), str(d['stamp'])
        row = d['row'] if 'row' in d.files else None          # absent in older .npz files
        resid = d['resid'] if 'resid' in d.files else None
    else:
        import torch
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        for p in (args.checkpoint, args.bundle):
            if not p.is_file():
                ap.error(f"not found: {p}")
        table, headers = bundle_inputs(args.bundle, args.batch, args.seed)
        G, it, row, resid, kappa = capture_G(args.checkpoint, args.api, device, table, headers,
                                             args.mass_balance)
        stamp = f"{args.checkpoint.name} | {args.bundle.name} | {table.shape[0]} rows"
        npz_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_out, G=G, iteration=it, row=row, resid=resid, kappa=kappa, stamp=stamp)
        print(f"Saved captured G to {npz_out}")

    if args.iteration != 'all':
        keep = it == int(args.iteration)
        G = G[keep]
        row = row[keep] if row is not None else None
        stamp += f" | iteration {args.iteration}"
    else:
        stamp += f" | all {int(it.max())} iterations"

    vals, counts, min_eig_row = reduce_G(G, row)
    summarise(vals, counts, kappa, min_eig_row, resid)
    print(f"Provenance (stored in PDF metadata): {stamp}")
    plot(vals, kappa, args.out, log_y=not args.linear_y, label=stamp)
    out = Path(args.out)
    for scale in ('log', 'linear'):
        plot_diag(vals, kappa, out.with_name(f"{out.stem}_DtD_diag_{scale}.pdf"),
                  log_y=scale == 'log', label=stamp)


if __name__ == '__main__':
    main()
