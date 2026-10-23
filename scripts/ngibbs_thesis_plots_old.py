"""
Thesis summary plots built entirely on the installed ``ngibbs`` package.

Nothing here touches ``src/``, ``builder`` or ``scripts`` on sys.path -- every
import is ``ngibbs.<...>``, and every data file is located relative to the
installed package (``ngibbs.engine`` / ``ngibbs.deployment_tests``).

Produces, under ``--out-dir``:

1. Training/test-data density clouds (one figure per data family), drawn from
   the ``*_Test_subset*.tar.gz`` bundles the deployment tests evaluate on,
   with the deployment-test standard rocks overlaid as red 'x':

     density_PT_HeFESToMars.png        P-T cloud (NPT + NPS subsets pooled)
     density_PT_HeFESToEarth.png
     density_Harker_HeFESToMars.png    Si vs every other element (cation mol frac)
     density_Harker_HeFESToEarth.png
     density_Harker_MELTS102.png       SiO2 vs every other oxide (oxide wt frac),
     density_Harker_MELTS120.png       closed-system Cr bundles (NPT + NPS pooled)

2. Phase and property sections for the deployment-test standard rocks, one
   figure per model x pathway, one row per standard rock -- the same layout as
   scripts/phase_comparison.py and scripts/property_comparison.py:

     sections_phase_<model>_<pathway>.png     GT phase stack | emulator phase stack
     sections_property_<model>_<pathway>.png  one column per property: GT (k-),
                                              emulator (r--), GT assemblage +
                                              internal EOS (b:); isentropic adds
                                              a T panel (GT vs temperature model)
                                              HeFESTo: rho, VP, VS, S, Cp, KS, alpha vs P
                                              MELTS:   rho, S, Cp, V, H vs T

   <model> in {HeFESToMars, HeFESToEarth, MELTS102, MELTS120}; MELTS uses the
   Cr sub-API and the Cr standards. <pathway> in {isothermal, isentropic}.

Usage
-----
    python ngibbs_thesis_plots.py --out-dir thesis_plots
    python ngibbs_thesis_plots.py --only density          # no model loading
    python ngibbs_thesis_plots.py --only sections --gpu
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ngibbs.engine as _ngibbs_engine
import ngibbs.deployment_tests as _ngibbs_dt
from ngibbs.deployment_tests.training_coverage import (
    _read_bundle_arrays,
    _bundle_ps_t,
    _bundle_composition,
    _density_heatmap,
    hefesto_standard_points,
    melts_standard_points,
)

ENGINE_DIR = Path(_ngibbs_engine.__file__).resolve().parent
TRAINED_DIR = ENGINE_DIR / 'TrainedModels'
DT_DIR = Path(_ngibbs_dt.__file__).resolve().parent
HEFESTO_STANDARDS = DT_DIR / 'HeFESToAdiabatStandards'
MELTS_STANDARDS = DT_DIR / 'MELTSIsobaricStandards'

MELTS_VARIANT = 'Cr'
PATHWAYS = ('isothermal', 'isentropic')

# --------------------------------------------------------------------------- #
# data families
# --------------------------------------------------------------------------- #
# name -> (model sub-dir, bundle glob(s), composition space, standards)
FAMILIES = {
    'HeFESToMars': dict(
        model_dir='HeFESToMars', label='HeFESTo Mars',
        bundles=['HeFESToMars_NPT_Test_subset*.tar.gz', 'HeFESToMars_NPS_Test_subset*.tar.gz'],
        space='elements', singleton='HeFESToMarsEmulator', std_planet='Mars'),
    'HeFESToEarth': dict(
        model_dir='HeFESToEarthAdiabat', label='HeFESTo Earth',
        bundles=['HeFESToEarthAdiabat_NPT_Test_subset*.tar.gz',
                 'HeFESToEarthAdiabat_NPS_Test_subset*.tar.gz'],
        space='elements', singleton='HeFESToEmulator', std_planet='Earth'),
    'MELTS102': dict(
        model_dir='102', label='MELTS 1.0.2',
        bundles=[f'102Closed_{MELTS_VARIANT}_NPT_Test_subset*.tar.gz',
                 f'102Closed_{MELTS_VARIANT}_NPS_Test_subset*.tar.gz'],
        space='oxides', singleton='MELTS102Emulator', std_version='102'),
    'MELTS120': dict(
        model_dir='120', label='MELTS 1.2.0',
        bundles=[f'120SedIgClosed_{MELTS_VARIANT}_NPT_Test_subset*.tar.gz',
                 f'120SedIgClosed_{MELTS_VARIANT}_NPS_Test_subset*.tar.gz'],
        space='oxides', singleton='MELTS120Emulator', std_version='120'),
}


def _find_bundle(model_dir: str, pattern: str) -> Optional[Path]:
    """Same lookup order as EmulatorAPI: the model's own directory first,
    then anywhere under ngibbs/deployment_tests."""
    hits = sorted((TRAINED_DIR / model_dir).glob(pattern))
    if not hits:
        hits = sorted(DT_DIR.rglob(pattern))
    if not hits:
        warnings.warn(f'No test bundle matching {pattern!r} for {model_dir}')
        return None
    return hits[-1]


def _load_family(fam: dict):
    """Pool every bundle of a family -> (P, T, comp dict, oxide labels)."""
    P_all, T_all, comps = [], [], []
    for pat in fam['bundles']:
        b = _find_bundle(fam['model_dir'], pat)
        if b is None:
            continue
        print(f'  reading {b.name}')
        feats, free, bounds, meta = _read_bundle_arrays(b)
        P, _S, T = _bundle_ps_t(feats, free, bounds, meta)
        comp = _bundle_composition(feats, meta, fam['space'])
        P_all.append(P)
        T_all.append(T if T is not None else np.full_like(P, np.nan))
        comps.append(comp)
    if not comps:
        raise FileNotFoundError(f"No test bundles found for {fam['label']}")
    keys = [k for k in comps[0] if all(k in c for c in comps[1:])]
    comp = {k: np.concatenate([c[k] for c in comps]) for k in keys}
    return np.concatenate(P_all), np.concatenate(T_all), comp, keys


# --------------------------------------------------------------------------- #
# 1. density clouds
# --------------------------------------------------------------------------- #
def plot_pt_density(name, fam, P, T, std, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sxy = (std[1], std[0]) if std is not None else None
    _density_heatmap(ax, T, P, 'T (K)', 'P (GPa)',
                     f"{fam['label']}: P-T distribution of test data (n={np.isfinite(T).sum():,})",
                     bins=80, standard_xy=sxy)
    ax.invert_yaxis()
    fig.tight_layout()
    path = out_dir / f'density_PT_{name}.png'
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    print(f'  -> {path}')


def plot_harker_density(name, fam, comp, std_comp, out_dir, ncols=4):
    pivot = 'SiO2' if fam['space'] == 'oxides' else 'Si'
    unit = 'oxide wt frac.' if fam['space'] == 'oxides' else 'cation mol frac.'
    others = [k for k in comp if k != pivot]
    nrows = int(np.ceil(len(others) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.9 * nrows), squeeze=False)
    for k, other in enumerate(others):
        ax = axes[k // ncols][k % ncols]
        sxy = None
        if std_comp is not None and pivot in std_comp and other in std_comp:
            sxy = (std_comp[pivot], std_comp[other])
        _density_heatmap(ax, comp[pivot], comp[other], f'{pivot} ({unit})', other, other,
                         bins=50, standard_xy=sxy)
    for k in range(len(others), nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    n = len(next(iter(comp.values())))
    fig.suptitle(f"{fam['label']}: Harker diagrams of test data (n={n:,})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / f'density_Harker_{name}.png'
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    print(f'  -> {path}')


def run_density(out_dir: Path):
    for name, fam in FAMILIES.items():
        print(f'[density] {fam["label"]}')
        P, T, comp, keys = _load_family(fam)
        if fam['space'] == 'elements':
            std = hefesto_standard_points(HEFESTO_STANDARDS / fam['std_planet'])
            plot_pt_density(name, fam, P, T, std, out_dir)
        else:
            std = melts_standard_points(MELTS_STANDARDS / fam['std_version'],
                                        variant=MELTS_VARIANT, oxide_cols=keys)
        plot_harker_density(name, fam, comp, std[3] if std is not None else None, out_dir)


# --------------------------------------------------------------------------- #
# 2. sections -- shared plotting helpers (layout mirrors scripts/
#    phase_comparison.py and scripts/property_comparison.py)
# --------------------------------------------------------------------------- #
def _np(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


_PATHWAY_LABEL = {'isothermal': 'Isothermal (P, T)', 'isentropic': 'Isentropic (P, S)'}


def _save(fig, path, top=0.96):
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    print(f'  -> {path}')


def _phase_grid(rows, draw, gt_label, pathway, xlabel, title, path):
    """One row per rock: GT phase stack (left) | emulator phase stack (right).
    ``draw(ax, fractions, rock, panel_title)`` renders one stack."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.4 * len(rows)), squeeze=False)
    for r, d in enumerate(rows):
        draw(axes[r][0], d['gt_phase'], d, f"{d['title']} -- {gt_label} (GT)")
        draw(axes[r][1], d['em_phase'], d, f"{d['title']} -- Emulator ({pathway})")
        for ax in axes[r]:
            ax.set_xlabel(xlabel)
    fig.suptitle(f'{title} -- {_PATHWAY_LABEL[pathway]} pathway: GT vs emulator', fontsize=12)
    _save(fig, path)


def _property_grid(rows, props, gt_label, pathway, xlabel, title, path):
    """One row per rock, one column per property: ground truth (k-), emulator
    predicted assemblage (r--), GT assemblage through the internal EOS (b:).
    The isentropic pathway adds a final T panel (GT vs temperature model)."""
    show_T = pathway == 'isentropic'
    n_cols = len(props) + (1 if show_T else 0)
    col_w = 3.6 if n_cols <= 5 else 3.0
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(col_w * n_cols, 3.2 * len(rows)),
                             squeeze=False)
    for r, d in enumerate(rows):
        order = np.argsort(d['x'])
        x = d['x'][order]
        for c, (key, label) in enumerate(props):
            ax = axes[r][c]
            ax.plot(x, d['gt'][key][order], 'k-', lw=1.5, label=f'{gt_label} (GT)')
            ax.plot(x, d['em'][key][order], 'r--', lw=1.5, label='Emulator (pred. assemblage)')
            ax.plot(x, d['gi'][key][order], 'b:', lw=1.8, label='GT assemblage + internal EOS')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(label if c > 0 else f"{d['title']}\n{label}", fontsize=9)
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(label, fontsize=10)
            if r == 0 and c == len(props) - 1:
                ax.legend(fontsize=7, loc='best')
        if show_T:
            ax = axes[r][-1]
            ax.plot(x, d['T_gt'][order], 'k-', lw=1.5, label=f'{gt_label} (GT)')
            ax.plot(x, d['T_em'][order], 'r--', lw=1.5, label='Emulator T(P, S)')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(d['T_label'])
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(d['T_label'], fontsize=10)
                ax.legend(fontsize=7, loc='best')
    fig.suptitle(f'{title} -- {_PATHWAY_LABEL[pathway]} pathway: emulator vs {gt_label} ground truth',
                 fontsize=12)
    _save(fig, path, top=0.97)


# --------------------------------------------------------------------------- #
# 2a. HeFESTo sections
# --------------------------------------------------------------------------- #
HEFESTO_SECTION_PROPS = [('rho', r'$\rho$ (g/cm$^3$)'), ('Vp_fast', r'$V_P$ (km/s)'),
                         ('Vs', r'$V_S$ (km/s)'), ('S', 'S (J/g/K)'), ('cptot', r'$C_p$ (J/g/K)'),
                         ('KStot', r'$K_S$ (GPa)'), ('alptot', r'$\alpha$ ($10^{-5}$ K$^{-1}$)')]


def hefesto_sections(name, fam, api, out_dir):
    from ngibbs.deployment_tests.meltstable_comparison import (
        find_tables, _read_conditions, _load_component_moles,
        _load_composition, _load_fort99_phase_totals, _gt_phase_fractions,
        _emulator_phase_fractions, _emulator_assemblage, _eval_assemblage, PROPERTY_MAP,
    )
    from ngibbs.engine.EOS_arithmetic.hefesto_vec import compute_within_phase_frac
    from ngibbs.deployment_tests._phase_plotting import (
        phase_colors, build_ordered_phases, draw_phase_stack,
    )
    rocks = find_tables(HEFESTO_STANDARDS / fam['std_planet'])[:3]
    indexer = api.isothermal_emulator.ml_indexer
    mpd = indexer.mass_phasedict
    ordered = build_ordered_phases(mpd)
    colors = phase_colors()

    data = []
    for sim_dir in rocks:
        df, P, T, S = _read_conditions(sim_dir)
        # Restrict to the model's training pressure range, applying the SAME
        # row mask to fort.56 (conditions/properties) and fort.99 (phases).
        lo, hi = api.training_pressure_bounds('both')
        totals_all = _load_fort99_phase_totals(sim_dir)
        n = min(len(P), *(len(v) for v in totals_all.values()))
        keep = np.zeros(len(P), dtype=bool)
        keep[:n] = np.isfinite(P[:n]) & (P[:n] >= lo) & (P[:n] <= hi)
        df = df.loc[keep].reset_index(drop=True)
        P, T, S = P[keep], T[keep], S[keep]
        totals = {k: v[:n][keep[:n]] for k, v in totals_all.items()}
        gt_cm = _load_component_moles(sim_dir, indexer)[:n][keep[:n]]
        gt_wpf = compute_within_phase_frac(gt_cm, list(indexer.label_indices.values()))
        gi = _eval_assemblage(api, gt_cm, P, T, gt_wpf)
        data.append(dict(
            title=sim_dir.name.replace('adiabat', '').replace('mars', ''), df=df, P=P, T=T, S=S,
            comp=_load_composition(sim_dir), gt_phase=_gt_phase_fractions(totals, mpd, len(P)),
            gt={k: pd.to_numeric(df[PROPERTY_MAP[k][0]], errors='coerce').to_numpy(dtype=np.float64)
                for k, _ in HEFESTO_SECTION_PROPS},
            gi={k: _np(gi[k]) * PROPERTY_MAP[k][2] for k, _ in HEFESTO_SECTION_PROPS},
        ))

    def draw(ax, fracs, d, title):
        draw_phase_stack(ax, fracs, np.argsort(d['P']), d['P'], ordered, mpd, colors, title)

    for pathway in PATHWAYS:
        isen = pathway == 'isentropic'
        header = 'S(J/g/K)(System_main)' if isen else 'T(K)(System_main)'
        rows = []
        for d in data:
            second = d['S'] if isen else d['T']
            cm, wpf, T_em = _emulator_assemblage(api, d['P'], second, d['comp'], header, isen)
            T_eval = _np(T_em) if isen else d['T']
            props = _eval_assemblage(api, cm, d['P'], T_eval, wpf)
            rows.append(dict(
                d, x=d['P'],
                em_phase=_emulator_phase_fractions(api, d['P'], second, d['comp'], header),
                em={k: _np(props[k]) * PROPERTY_MAP[k][2] for k, _ in HEFESTO_SECTION_PROPS},
                T_gt=d['T'], T_em=T_eval, T_label='T (K)',
            ))
        _phase_grid(rows, draw, 'HeFESTo', pathway, 'P (GPa)', f"{fam['label']}: phase abundances",
                    out_dir / f'sections_phase_{name}_{pathway}.png')
        _property_grid(rows, HEFESTO_SECTION_PROPS, 'HeFESTo', pathway, 'P (GPa)', fam['label'],
                       out_dir / f'sections_property_{name}_{pathway}.png')


# --------------------------------------------------------------------------- #
# 2b. MELTS sections (Cr sub-API, Cr standards; isobaric cooling -> x = T)
# --------------------------------------------------------------------------- #
MELTS_SECTION_PROPS = [('rho', r'$\rho$ (g/cm$^3$)'), ('S', 'S (J/g/K)'), ('Cp', r'$C_p$ (J/g/K)'),
                       ('V', r'V (cm$^3$/g)'), ('H', 'H (J/g)')]


def melts_sections(name, fam, api, out_dir):
    from ngibbs.deployment_tests.melts_comparison import (
        ROCKS, read_system_main, read_bulk_comp, _system_main_feature, _ENTROPY_FEATURE,
        _emulator_forward, _emulator_input_oxides, ml_indexer_to_melts_vec_assemblage,
        _bulk_props_and_coverage, _gt_property_table, _gt_phase_mass_fractions,
        _emulator_phase_mass_fractions, _ensure_alkali_feldspar_alias, _BARS_PER_GPA,
        load_gt_assemblage,
    )
    from ngibbs.deployment_tests._melts_phase_plotting import (
        MELTS_PHASE_STACK_ORDER, melts_phase_colors, build_ordered_melts_phases,
        draw_melts_phase_stack,
    )
    _ensure_alkali_feldspar_alias(api)
    sub = api.cr if MELTS_VARIANT == 'Cr' else api.nocr
    std_dir = MELTS_STANDARDS / fam['std_version'] / MELTS_VARIANT
    ordered = build_ordered_melts_phases(MELTS_PHASE_STACK_ORDER)
    colors = melts_phase_colors()
    have_eos = getattr(api, 'melts_solid_params', None) is not None \
        and getattr(api, 'melts_liquid_params', None) is not None
    if not have_eos:
        warnings.warn(f"{fam['label']}: MELTS EOS params not loaded -- property sections skipped.")

    def _props(asm, PT, mass):
        props, _cov = _bulk_props_and_coverage(
            api, asm['phase_composition'], asm['phase_moles'], PT, mass,
            liquid_oxides=asm['liquid_oxides'], liquid_oxide_labels=asm['liquid_oxide_labels'])
        n = PT.shape[0]
        return {k: _np(props.get(k, np.full(n, np.nan))) for k, _ in MELTS_SECTION_PROPS}

    data = []
    for rock in ROCKS[:3]:
        rock_dir = std_dir / rock
        if not rock_dir.exists():
            warnings.warn(f'missing MELTS standard {rock_dir}')
            continue
        sysmain = read_system_main(rock_dir)
        mass_g = sysmain['mass'].values
        P_bar = sysmain['Pressure'].values
        d = dict(
            rock_dir=rock_dir, bulk=read_bulk_comp(rock_dir),
            title=f"{rock} (P={P_bar[0] / _BARS_PER_GPA:.3g} GPa)",
            T_C=sysmain['Temperature'].values, P_bar=P_bar,
            S_spec=_system_main_feature(sysmain, _ENTROPY_FEATURE),
            gt_phase=_gt_phase_mass_fractions(rock_dir, sysmain['index'].values.astype(int), mass_g),
            gt={k: _np(v) for k, v in _gt_property_table(sysmain, mass_g).items()},
        )
        if have_eos:
            gt_asm = load_gt_assemblage(rock_dir, api)
            d['gi'] = _props(gt_asm, gt_asm['PT'], mass_g)
        data.append(d)

    def draw(ax, fracs, d, title):
        draw_melts_phase_stack(ax, fracs, d['T_C'], ordered, colors, title)

    for pathway in PATHWAYS:
        isen = pathway == 'isentropic'
        emu = sub.isentropic_emulator if isen else sub.isothermal_emulator
        header = _ENTROPY_FEATURE if isen else 'Temperature(System_main)'
        title = f"{fam['label']} ({MELTS_VARIANT})"
        rows = []
        for d in data:
            second = d['S_spec'] if isen else d['T_C']
            rows.append(dict(d, x=d['T_C'], second=second, em_phase=_emulator_phase_mass_fractions(
                api, emu, d['P_bar'], second, header, d['bulk'])))
        _phase_grid(rows, draw, 'alphaMELTS', pathway, 'T (°C)', f'{title}: phase mass fractions',
                    out_dir / f'sections_phase_{name}_{pathway}.png')
        if not have_eos:
            continue

        # Separate from the phase figure so a failing temperature model (needed
        # only by the isentropic property pathway) doesn't cost the phase plot.
        try:
            for row in rows:
                cm, comp, mass, T_C_em = _emulator_forward(
                    api, _emulator_input_oxides(emu), row['P_bar'], row['second'], header,
                    row['bulk'], temperature=isen)
                bridged = ml_indexer_to_melts_vec_assemblage(sub, api, emu, cm, mass, comp)
                T_C_eval = _np(T_C_em) if isen else row['T_C']
                PT = np.column_stack([row['P_bar'] / _BARS_PER_GPA, T_C_eval + 273.15])
                row.update(em=_props(bridged, PT, 100.0),
                           T_gt=row['T_C'], T_em=T_C_eval, T_label='T (°C)')
        except Exception as e:
            warnings.warn(f"{title} {pathway} property section skipped: {type(e).__name__}: {e}")
            continue
        _property_grid(rows, MELTS_SECTION_PROPS, 'alphaMELTS', pathway, 'T (°C)', title,
                       out_dir / f'sections_property_{name}_{pathway}.png')


def run_sections(out_dir: Path, use_gpu: bool):
    import ngibbs.engine.models as models  # loads every packaged emulator
    suffix = 'GPU' if use_gpu and torch.cuda.is_available() else 'CPU'
    for name, fam in FAMILIES.items():
        api = getattr(models, f"{fam['singleton']}{suffix}")
        print(f"[sections] {fam['label']} ({suffix})")
        try:
            if fam['space'] == 'elements':
                hefesto_sections(name, fam, api, out_dir)
            else:
                melts_sections(name, fam, api, out_dir)
        except Exception as e:  # keep going so one model can't kill the batch
            warnings.warn(f"{fam['label']} sections failed: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default='ngibbs_thesis_plots')
    ap.add_argument('--only', choices=['density', 'sections'], default=None)
    ap.add_argument('--gpu', action='store_true', help='use the GPU singletons when CUDA is available')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.only in (None, 'density'):
        run_density(out)
    if args.only in (None, 'sections'):
        # No torch.no_grad() here: the MELTS internal EOS (melts_vec's Duan
        # fluid terms) takes autograd derivatives; the emulator forward passes
        # already run without grad on their own.
        run_sections(out, args.gpu)