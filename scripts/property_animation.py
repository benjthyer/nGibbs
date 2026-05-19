"""Animated thermodynamic property transects — emulator-only animations across pressure.

For each animation frame, plots up to five isentropes as lines on property vs. pressure axes.
Pressure and entropy must vary; exactly one additional column may also vary (the animation axis).
"""

import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

base_path = str(Path(__file__).parent.parent)
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from src.nMELTS.engine.API import HeFESToEmulatorCPU, HeFESToEmulatorGPU
from src.nMELTS.utils.math_utils import grid_sample, mix_compositions

PROPERTY_LABELS = {
    'entropy_by_mass':               'S (J/kg/K)',
    'density':                       'ρ (kg/m³)',
    'p_wave_velocity':               'VP (m/s)',
    's_wave_velocity':               'VS (m/s)',
    'isentropic_bulk_modulus_reuss': 'KS (Pa)',
    'isothermal_bulk_modulus_reuss': 'KT (Pa)',
    'heat_capacity_p_by_mass':       'cp (J/kg/K)',
    'heat_capacity_v_by_mass':       'cv (J/kg/K)',
    'temperature':                   'T (K)',
}

ISO_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def _strip_suffix(header: str) -> str:
    return header.replace('(System_main)', '').strip()


def animate_property_transects(
    features: np.ndarray,
    headers: list,
    property_names: list,
    save_path: str = 'property_animation.gif',
    interval: int = 50,
    n_isentropes: int = 5,
    frame_values: np.ndarray = None,
    frame_label: str = '',
) -> animation.FuncAnimation:
    """Animate Burnman thermodynamic property transects over pressure for selected isentropes.

    Parameters
    ----------
    features       : (N, F) array of input features
    headers        : length-F list of column names, e.g.
                     ['P(GPa)(System_main)', 'S(J/g/K)(System_main)', 'Na2O', 'SiO2', ...]
    property_names : Burnman property names to plot, e.g. ['density', 'p_wave_velocity']
    save_path      : output GIF path
    interval       : milliseconds per animation frame
    n_isentropes   : number of isentropes to overlay per frame (max 5)
    frame_values   : optional 1-D array of length B labelling each animation frame.
                     When supplied, features must contain exactly B equal-sized blocks
                     (rows 0..N/B-1 = frame 0, rows N/B..2N/B-1 = frame 1, …).
                     Frames are sorted and labelled by these values; composition may
                     vary freely across blocks.  Pressure and entropy are still used as
                     x-axis and isentrope selectors respectively.
    frame_label    : axis / title label for the quantity represented by frame_values
                     (e.g. 'mixing fraction', 'XMg').

    Pressure and entropy must both vary. When frame_values is not supplied, at most one
    other column may vary; if present, the animation sweeps over it in ascending order.
    """
    assert 1 <= n_isentropes <= 5, "n_isentropes must be between 1 and 5"
    features = np.asarray(features, dtype=np.float32)
    N = len(features)

    if frame_values is not None:
        frame_values = np.asarray(frame_values)
        if len(frame_values) != N:
            raise ValueError(
                f'frame_values must have the same length as features rows '
                f'(got {len(frame_values)}, expected {N})'
            )
        fv_rounded = np.round(frame_values, 4)
        unique_fv  = np.unique(fv_rounded)
        print(f'[frame_values] {N} rows → {len(unique_fv)} unique frames '
              f'in [{unique_fv.min():.4g}, {unique_fv.max():.4g}]')

    # --- Run emulator for component moles and temperature ---
    t0 = time.time()
    with torch.no_grad():
        output = HeFESToEmulatorCPU.ForwardMB(
            features, headers=headers, outputs=['component_moles', 'temperature']
        )
    print(f'Emulator: {features.shape[0]} assemblages in {time.time() - t0:.2f}s')

    component_moles = output['component_moles']
    if hasattr(component_moles, 'detach'):
        component_moles = component_moles.detach().cpu().numpy()
    component_moles = np.asarray(component_moles, dtype=np.float64)

    T_em = output['temperature']
    if hasattr(T_em, 'detach'):
        T_em = T_em.detach().cpu().numpy()
    T_em = np.asarray(T_em, dtype=np.float64).reshape(-1)

    # --- Find required columns ---
    p_col = next((i for i, h in enumerate(headers) if 'P(GPa)' in h), None)
    s_col = next((i for i, h in enumerate(headers) if 'S(J/g/K)' in h), None)
    if p_col is None:
        raise ValueError("No pressure column found — expected a header containing 'P(GPa)'")
    if s_col is None:
        raise ValueError("No entropy column found — expected a header containing 'S(J/g/K)'")

    P_gpa = features[:, p_col].astype(np.float64)   # GPa, used for x-axis
    P_pa  = P_gpa * 1e9                              # Pa, for Burnman

    # Build (N, 2) PT in SI units [Pa, K]
    PT = np.stack([P_pa, T_em], axis=1)

    # --- Compute Burnman properties ---
    t1 = time.time()
    burnman_raw = HeFESToEmulatorGPU.get_property_burnman_vectorized_from_assemblage(
        torch.tensor(component_moles, dtype=torch.float64, device='cuda'),
        torch.tensor(PT,              dtype=torch.float64, device='cuda'),
        property_names=property_names,
    )
    print(f'Burnman:  {features.shape[0]} assemblages in {time.time() - t1:.2f}s')

    prop_arrays = {
        k: np.asarray(v.detach().cpu() if hasattr(v, 'detach') else v, dtype=np.float64)
        for k, v in burnman_raw.items()
    }

    # --- Detect the one other varying column (besides P and S) ---
    if frame_values is None:
        other_varying = []
        for i, h in enumerate(headers):
            if i in (p_col, s_col):
                continue
            unique_vals = np.unique(np.round(features[:, i], 4))
            if len(unique_vals) > 1:
                other_varying.append((i, _strip_suffix(h), unique_vals))

        if len(other_varying) > 1:
            names = ', '.join(name for _, name, _ in other_varying)
            raise ValueError(
                f'Expected at most 1 other varying feature besides P and S, '
                f'found {len(other_varying)}: {names}'
            )

    # --- Select isentropes evenly spaced across the entropy range ---
    S_jgk    = features[:, s_col]
    S_rounded = np.round(S_jgk, 4)
    unique_S  = np.unique(S_rounded)
    n_iso     = min(n_isentropes, len(unique_S))
    selected_S = unique_S[np.linspace(0, len(unique_S) - 1, n_iso, dtype=int)]
    iso_colors = ISO_COLORS[:n_iso]

    n_props     = len(property_names)
    prop_labels = [PROPERTY_LABELS.get(p, p) for p in property_names]

    # --- Global axis limits (locked for the whole animation) ---
    P_xlim = (float(P_gpa.min()), float(P_gpa.max()))

    def _padded_ylim(arr: np.ndarray, pad: float = 0.04):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo, hi = finite.min(), finite.max()
        margin = (hi - lo) * pad if hi != lo else abs(hi) * pad or 1.0
        return (lo - margin, hi + margin)

    prop_ylims = {name: _padded_ylim(prop_arrays[name]) for name in property_names}

    # --- Drawing helper ---
    def _draw_transects(axes: list, row_mask: np.ndarray) -> None:
        for ax, prop_name, label in zip(axes, property_names, prop_labels):
            ax.cla()
            prop_vals = prop_arrays[prop_name]
            for color, s_val in zip(iso_colors, selected_S):
                mask = row_mask & (S_rounded == s_val)
                if not np.any(mask):
                    continue
                idx = np.where(mask)[0]
                idx = idx[np.argsort(P_gpa[idx])]
                finite = np.isfinite(prop_vals[idx])
                ax.plot(P_gpa[idx][finite], prop_vals[idx][finite],
                        color=color, lw=1.5, label=f'S={s_val:.3f} J/g/K')
            ax.set_xlim(*P_xlim)
            ax.set_ylim(*prop_ylims[prop_name])
            ax.set_xlabel('P (GPa)')
            ax.set_ylabel(label)
            ax.set_title(label, fontsize=9)
            ax.grid(True, alpha=0.25)
        axes[-1].legend(fontsize=7, loc='upper left')

    # --- Build animation ---
    n_cols = 2 if n_props >= 4 else 1
    n_rows = (n_props + n_cols - 1) // n_cols
    fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = np.asarray(axes_arr).reshape(-1).tolist()
    for ax in axes_flat[n_props:]:
        ax.set_visible(False)
    axes = axes_flat[:n_props]

    if frame_values is not None:
        # Treat frame_values like an external "other_varying" column:
        # unique values drive frames; rows matching each value are masked for plotting.
        n_frames = len(unique_fv)

        def update(frame):
            fv_cur   = unique_fv[frame]
            row_mask = fv_rounded == fv_cur
            n_sel    = int(row_mask.sum())
            print(f'  frame {frame+1:>4d}/{n_frames}  {frame_label}={fv_cur:.4g}  '
                  f'rows selected: {n_sel}  '
                  f'P range: [{P_gpa[row_mask].min():.1f}, {P_gpa[row_mask].max():.1f}] GPa  '
                  f'unique S: {np.unique(S_rounded[row_mask])}')
            _draw_transects(axes, row_mask)
            fig.suptitle(
                f'{frame_label} = {fv_cur:.4g}  |  Frame {frame + 1}/{n_frames}',
                fontsize=12,
            )
            fig.tight_layout()
            return []

        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=interval, blit=False, repeat=True,
        )

    elif not other_varying:
        # No third variable — single frame
        _draw_transects(axes, np.ones(len(features), dtype=bool))
        fig.suptitle('Property transects — emulator', fontsize=12)
        fig.tight_layout()

        anim = animation.FuncAnimation(
            fig, lambda _: [], frames=1, interval=interval, blit=False,
        )

    else:
        an_col, an_name, an_vals = other_varying[0]
        an_rounded = np.round(features[:, an_col], 4)
        n_frames   = len(an_vals)

        def update(frame):
            an_v = an_vals[frame]
            _draw_transects(axes, an_rounded == an_v)
            fig.suptitle(
                f'{an_name} = {an_v:.4g}  |  Frame {frame + 1}/{n_frames}',
                fontsize=12,
            )
            fig.tight_layout()
            return []

        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=interval, blit=False, repeat=True,
        )

    plt.close(fig)
    anim.save(save_path, writer='pillow', savefig_kwargs={'facecolor': 'white'})
    print(f'Saved → {save_path}')
    return anim


if __name__ == '__main__':
    # Caracas et al., 2019 pyrolite (fixed composition)
    input_dict = {
        'SiO2': 45.0,
        'Al2O3': 4.45,
        'FeO': 8.05,
        'MgO': 37.8,
        'CaO': 3.55,
    }

    input_dict = { #Caracas et al., 2019
    'Si':24,
    'Al':3,
    'Fe':4,
    'Mg':30,
    'Ca':2,
    'Na':1,
}
        
    DMM_PS = {
        'Si': 3.79222,
        'Mg': 4.88475,
        'Fe': 0.57874,
        'Ca': 0.28734,
        'Al': 0.39685,
        'Na': 0.02197,
        'Cr': 0.03813,
        'O': 14.00450
        }

    basalt= {
        'Si': 4.61116,
        'Mg': 1.33197,
        'Fe': 0.61701,
        'Ca': 1.22855,
        'Al': 1.81090,
        'Na': 0.39533,
        'Cr': 0.00508,
        'O': 15.35841
        }
    
    BSMarsOxides = { #Yoshizaki and McDonough, 2020
        'SiO2':45.5,
        'Al2O3': 3.59,
        'FeO': 13.0, # Fe Speciation ballparked for Fe3 = 0.1Fetot
        'Fe2O3': 2.1,
        'MgO': 31.0,
        'CaO': 2.88,
        'Na2O':0.59,
        'Cr2O3':0.88
    }

    BSMoonOxides = { #Elkins-Tanton et al., 2011
        'SiO2': 47.1,
        'Al2O3': 4,
        'FeO': 11.0,
        'Fe2O3': 1.2,
        'MgO': 31.1,
        'CaO': 3.0,
        'Na2O': 0.5,
        'Cr2O3': 0.3
    }

    BSEarthOxides = { # McDonough and Sun, 1995
        'SiO2':45.0,
        'Al2O3': 4.45,
        'FeO': 7.75, # Fe Speciation ballparked for Fe3 = 0.03 Fetot
        'Fe2O3': 0.35,
        'MgO': 37.8,
        'CaO': 3.55,
        'Na2O':0.36,
        'Cr2O3':0.38
    }


    BSM_fraction = np.linspace(0,1, 100)
  
    input_array = np.zeros((len(BSM_fraction), len(BSMarsOxides)), dtype=np.float32)

    for i, f in enumerate(BSM_fraction):
        input_dict = mix_compositions([BSEarthOxides, BSMarsOxides], [1 - f, f])
        for j, (element, value) in enumerate(input_dict.items()):
            input_array[i, j] = value

    headers = []
    for element, value in input_dict.items():
        headers.append(element)

    print(f"Input Array: {input_array}")
    # Grid: P(300 steps) × S(5 steps) × Na2O(30 steps)
    # → isentropes = S values (5), animation variable = Na2O (30 frames)
    features = grid_sample([[0, 140, 300], [2.3, 2.7, 3]], input_array)
    headers  = ['P(GPa)(System_main)', 'S(J/g/K)(System_main)'] + headers

    property_names = [
        'density', 'p_wave_velocity', 's_wave_velocity', 'temperature',]
        #'isentropic_bulk_modulus_reuss', 'heat_capacity_p_by_mass',
    #]

    animate_property_transects(
        features, headers, property_names,
        save_path='mixture_property_animation.gif',
        n_isentropes=3,
        frame_values=np.tile(BSM_fraction, 3*300),
        frame_label = 'Mars fraction in BSE-BSM Mixture'
    )
