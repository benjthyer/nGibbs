"""Figure 2 analogue for the harzburgite composition (data/HeFESToWorkspace/
Htz_transition), with the real HeFESTo ground truth (fort.56 thermodynamics,
fort.99 phase moles) overlaid on the emulator's two Pe brackets.

Ground truth is isentropic only -- HeFESTo was never run in a conductive-limit
mode here, so there is no real counterpart for the dashed Pe -> 0 branch. It
is fetched independently of the emulator brackets (fort.56 / fort.99 parsed
directly, same as htz_compare.py / load_fort42_check.py) and plotted as a
dotted marker trace alongside the emulator's solid/dashed pair.
"""
import warnings, re, sys
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))            # bracket_lib.py is one level up
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import bracket_lib as B

REPO = HERE.parents[2]
D = str(REPO / 'data' / 'HeFESToWorkspace' / 'Htz_transition') + '/'

BLUE, ORANGE, AQUA, YELL = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
VIOL, RED = '#4a3aa7', '#e34948'
INK, INK2, MUTED, GATE, SURF = '#0b0b0b', '#52514e', '#8a8880', '#d6d5ce', '#fcfcfb'
DASH = (0, (4.5, 2.5))
plt.rcParams.update({'font.size':9,'axes.edgecolor':MUTED,'axes.linewidth':0.7,
    'xtick.color':INK2,'ytick.color':INK2,'axes.labelcolor':INK,'text.color':INK,
    'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.grid':True,
    'grid.color':'#eceae5','grid.linewidth':0.6,'axes.axisbelow':True,
    'legend.frameon':False,'font.family':'DejaVu Sans'})

# ---------------------------------------------------------------- ground truth
hdr = open(D+'fort.56').readlines()[1].split()
raw = np.genfromtxt(D+'fort.56', skip_header=2, usecols=range(len(hdr)-1))
H = {k: raw[:, i] for i, k in enumerate(hdr[:-1])}
Pgt, Tgt = H['P(GPa)'], H['T(K)']
S0 = float(H['S(J/g/K)'].mean())

lines = [l.rstrip('\n') for l in open(D+'control')]
els = {}
for l in lines[3:11]:
    p = l.split(); els[p[0]] = float(p[1])
EL = ['Si','Mg','Fe','Ca','Al','Na','Cr','O']
comp = {k: els[k] for k in EL}
MGSI = els['Mg'] / els['Si']

num = re.compile(r'^[\s\-\d\.eEdD\+]+$')
L99 = open(D+'fort.99').readlines(); cols = L99[0].split()
rows99 = [l for l in L99[1:] if num.match(l.rstrip('\n')) and len(l.split()) == len(cols)]
A99 = np.array([[float(x) for x in l.split()] for l in rows99])
sp = cols[3:]
Ngt = A99[:, 3:]
Pgt99 = A99[:, 0]
# fort.99 carries 75 columns for 73 species: 'Gibbs' (free energy) and
# 'Quality' (a diagnostic metric) are not mole numbers and must be excluded
# from the normalising total, or every fraction is corrupted.
_species_idx = [i for i, c in enumerate(sp) if c not in ('Gibbs', 'Quality')]
tot_gt = np.clip(Ngt[:, _species_idx].sum(1), 1e-30, None)

def gt_group(*names):
    """Same-named-endmember sum, normalised per-row like the emulator's own
    _fractions -- only the mg/fe endmembers the emulator itself tracks for
    each phase, not HeFESTo's full garnet/perovskite solid solution."""
    idx = [sp.index(n) for n in names if n in sp]
    return Ngt[:, idx].sum(1) / tot_gt

ri_gt, bm_gt, fp_gt, mj_gt = (gt_group('mgri', 'feri'), gt_group('mgpv', 'fepv'),
                              gt_group('pe', 'wu'), gt_group('mgmj', 'namj'))
GT = dict(ri=ri_gt, bm=bm_gt, fp=fp_gt, mj=mj_gt)

# ------------------------------------------------------------------ emulator
E = B.E
MJ = [B.LAB.index('mg-majorite'), B.LAB.index('na-majorite')]
OXH = list(comp)

def majorite(P, T):
    f = np.zeros((len(P), len(OXH)+2), dtype=np.float32)
    f[:, 0] = P; f[:, 1] = T; f[:, 2:] = list(comp.values())
    cm = E.ForwardMB(f, headers=['P(GPa)(System_main)', 'T(K)(System_main)']+OXH,
                     outputs=['component_moles'])['component_moles'].numpy()
    return cm[:, MJ].sum(1) / np.clip(cm.sum(1), 1e-30, None)

# Locate the reacting field once, over the full range HeFESTo covers, so WIDE
# and ZOOM below don't need a hand-tuned P window per composition.
Pf = np.arange(Pgt.min(), Pgt.max(), 0.001)
Tf, (rif, bmf, fpf), _ = B.isentropic_path(comp, S0, Pf, props=('rho',))
P_in, P_out = B.field_edges(Pf, rif, fpf)
GAMMA = B.background_gradient(Pf, Tf, rif, fpf)
T_at_in = float(np.interp(P_in, Pf, Tf))

def run(p0, p1, dp):
    P = np.arange(p0, p1, dp)
    T_isen, _, _ = B.isentropic_path(comp, S0, P, props=('rho',))
    T_cond = np.where(P < P_in, T_isen, T_at_in + GAMMA*(P - P_in))
    out = {}
    for key, T in (('isen', T_isen), ('cond', T_cond)):
        (ri, bm, fp), pr = B.isothermal_path(comp, T.astype(np.float32), P)
        out[key] = dict(T=T, ri=ri, bm=bm, fp=fp, mj=majorite(P, T.astype(np.float32)),
                        Vp=pr['Vp'], Vs=pr['Vs'])
    return P, out

WIDE = run(float(Pgt.min()), float(Pgt.max()), 0.005)
zc = 0.5 * (P_in + P_out)
ZOOM = run(zc - 0.15, zc + 0.15, 0.0002)

def gates(P, y, thr=0.02):
    return [0.5*(P[i]+P[i+1]) for i in np.where(np.abs(np.diff(y)) > thr)[0]]

def spread(vals, lo, hi, gap):
    order = np.argsort(vals); out = np.array(vals, dtype=float)
    for k in range(1, len(order)):
        i, j = order[k-1], order[k]
        if out[j] - out[i] < gap: out[j] = out[i] + gap
    return np.clip(out, lo, hi)

zP = B.prem_depth_interpolator(str(REPO / 'recipes' / 'PrEMPressureDepth.csv'))

def depth_axis(ax, P0, P1):
    ta = ax.secondary_xaxis('top', functions=(
        lambda p: zP(p),
        lambda z: np.interp(z, [float(zP(P0)), float(zP(P1))], [P0, P1])))
    z0, z1 = float(zP(P0)), float(zP(P1))
    step = 10 if (z1-z0) > 40 else (2 if (z1-z0) > 8 else 1)
    ticks = np.arange(np.ceil(z0/step)*step, z1, step)
    ta.set_xticks(ticks)
    ta.set_xlabel('Depth (km, PREM)', fontsize=8, labelpad=4)
    ta.tick_params(labelsize=7.5)

PHASES = (('ri', BLUE, 'ringwoodite'), ('bm', ORANGE, 'bridgmanite'),
          ('fp', AQUA, 'ferropericlase'), ('mj', YELL, 'majorite'))

fig = plt.figure(figsize=(12.8, 8.6))
sfigs = fig.subfigures(3, 1, height_ratios=[0.14, 1, 1], hspace=0.06)
head = sfigs[0]; head.set_facecolor(SURF)
blocks = [(WIDE, f'Wide -- {Pgt.max()-Pgt.min():.1f} GPa (HeFESTo run span)', 'abc', sfigs[1]),
          (ZOOM, 'Zoom -- 0.30 GPa', 'def', sfigs[2])]

for (data, rowlab, tags, sf) in blocks:
    sf.set_facecolor(SURF)
    gs = sf.add_gridspec(2, 3, width_ratios=[1, 1, 0.95], hspace=0.16, wspace=0.30)
    a  = sf.add_subplot(gs[:, 0])
    b  = sf.add_subplot(gs[:, 1])
    cp = sf.add_subplot(gs[0, 2])
    cs = sf.add_subplot(gs[1, 2], sharex=cp)
    P, Dd = data
    m_gt   = (Pgt   >= P[0]) & (Pgt   <= P[-1])
    m_gt99 = (Pgt99 >= P[0]) & (Pgt99 <= P[-1])
    G = sorted(set(gates(P, Dd['isen']['ri']) + gates(P, Dd['cond']['ri'])))
    span = P[-1] - P[0]

    a.plot(P, Dd['isen']['T'], color=BLUE, lw=1.8, solid_capstyle='round')
    a.plot(P, Dd['cond']['T'], color=BLUE, lw=1.5, ls=DASH, alpha=0.9)
    a.plot(Pgt[m_gt], Tgt[m_gt], color=INK, lw=0, marker='o', ms=2.4, alpha=0.85)
    a.set_ylabel('Temperature (K)'); a.set_xlabel('Pressure (GPa)')
    a.set_title(tags[0] + '   Thermal path', loc='left', fontsize=9.5, pad=18)

    ends = spread([Dd['isen'][k][-1] for k, _, _ in PHASES], -0.02, 0.80, 0.062)
    for (k, cl, nm), ye in zip(PHASES, ends):
        b.plot(P, Dd['isen'][k], color=cl, lw=1.7, solid_capstyle='round')
        b.plot(P, Dd['cond'][k], color=cl, lw=1.4, ls=DASH, alpha=0.9)
        b.plot(Pgt99[m_gt99], GT[k][m_gt99], color=cl, lw=0, marker='o', ms=2.4,
               alpha=0.8, markeredgecolor=INK, markeredgewidth=0.3)
        b.text(P[-1] + 0.025*span, ye, ' '+nm, fontsize=7.7, color=INK2, va='center')
    b.set_ylabel('Mole fraction'); b.set_ylim(-0.05, 0.84)
    b.set_xlim(P[0], P[-1] + 0.34*span); b.set_xlabel('Pressure (GPa)')
    b.set_title(tags[1] + '   Phase assemblage', loc='left', fontsize=9.5, pad=18)

    gt_v = {'Vp': H['VP(km/s)'], 'Vs': H['VS(km/s)']}
    for ax, k, cl, nm in ((cp, 'Vp', VIOL, '$V_p$'), (cs, 'Vs', RED, '$V_s$')):
        ax.plot(P, Dd['isen'][k], color=cl, lw=1.8, solid_capstyle='round')
        ax.plot(P, Dd['cond'][k], color=cl, lw=1.5, ls=DASH, alpha=0.9)
        ax.plot(Pgt[m_gt], gt_v[k][m_gt], color=INK, lw=0, marker='o', ms=2.4, alpha=0.85)
        ax.set_xlim(P[0], P[-1] + 0.13*span)
        ax.text(P[-1]+0.02*span, Dd['isen'][k][-1], ' '+nm, fontsize=9.5, color=INK2, va='center')
        ax.set_ylabel(nm + '  (km s$^{-1}$)', fontsize=8.5)
    cp.tick_params(labelbottom=False)
    cs.set_xlabel('Pressure (GPa)')
    cp.set_title(tags[2] + '   Seismic velocities', loc='left', fontsize=9.5, pad=18)

    for ax in (a, b, cp, cs):
        for x in G: ax.axvline(x, color=GATE, lw=1.1, zorder=0)
    for ax in (a, b, cp): depth_axis(ax, P[0], P[-1])
    sf.text(0.006, 0.5, rowlab, rotation=90, va='center', ha='left',
            fontsize=9.2, color=INK)

handles = [Line2D([], [], color=INK2, lw=1.8, label='Pe $\\rightarrow\\infty$   isentrope (emulator)'),
           Line2D([], [], color=INK2, lw=1.5, ls=DASH, label='Pe $\\rightarrow$ 0   conductive (emulator)'),
           Line2D([], [], color=INK, lw=0, marker='o', ms=4, label='HeFESTo ground truth (isentropic only)'),
           Line2D([], [], color=GATE, lw=2.2, label='binary phase gate')]
head.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.03, 0.02),
            ncol=4, fontsize=8.6, handlelength=2.8, columnspacing=1.6)
head.text(0.005, 0.86,
    f'The 660 at two scales, harzburgite (Mg/Si = {MGSI:.2f}) -- emulator brackets vs real HeFESTo, S = {S0:.3f} J/g/K  (nGibbs HeFESTo emulator)',
    ha='left', va='top', fontsize=11.2, color=INK)
fig.text(0.005, 0.002,
 "Emulator brackets as in fig2 (pyrolite): solid = isentrope (Pe->inf, latent heat retained), dashed = conductive limit (Pe->0), both fetched from bracket_lib.\n"
 "HeFESTo ground truth (dotted markers) is fort.56 / fort.99 from the real run at S = %.3f J/g/K, %.1f-%.1f GPa at 10 MPa steps, parsed independently of the emulator --\n"
 "it is an isentrope, so it has no conductive-limit counterpart here. Phase ground truth sums only the same-named endmembers the emulator tracks (mgri+feri, mgpv+fepv, pe+wu,\n"
 "mgmj+namj), normalised by the full 73-species row total, so it is not a full accounting of HeFESTo's garnet/perovskite solid solution -- see htz/README_Htz.md."
 % (S0, Pgt.min(), Pgt.max()), ha='left', va='top', fontsize=7.3, color=INK2, transform=fig.transFigure)
fig.savefig(str(HERE / 'fig2_htz_gate_diagnostic.png'), dpi=200,
            bbox_inches='tight', pad_inches=0.62, facecolor=SURF)

for nm, (P, Dd) in (('wide', WIDE), ('zoom', ZOOM)):
    print(f"{nm}: isen {[round(x,4) for x in gates(P,Dd['isen']['ri'])]}  "
          f"cond {[round(x,4) for x in gates(P,Dd['cond']['ri'])]}")
print(f"P_in={P_in:.4f}  P_out={P_out:.4f} GPa   GAMMA={GAMMA:.2f} K/GPa   "
      f"S0={S0:.4f} J/g/K   Mg/Si={MGSI:.4f}")
