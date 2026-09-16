"""Figure 2: the post-spinel transition at two pressure scales (x10 apart),
with BOTH bracket paths overlaid.

  solid   Pe -> inf : the true isentrope. Latent heat cools the reacting
                      interval; the negative Clapeyron slope pushes completion
                      deeper, so the transition self-broadens.
  dashed  Pe -> 0   : conduction resupplies the latent heat. T is linear in
                      depth at the reaction-free adiabatic gradient. Below the
                      transition this path stays hotter by the latent-heat
                      deficit, because heat was added -- it is not an isentrope.

Assemblage and velocities for BOTH paths are evaluated with the SAME (isothermal)
network at prescribed (P, T), so the difference between them is physics rather
than inter-network decision-surface bias.
"""
import numpy as np, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import bracket_lib as B

BLUE, ORANGE, AQUA, YELL = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
VIOL, RED = '#4a3aa7', '#e34948'
INK, INK2, MUTED, GATE, SURF = '#0b0b0b', '#52514e', '#8a8880', '#d6d5ce', '#fcfcfb'
DASH = (0, (4.5, 2.5))
plt.rcParams.update({'font.size':9,'axes.edgecolor':MUTED,'axes.linewidth':0.7,
    'xtick.color':INK2,'ytick.color':INK2,'axes.labelcolor':INK,'text.color':INK,
    'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.grid':True,
    'grid.color':'#eceae5','grid.linewidth':0.6,'axes.axisbelow':True,
    'legend.frameon':False,'font.family':'DejaVu Sans'})

BSE={'SiO2':45.0,'Al2O3':4.45,'FeO':7.75,'Fe2O3':0.35,'MgO':37.8,'CaO':3.55,'Na2O':0.36,'Cr2O3':0.38}
OX = list(BSE); VALS = list(BSE.values())
S  = B.reference_entropy(BSE, 1600.)
zP = B.prem_depth_interpolator(str(HERE.parents[1] / 'recipes' / 'PrEMPressureDepth.csv'))
MJ = [B.LAB.index('mg-majorite'), B.LAB.index('na-majorite')]

def majorite(P, T):
    f = np.zeros((len(P), len(OX)+2), dtype=np.float32)
    f[:,0]=P; f[:,1]=T; f[:,2:]=VALS
    cm = B.E.ForwardMB(f, headers=['P(GPa)(System_main)','T(K)(System_main)']+OX,
                       outputs=['component_moles'])['component_moles'].numpy()
    return cm[:,MJ].sum(1)/np.clip(cm.sum(1),1e-30,None)

Pf = np.arange(22.2, 24.4, 0.001)
Tf, (rif, bmf, fpf), _ = B.isentropic_path(BSE, S, Pf, props=('rho',))
P_in, P_out = B.field_edges(Pf, rif, fpf)
deep = (Pf > P_out+0.20) & (Pf < P_out+0.70)
GAMMA = float(np.polyfit(Pf[deep], Tf[deep], 1)[0])
T_at_in = float(np.interp(P_in, Pf, Tf))

def run(p0, p1, dp):
    P = np.arange(p0, p1, dp)
    T_isen, _, _ = B.isentropic_path(BSE, S, P, props=('rho',))
    T_cond = np.where(P < P_in, T_isen, T_at_in + GAMMA*(P - P_in))
    out = {}
    for key, T in (('isen', T_isen), ('cond', T_cond)):
        (ri, bm, fp), pr = B.isothermal_path(BSE, T.astype(np.float32), P)
        out[key] = dict(T=T, ri=ri, bm=bm, fp=fp, mj=majorite(P, T.astype(np.float32)),
                        Vp=pr['Vp'], Vs=pr['Vs'])
    return P, out

WIDE = run(21.80, 24.80, 0.001)     # 3.00 GPa  ~74 km
ZOOM = run(23.15, 23.45, 0.0002)    # 0.30 GPa  ~7.4 km

def gates(P, y, thr=0.02):
    return [0.5*(P[i]+P[i+1]) for i in np.where(np.abs(np.diff(y)) > thr)[0]]

def spread(vals, lo, hi, gap):
    """Nudge end-of-line label positions apart so they never overlap."""
    order = np.argsort(vals); out = np.array(vals, dtype=float)
    for k in range(1, len(order)):
        i, j = order[k-1], order[k]
        if out[j] - out[i] < gap: out[j] = out[i] + gap
    return np.clip(out, lo, hi)

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

PHASES = (('ri',BLUE,'ringwoodite'), ('bm',ORANGE,'bridgmanite'),
          ('fp',AQUA,'ferropericlase'), ('mj',YELL,'majorite'))

fig = plt.figure(figsize=(12.8, 8.6))
sfigs = fig.subfigures(3, 1, height_ratios=[0.14, 1, 1], hspace=0.06)
head = sfigs[0]; head.set_facecolor(SURF)
blocks = [(WIDE, 'Wide — 3.0 GPa ($\\approx$74 km), 1 MPa steps', 'abc', sfigs[1]),
          (ZOOM, 'Zoom $\\times$10 — 0.30 GPa ($\\approx$7.4 km), 0.2 MPa steps', 'def', sfigs[2])]

for (data, rowlab, tags, sf) in blocks:
    sf.set_facecolor(SURF)
    gs = sf.add_gridspec(2, 3, width_ratios=[1, 1, 0.95], hspace=0.16, wspace=0.30)
    a  = sf.add_subplot(gs[:, 0])
    b  = sf.add_subplot(gs[:, 1])
    cp = sf.add_subplot(gs[0, 2])
    cs = sf.add_subplot(gs[1, 2], sharex=cp)
    P, D = data
    G = sorted(set(gates(P, D['isen']['ri']) + gates(P, D['cond']['ri'])))
    span = P[-1] - P[0]

    a.plot(P, D['isen']['T'], color=BLUE, lw=1.8, solid_capstyle='round')
    a.plot(P, D['cond']['T'], color=BLUE, lw=1.5, ls=DASH, alpha=0.9)
    a.set_ylabel('Temperature (K)'); a.set_xlabel('Pressure (GPa)')
    a.set_title(tags[0] + '   Thermal path', loc='left', fontsize=9.5, pad=18)
    j = -1
    a.annotate('', (P[j], D['isen']['T'][j]), (P[j], D['cond']['T'][j]),
               arrowprops=dict(arrowstyle='<->', color=INK2, lw=0.8))
    a.text(P[j]-0.012*span, 0.5*(D['isen']['T'][j]+D['cond']['T'][j]),
           f"$\\Delta T$ = {D['cond']['T'][j]-D['isen']['T'][j]:.0f} K  ",
           fontsize=8, color=INK2, ha='right', va='center')

    ends = spread([D['isen'][k][-1] for k, _, _ in PHASES], -0.02, 0.80, 0.062)
    for (k, cl, nm), ye in zip(PHASES, ends):
        b.plot(P, D['isen'][k], color=cl, lw=1.7, solid_capstyle='round')
        b.plot(P, D['cond'][k], color=cl, lw=1.4, ls=DASH, alpha=0.9)
        b.text(P[-1] + 0.025*span, ye, ' '+nm, fontsize=7.7, color=INK2, va='center')
    b.set_ylabel('Mole fraction'); b.set_ylim(-0.05, 0.84)
    b.set_xlim(P[0], P[-1] + 0.34*span); b.set_xlabel('Pressure (GPa)')
    b.set_title(tags[1] + '   Phase assemblage', loc='left', fontsize=9.5, pad=18)

    for ax, k, cl, nm in ((cp,'Vp',VIOL,'$V_p$'), (cs,'Vs',RED,'$V_s$')):
        ax.plot(P, D['isen'][k], color=cl, lw=1.8, solid_capstyle='round')
        ax.plot(P, D['cond'][k], color=cl, lw=1.5, ls=DASH, alpha=0.9)
        ax.set_xlim(P[0], P[-1] + 0.13*span)
        ax.text(P[-1]+0.02*span, D['isen'][k][-1], ' '+nm, fontsize=9.5, color=INK2, va='center')
        ax.set_ylabel(nm + '  (km s$^{-1}$)', fontsize=8.5)
    cp.tick_params(labelbottom=False)
    cs.set_xlabel('Pressure (GPa)')
    cp.set_title(tags[2] + '   Seismic velocities', loc='left', fontsize=9.5, pad=18)

    for ax in (a, b, cp, cs):
        for x in G: ax.axvline(x, color=GATE, lw=1.1, zorder=0)
    for ax in (a, b, cp): depth_axis(ax, P[0], P[-1])
    sf.text(0.006, 0.5, rowlab, rotation=90, va='center', ha='left',
            fontsize=9.2, color=INK)

handles = [Line2D([],[], color=INK2, lw=1.8, label='Pe $\\rightarrow\\infty$   isentrope (latent heat retained)'),
           Line2D([],[], color=INK2, lw=1.5, ls=DASH, label='Pe $\\rightarrow$ 0   conductive (latent heat resupplied)'),
           Line2D([],[], color=GATE, lw=2.2, label='binary phase gate')]
head.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.055, 0.02),
            ncol=3, fontsize=8.8, handlelength=2.8, columnspacing=2.4)
head.text(0.005, 0.86, 'The 660 at two scales, both bracket limits — pyrolite, $M_p$ = 1600 K  (nGibbs HeFESTo emulator)',
          ha='left', va='top', fontsize=11.8, color=INK)
fig.text(0.005, 0.002,
 'Both paths are evaluated with the same isothermal network at prescribed ($P$, $T$), so the solid–dashed difference is physics, not inter-network bias. The conductive path is the reaction-free adiabat\n'
 'extrapolated at %.1f K/GPa from the ferropericlase-in boundary; below the transition it stays hotter by the latent-heat deficit, because heat was added — it is not an isentrope. The isentrope completes\n'
 'the reaction $\\approx$0.06 GPa ($\\approx$1.5 km) deeper, which is the bracket separation. Vertical lines are binary phase gates: `binary_pred = (likelihoods > 0.5)` in engine/NN.py zeroes a phase discontinuously\n'
 'instead of letting it decrease to zero. They are still zero-width at 0.2 MPa ($\\approx$5 m of depth), so $V_p$ and $V_s$ inherit the steps and any 10–90%% velocity width metric measures the classifier rather\n'
 'than the thermodynamics. Top row: the post-spinel step rides on the broader post-garnet ramp, where majorite dissolves into bridgmanite from $\\approx$22 GPa. $V_s$ is unaffected by order-disorder\n'
 'relaxation; $V_p$ is the isomorphic value, $\\approx$0.26%% high against fort.56 in the lower mantle. Note that the step in the thermal path sits $\\approx$0.09 GPa above the gates in the other panels: $T$ comes\n'
 'from the isentropic network while assemblage and velocities come from the isothermal one, so the offset is a direct view of the inter-network decision-surface bias, and is why both paths here share one network.'
 % GAMMA, ha='left', va='top', fontsize=7.3, color=INK2, transform=fig.transFigure)
fig.savefig(str(HERE / 'fig2_gate_diagnostic.png'), dpi=200,
            bbox_inches='tight', pad_inches=0.62, facecolor=SURF)

for nm, (P, D) in (('wide',WIDE), ('zoom',ZOOM)):
    print(f"{nm}: isen {[round(x,4) for x in gates(P,D['isen']['ri'])]}  "
          f"cond {[round(x,4) for x in gates(P,D['cond']['ri'])]}")
gi = gates(ZOOM[0], ZOOM[1]['isen']['ri']); gc = gates(ZOOM[0], ZOOM[1]['cond']['ri'])
if gi and gc:
    print("separation: %.4f GPa = %.2f km" % (max(gi)-max(gc), float(zP(max(gi))-zP(max(gc)))))
else:
    print("separation: no discrete >0.02 ringwoodite step found in isen and/or cond path "
          "(transition is no longer a hard binary-gate step in this model -- see field_edges-based "
          "P_in/P_out in bracket_lib for a threshold-free measure of the reacting field width).")
