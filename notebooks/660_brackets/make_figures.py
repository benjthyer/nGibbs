import numpy as np, json, sys
sys.path.insert(0,'/home/claude/bracket')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bracket_lib as B

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb'
plt.rcParams.update({'font.size':9,'axes.edgecolor':MUTED,'axes.linewidth':0.7,
    'xtick.color':INK2,'ytick.color':INK2,'axes.labelcolor':INK,'text.color':INK,
    'xtick.direction':'out','ytick.direction':'out','figure.facecolor':SURF,
    'axes.facecolor':SURF,'axes.grid':True,'grid.color':'#e6e5e0','grid.linewidth':0.6,
    'axes.axisbelow':True,'legend.frameon':False,'font.family':'DejaVu Sans'})

D = json.load(open('/home/claude/bracket/robust.json'))
r = [x for x in D['rows'] if x['Tp'] <= 1750]
Tp   = np.array([x['Tp'] for x in r])
z    = np.array([x['z'] for x in r])
dT   = np.array([x['dT_dip'] for x in r])
sb   = np.array([x['self_broaden_km'] for x in r])
dVs  = np.array([x['dVs_pct'] for x in r])

# ============================================================ FIGURE 1
fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.5))

a = ax[0]
a.axhline(660, color=MUTED, lw=1.0, ls=(0,(4,3)))
a.text(1302, 659.4, 'PREM 660', ha='left', va='bottom', fontsize=8, color=INK2)
a.plot(Tp, z, color=BLUE, lw=2.0, solid_capstyle='round')
i = int(np.argmin(np.abs(z-660)))
a.plot([Tp[i]],[z[i]], 'o', ms=7, mfc=SURF, mec=BLUE, mew=2.0, zorder=5)
a.annotate(f'660 km at $M_p\\approx${Tp[i]:.0f} K', (Tp[i], z[i]), (14, -22),
           textcoords='offset points', fontsize=8, color=INK2,
           arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7))
a.invert_yaxis(); a.set_xlabel('Mantle potential temperature (K)')
a.set_ylabel('Depth of post-spinel transition (km)')
a.set_title('a   Where the discontinuity sits', loc='left', fontsize=9.5, color=INK, pad=8)

b = ax[1]
b.plot(Tp, dT, color=ORANGE, lw=2.0, solid_capstyle='round')
b.set_xlabel('Mantle potential temperature (K)')
b.set_ylabel('Latent-heat deficit  $\\Delta T$  (K)')
b.set_ylim(0, max(dT)*1.25)
b.text(Tp[-1], dT[-1], f'  {dT[-1]:.0f} K', va='center', fontsize=8, color=INK2)
b.set_title('b   Isentropic cooling across the loop', loc='left', fontsize=9.5, color=INK, pad=8)

c = ax[2]
c.axhspan(0, 2, color='#e8f4ee', zorder=0)
c.text(1305, 0.30, 'seismically resolved sharpness  $\\leq$ 2 km',
       fontsize=7.8, color='#166b4e', va='center')
c.axhline(7, color=MUTED, lw=1.0, ls=(0,(4,3)))
c.text(1745, 7.15, 'Ishii et al. broadening estimate $\\approx$ 7 km',
       ha='right', va='bottom', fontsize=7.8, color=INK2)
c.plot(Tp, sb, color=AQUA, lw=2.2, solid_capstyle='round')
c.text(Tp[-1], sb[-1], f'  {sb[-1]:.1f} km', va='center', fontsize=8, color=INK2)
c.set_ylim(0, 8.2); c.set_xlabel('Mantle potential temperature (K)')
c.set_ylabel('Isentropic self-broadening (km)')
c.set_title('c   Bracket separation: Pe$\\rightarrow\\infty$ minus Pe$\\rightarrow$0',
            loc='left', fontsize=9.5, color=INK, pad=8)

fig.suptitle('Sharpness of the 660-km discontinuity for pyrolite — nGibbs HeFESTo emulator',
             x=0.006, ha='left', fontsize=11, color=INK, y=1.02)
fig.text(0.006, -0.075,
  'Ringwoodite $\\rightarrow$ bridgmanite + ferropericlase along isentropes for BSE (McDonough & Sun 1995), 1 MPa pressure steps ($\\approx$25 m), depths via PREM.\n'
  'Self-broadening = |d$P$/d$T$| $\\Delta T$ / (d$P$/d$z$), with d$P$/d$T$ = %.2f MPa/K fitted from the emulator\'s own rw-out surface. It is the extra thickness the\n'
  'latent-heat dip adds over the conduction-dominated limit; the intrinsic divariant width sits underneath it and is not resolvable in this emulator (see Fig. 2).'
  % (D['clapeyron_GPa_per_K']*1000), ha='left', va='top', fontsize=7.4, color=INK2)
fig.tight_layout()
fig.savefig('/home/claude/bracket/fig1_660_brackets.png', dpi=220, bbox_inches='tight',
            facecolor=SURF)
print('fig1 done')

# ============================================================ FIGURE 2 (diagnostic)
BSE={'SiO2':45.0,'Al2O3':4.45,'FeO':7.75,'Fe2O3':0.35,'MgO':37.8,'CaO':3.55,'Na2O':0.36,'Cr2O3':0.38}
S = B.reference_entropy(BSE, 1600.)
P = np.arange(23.00, 23.30, 0.0002)
T, (ri, bm, fp), pr = B.isentropic_path(BSE, S, P)

fig2, ax2 = plt.subplots(1, 2, figsize=(9.0, 3.4))
g = np.where(np.abs(np.diff(ri)) > 0.02)[0]
gates = [0.5*(P[i]+P[i+1]) for i in g]

a = ax2[0]
a.plot(P, T, color=BLUE, lw=1.8, solid_capstyle='round')
for x in gates: a.axvline(x, color=MUTED, lw=0.9, ls=(0,(3,3)))
a.set_xlabel('Pressure (GPa)'); a.set_ylabel('Temperature (K)')
a.set_title('a   Isentrope through the transition', loc='left', fontsize=9.5, pad=8)
a.text(gates[0], a.get_ylim()[1], ' gate 1 ', fontsize=7.5, color=INK2, va='top')
a.text(gates[-1], a.get_ylim()[1], ' gate 2 ', fontsize=7.5, color=INK2, va='top')

b = ax2[1]
for y, cl, nm in ((ri, BLUE, 'ringwoodite'), (bm, ORANGE, 'bridgmanite'), (fp, AQUA, 'ferropericlase')):
    b.plot(P, y, color=cl, lw=1.8, solid_capstyle='round', label=nm)
    b.text(P[-1], y[-1], '  '+nm, fontsize=7.8, color=INK2, va='center')
for x in gates: b.axvline(x, color=MUTED, lw=0.9, ls=(0,(3,3)))
b.set_xlim(P[0], P[-1] + 0.085)
b.set_xlabel('Pressure (GPa)'); b.set_ylabel('Mole fraction')
b.set_title('b   61% of the reaction happens in two zero-width steps',
            loc='left', fontsize=9.5, pad=8)

fig2.suptitle('Diagnostic: the emulator cannot resolve transition WIDTH', x=0.006, ha='left',
              fontsize=11, color=INK, y=1.03)
fig2.text(0.006, -0.10,
  'Same isentrope at 0.2 MPa steps ($\\approx$5 m of depth). Phase abundances are gated by a hard binary threshold — `binary_pred = (likelihoods > 0.5)` in\n'
  'engine/NN.py — so a phase leaves the assemblage discontinuously instead of decreasing to zero. Transition depth, the total property jumps and the total\n'
  '$\\Delta T$ survive this (endpoint differences of state functions); the width between the gates does not. Absolute widths need HeFESTo run at this spacing.',
  ha='left', va='top', fontsize=7.4, color=INK2)
fig2.tight_layout()
fig2.savefig('/home/claude/bracket/fig2_gate_diagnostic.png', dpi=220, bbox_inches='tight',
             facecolor=SURF)
print('fig2 done; gates at', [round(x,4) for x in gates])
