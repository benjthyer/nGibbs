import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
d = np.load(HERE / 'cmp.npz')
P, Th, Te, Vsh, Vse, Z = d['P'], d['Th'], d['Te'], d['Vsh'], d['Vse'], d['depth']
rhoh, rhoe = d['rhoh'], d['rhoe']
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb'
HEF, EMU, RED = '#0b0b0b', '#2a78d6', '#e34948'
plt.rcParams.update({'font.size':9,'axes.edgecolor':MUTED,'axes.linewidth':0.7,
    'xtick.color':INK2,'ytick.color':INK2,'axes.labelcolor':INK,'text.color':INK,
    'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.grid':True,'grid.color':'#eceae5',
    'grid.linewidth':0.6,'axes.axisbelow':True,'legend.frameon':False,'font.family':'DejaVu Sans'})

def window(Vs, edge=5):
    dp = np.median(np.diff(P)); d2 = np.gradient(np.gradient(Vs))/dp**2
    c = d2[edge:-edge]; return int(np.argmax(c)+edge), int(np.argmin(c)+edge), d2
tH, bH, d2H = window(Vsh); tE, bE, d2E = window(Vse)
wH, wE = Z[bH]-Z[tH], Z[bE]-Z[tE]

fig, ax = plt.subplots(1, 4, figsize=(14.2, 3.5))
lim = (23.15, 23.70)
m = (P >= lim[0]) & (P <= lim[1])

for a, (yh, ye, lab) in zip(ax[:3], [(Th, Te, 'Temperature (K)'),
                                     (Vsh, Vse, '$V_s$ (km s$^{-1}$)'),
                                     (d2H, d2E, '$d^2V_s/dP^2$')]):
    a.plot(P[m], yh[m], color=HEF, lw=1.9, solid_capstyle='round')
    a.plot(P[m], ye[m], color=EMU, lw=1.6, ls=(0,(4,2.5)))
    a.set_ylabel(lab); a.set_xlim(*lim)
ax[0].axvspan(P[tH], P[bH], color='#e9e8e2', zorder=0)
ax[1].axvspan(P[tH], P[bH], color='#e9e8e2', zorder=0)
ax[2].axvspan(P[tH], P[bH], color='#e9e8e2', zorder=0)
ax[2].axvspan(P[tE], P[bE], color='#dbe9fb', zorder=0)
ax[0].set_title('a   Isentrope', loc='left', fontsize=9.5, pad=6)
ax[1].set_title('b   Shear velocity', loc='left', fontsize=9.5, pad=6)
ax[2].set_title('c   Window detector', loc='left', fontsize=9.5, pad=6)
ax[0].text(0.04, 0.10, 'HeFESTo', transform=ax[0].transAxes, color=HEF, fontsize=9)
ax[0].text(0.04, 0.02, 'emulator', transform=ax[0].transAxes, color=EMU, fontsize=9)
ax[2].text(P[bH]+0.01, ax[2].get_ylim()[1]*0.80, f'  HeFESTo {wH:.2f} km', fontsize=8, color=INK2)
ax[2].text(P[bH]+0.01, ax[2].get_ylim()[1]*0.60, f'  emulator {wE:.2f} km', fontsize=8, color=EMU)

for k, (h, e, lab, sc) in enumerate([(Th, Te, '$\\Delta T$ (K)', 1)]):
    pass
a = ax[3]
a.axhline(0, color=MUTED, lw=0.8)
a.plot(P, Te-Th, color='#4a3aa7', lw=1.4, label='$T$ (K)')
a.plot(P, 100*(Vse-Vsh)/Vsh, color=RED, lw=1.4, label='$V_s$ (%)')
a.plot(P, 100*(rhoe-rhoh)/rhoh, color='#1baf7a', lw=1.4, label=r'$\rho$ (%)')
a.legend(fontsize=8, loc='upper left'); a.set_ylabel('emulator $-$ HeFESTo')
a.set_title('d   Residuals, full run', loc='left', fontsize=9.5, pad=6)
for a in ax: a.set_xlabel('Pressure (GPa)')

fig.suptitle('Emulator against real HeFESTo — harzburgite (Mg/Si = 1.58), isentrope at S = 2.45 J g$^{-1}$ K$^{-1}$',
             x=0.005, ha='left', fontsize=11.2, y=1.04)
fig.text(0.005, -0.10,
 'HeFESTo run: 501 steps, 20-25 GPa at 10 MPa, entropy held to 2.7e-10 J/g/K. Emulator evaluated at the same element moles, the same entropy and the same pressures.\n'
 'Away from the transition the agreement is close: $T$ rms 2.0 K (max 9.9 K), $\\rho$ rms 0.34%%, $V_s$ rms 0.42%%. Through the transition it is not. Shaded bands in c are the\n'
 '$d^2V_s/dP^2$ windows. HeFESTo resolves the reaction over %.2f km with $\\Delta V_s$ = %.2f%%; the emulator collapses it to %.2f km with $\\Delta V_s$ = %.2f%% and places the base\n'
 '%.1f km too deep, because the binary phase gate turns a finite divariant loop into a step. Latent-heat deficit: %.1f K in HeFESTo, %.1f K in the emulator.'
 % (wH, 100*(Vsh[bH]-Vsh[tH])/Vsh[tH], wE, 100*(Vse[bE]-Vse[tE])/Vse[tE], Z[bE]-Z[bH], 18.3, 43.2),
 ha='left', va='top', fontsize=7.4, color=INK2)
fig.tight_layout()
fig.savefig(str(HERE / 'fig_hefesto_vs_emulator.png'), dpi=200, bbox_inches='tight', facecolor=SURF)
print(f"HeFESTo {wH:.3f} km  emulator {wE:.3f} km  ratio {wH/wE:.1f}x")
