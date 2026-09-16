"""
660-km discontinuity sharpness brackets from the nGibbs HeFESTo emulator.

Two limits on the thickness of the ringwoodite -> bridgmanite + ferropericlase
transition, for one bulk composition swept over mantle potential temperature:

  Pe -> inf  (fast advection, no heat exchange)
      The true isentrope. Latent heat cools the reacting interval; with a
      negative Clapeyron slope the cooling pushes completion to higher P,
      so the transition self-broadens.

  Pe -> 0    (slow advection, conduction resupplies the latent heat)
      Steady state with negligible advection solves d/dz(k dT/dz) = 0, i.e.
      T linear in depth, matched to the reaction-free adiabat. The
      temperature deficit vanishes and the interval collapses toward the
      intrinsic divariant width.

Thickness is measured as the pressure interval over which the reacting
three-phase field (rw + bm + fp) is present: ferropericlase-in to
ringwoodite-out. See README notes on why this, and not a 10-90% velocity
metric, is the robust choice for this emulator.
"""
import warnings, numpy as np, torch
warnings.filterwarnings('ignore')
from ngibbs.engine.models import HeFESToEmulatorCPU as E

LAB = list(E.isothermal_emulator.ml_indexer.label_names)
RI  = [LAB.index('mg-ringwoodite'), LAB.index('fe-ringwoodite')]
BM  = [LAB.index('mg-bridgmanite'), LAB.index('fe-bridgmanite')]
FP  = [LAB.index(n) for n in LAB if n in ('periclase', 'wustite')]
if not FP:
    FP = [i for i, n in enumerate(LAB) if n.startswith(('peri', 'wus'))]

# ---------------------------------------------------------------- PREM depth
def prem_depth_interpolator(path):
    """Return z(P_GPa) in km from the PREM pressure-depth table."""
    raw = np.genfromtxt(path, delimiter=',', skip_header=1)
    z, p = raw[:, 0], raw[:, 1] / 10.0          # kb -> GPa
    o = np.argsort(p)
    p, z = p[o], z[o]
    keep = np.concatenate([[True], np.diff(p) > 0])
    return lambda P: np.interp(P, p[keep], z[keep])

# ------------------------------------------------------------ emulator calls
def _headers(ox, second):
    return ['P(GPa)(System_main)', second] + list(ox)

def reference_entropy(comp, Tp):
    """Entropy (J/g/K) of the composition at 1 bar and potential temperature Tp."""
    ox = list(comp); vals = list(comp.values())
    f = np.array([[1e-4, Tp] + vals], dtype=np.float32)
    cm = E.ForwardMB(f, headers=_headers(ox, 'T(K)(System_main)'),
                     outputs=['component_moles'])['component_moles'].to(torch.float64)
    PT = torch.tensor([[1e-4, Tp]], dtype=torch.float64)
    return float(E.get_property_hefesto_vectorized_from_assemblage(cm, PT, ('S',))['S'][0])

def _fractions(cm):
    a = cm.numpy() if torch.is_tensor(cm) else np.asarray(cm)
    tot = np.clip(a.sum(1, keepdims=True), 1e-30, None)
    f = a / tot
    return f[:, RI].sum(1), f[:, BM].sum(1), f[:, FP].sum(1)

def isentropic_path(comp, S, P, props=('rho', 'Vp', 'Vs')):
    """Sweep entropy S across pressure grid P. Returns T, phase fractions, properties."""
    ox = list(comp); vals = list(comp.values())
    f = np.zeros((len(P), len(ox) + 2), dtype=np.float32)
    f[:, 0] = P; f[:, 1] = S; f[:, 2:] = vals
    out = E.ForwardMB(f, headers=_headers(ox, 'S(J/g/K)(System_main)'),
                      outputs=['component_moles', 'temperature'])
    T = out['temperature'].detach().numpy().ravel()
    cm = out['component_moles'].to(torch.float64)
    pr = E.get_property_hefesto_vectorized_from_assemblage(
        cm, torch.tensor(np.stack([P, T], 1), dtype=torch.float64), tuple(props))
    return T, _fractions(cm), pr

def isothermal_path(comp, T, P, props=('rho', 'Vp', 'Vs')):
    """Equilibrium along a prescribed T(P) path. Returns phase fractions, properties."""
    ox = list(comp); vals = list(comp.values())
    f = np.zeros((len(P), len(ox) + 2), dtype=np.float32)
    f[:, 0] = P; f[:, 1] = T; f[:, 2:] = vals
    out = E.ForwardMB(f, headers=_headers(ox, 'T(K)(System_main)'),
                      outputs=['component_moles'])
    cm = out['component_moles'].to(torch.float64)
    pr = E.get_property_hefesto_vectorized_from_assemblage(
        cm, torch.tensor(np.stack([P, T], 1), dtype=torch.float64), tuple(props))
    return _fractions(cm), pr

# ------------------------------------------------------- field-edge detection
def field_edges(P, ri, fp, fp_tol=1e-6, ri_tol=1e-6):
    """
    Pressure bounds of the reacting rw + bm + fp field.
    P_in  : ferropericlase-in  (last P with fp == 0, first with fp > 0)
    P_out : ringwoodite-out    (last P with ri > 0)
    Returned as midpoints of the bracketing samples.
    """
    has_fp = fp > fp_tol
    has_ri = ri > ri_tol
    both = has_fp & has_ri
    if not both.any():
        return None, None
    i0 = np.argmax(both)
    i1 = len(both) - 1 - np.argmax(both[::-1])
    P_in = P[i0] - 0.5 * (P[i0] - P[i0 - 1]) if i0 > 0 else P[i0]
    P_out = P[i1] + 0.5 * (P[i1 + 1] - P[i1]) if i1 + 1 < len(P) else P[i1]
    return float(P_in), float(P_out)

def background_gradient(P, T, ri, fp, pad=0.05, span=0.5):
    """
    Reaction-free dT/dP (K/GPa), fit in clean windows on both sides of the
    reacting field where no rw+bm+fp coexistence is present.
    """
    P_in, P_out = field_edges(P, ri, fp)
    lo = (P > P_out + pad) & (P < P_out + pad + span)
    hi = (P < P_in - pad) & (P > P_in - pad - span)
    grads = [np.polyfit(P[m], T[m], 1)[0] for m in (lo, hi) if m.sum() > 5]
    return float(np.mean(grads)) if grads else np.nan
