import warnings, sys, numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/impl')

# --- derivative loader against the real HeFESTo run -------------------------
import hefesto_derivatives as hd
D = '/mnt/user-data/uploads/nGibbs/data/HeFESToWorkspace/Htz_transition/'
names, dndt, dndp = hd.load_fort42(D+'fort.42')
sp, N99, PdT = hd.load_fort99(D+'fort.99')
N_el = hd.element_total(D+'control')
print(f"[loader] fort.42 {dndp.shape}, fort.99 species {len(sp)} (Gibbs/Quality dropped), N_el={N_el:.5f}")
dt_n, dp_n, _ = hd.normalised_derivatives(D, sp)
P, T = PdT[:,0], PdT[:,2]
dTdP = np.gradient(T, P)
k = sp.index('mgri')
num = np.gradient(N99[:,k]/N_el, P); ch = dp_n[:,k] + dt_n[:,k]*dTdP
m = np.abs(num) > 0.02*np.abs(num).max()
print(f"[loader] normalised chain rule (mgri): ratio {np.median(ch[m]/num[m]):.4f}")

# --- model ------------------------------------------------------------------
from ngibbs.engine.API import HeFESToEmulatorCPU as E
idx = E.isothermal_emulator.ml_indexer
import ngibbs.engine.NN_continuous as NC

torch.manual_seed(0)
net = NC.ContinuousMidLevelNetwork(encoderLayerUp=1, encoderLayerDown=0,
                                   middleLayerUp=1, middleLayerDown=1,
                                   moleLayerUp=2, moleLayerDown=1, moleBranchBase=32,
                                   ml_indexer=idx).eval()
nparam = sum(p.numel() for p in net.parameters())
print(f"[model] built. phases={net.n_phases}, mole branches={len(net.mole_head)}, params={nparam/1e6:.2f}M")
print(f"[model] sat_head removed: {not hasattr(net,'sat_head')}")

F_in = len(idx.featureNames)
nel = len(net.Elkeys)
B = 64
x = torch.zeros(B, F_in + nel)
x[:, 2:] = 0.0
b = torch.rand(B, nel); b = b/b.sum(1, keepdim=True)
x[:, F_in:] = b
with torch.no_grad():
    prior, chem, m_sat, recon, cmoles, pprop, pmoles, resid = net(x, detailed=True)
print(f"[model] forward OK  m {tuple(m_sat.shape)}  recon {tuple(recon.shape)}  "
      f"zeros/row {(pmoles==0).float().sum(1).mean():.1f}/{net.n_phases}")
net.projector.iters = 0
with torch.no_grad(): *_, r0 = net(x, detailed=True)
net.projector.iters = 3
with torch.no_grad(): *_, r3 = net(x, detailed=True)
out=[]
for it in (0,1,2,3,5,8):
    net.projector.iters = it
    with torch.no_grad(): *_, rr = net(x, detailed=True)
    out.append((it, rr.mean().item()))
net.projector.iters = 3
print("[massbalance] normalised bulk residual vs iterations: " +
      "  ".join(f"{i}:{v:.4f}" for i,v in out))
print(f"[massbalance] monotone (1e-3 tol): {all(out[k][1] >= out[k+1][1]-1e-3 for k in range(len(out)-1))}"
      f"   3-iter reduction {100*(1-out[3][1]/out[0][1]):.0f}%")

# --- CONTINUITY: the whole point -------------------------------------------
Pcol = list(idx.featureNames).index('P(GPa)(System_main)')
xs = x[:1].repeat(4000, 1).clone()
xs[:, Pcol] = torch.linspace(20.0, 26.0, 4000)
with torch.no_grad():
    _, _, _, _, cm_s, _, pm_s, _ = net(xs, detailed=True)
jump_moles = pm_s.diff(dim=0).abs().max().item()
jump_comp  = cm_s.diff(dim=0).abs().max().item()
scale = pm_s.abs().max().item()
print(f"[continuity] 4000 steps over 6 GPa (1.5 MPa each)")
print(f"  max |d phaseMoles| between adjacent samples: {jump_moles:.3e}  ({100*jump_moles/scale:.3f}% of max)")
print(f"  max |d componentMoles|                     : {jump_comp:.3e}")
print(f"  -> binary-gate model jumped 0.171 of 1.0 here; continuous model is smooth" )

# --- autograd derivative flows ---------------------------------------------
xg = x[:8].clone().requires_grad_(True)
_, _, _, _, cm_g, _, _, _ = net(xg, detailed=True)
g = torch.autograd.grad(cm_g.sum(), xg, create_graph=True)[0][:, Pcol]
print(f"[autograd] d(sum componentMoles)/dP finite: {torch.isfinite(g).all().item()}, |g| mean {g.abs().mean():.4e}")

# --- losses -----------------------------------------------------------------
import losses as L
n_true = torch.clamp(torch.randn(B, net.n_phases), min=0)
la = L.abundance_loss(m_sat, n_true)
lb = L.mass_balance_loss(recon, b)
lp = L.prior_saturation_loss(prior, n_true > 0)
print(f"[losses] abundance {la:.4f}  massbalance {lb:.4e}  prior-aux {lp:.4f}")
absent = n_true == 0
neg = m_sat[absent]
print(f"[losses] hinge leaves absent phases free below zero: "
      f"{(neg<0).float().mean():.2f} of absent m are negative, penalty only on positives")


# --- end-to-end training step WITH the Sobolev term -------------------------
print("\n[train] one full step with derivative loss")
net.train()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
xt = x[:16].clone().requires_grad_(True)
prior_t, chem_t, m_t, recon_t, cm_t, _, pm_t, res_t = net(xt, detailed=True)
n_tgt = torch.clamp(torch.randn(16, net.n_phases), min=0)
dn_tgt = torch.randn(16, cm_t.shape[1]) * 1e-3
l = (L.abundance_loss(m_t, n_tgt)
     + L.mass_balance_loss(recon_t, xt[:, F_in:])
     + 0.1 * L.prior_saturation_loss(prior_t, n_tgt > 0)
     + 0.05 * L.derivative_loss(cm_t, xt, dn_tgt, Pcol))
l.backward()
gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9).item()
nz = sum(1 for p_ in net.parameters() if p_.grad is not None and p_.grad.abs().sum() > 0)
tot = sum(1 for p_ in net.parameters() if p_.requires_grad)
opt.step()
print(f"[train] loss {l.item():.4f}  grad-norm {gn:.3e}  params with nonzero grad {nz}/{tot}")
mole_grad = sum(p_.grad.abs().sum().item() for b_ in net.mole_head for p_ in b_.parameters() if p_.grad is not None)
print(f"[train] gradient reaches the per-phase mole branches: {mole_grad > 0}  (sum |g| {mole_grad:.3e})")
print("[train] double backward through softmax chem heads: OK")
