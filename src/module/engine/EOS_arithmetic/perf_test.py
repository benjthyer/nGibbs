"""Performance test: B=2^15 assemblages through hefesto_vec.compute."""
import numpy as np, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from hefesto_vec.params     import load_control
from hefesto_vec.dos_tables import load_tables
from hefesto_vec.compute    import compute

params = load_control('BENCHMARK/control', param_dir_override='HeFESTo_Parameters_010123')
load_tables()
S = params.nspec   # 73
B = 2**15          # 32768
print(f"B=2^15={B}, S={S}")

rng = np.random.default_rng(42)
d   = np.load('/sessions/epic-loving-bardeen/mnt/outputs/bench_PTX.npz')
X_ref = d['X']   # (15, 73)

idx    = rng.integers(0, 15, size=B)
P_test = rng.uniform(0, 140, size=B)
T_test = 1600.0 + P_test * 8.0 + rng.uniform(-200, 200, size=B)
X_test = np.maximum(X_ref[idx], 0.0)
print(f"Mean active species per row: {(X_test>0).sum(axis=1).mean():.1f}")

# Warmup (compile numba / JIT if any)
_ = compute(P_test[:64], T_test[:64], X_test[:64], params)

# Timed runs
times = []
for k in range(3):
    t0 = time.perf_counter()
    res = compute(P_test, T_test, X_test, params)
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  run {k+1}: {dt:.2f}s  ({int(B/dt):,} assemblages/s)")

med = float(np.median(times))
print(f"\nMedian {med:.2f}s  target<10s  -> {'PASS' if med<10 else 'FAIL'}")
print(f"rho  {res['rho'].min():.3f} – {res['rho'].max():.3f} g/cm3")
print(f"Vp   {res['Vp'].min():.2f}  – {res['Vp'].max():.2f}  km/s")
print(f"Vs   {res['Vs'].min():.2f}  – {res['Vs'].max():.2f}  km/s")
conv = res['converged'].all(axis=1).mean()*100
print(f"Fully converged rows: {conv:.1f}%")
