import warnings, sys, numpy as np; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/imp')
from ngibbs.engine.API import HeFESToEmulatorCPU as E
from builder.HeFESTo.HeFESTo_derivative_recovery import recover_derivatives, validate_against_fort42, write_fort42
D='/mnt/user-data/uploads/nGibbs/data/HeFESToWorkspace/Htz_transition'
PD='/usr/local/lib/python3.11/dist-packages/ngibbs/engine/EOS_arithmetic/HeFESTo_Parameters_010123'
for ns in (0.0, 1e-6, None):
    kw = {} if ns is None else dict(nsmall_rel=ns)
    dndt, dndp, sn = recover_derivatives(D, E.hefesto_params, PD, **kw)
    v = validate_against_fort42(D, dndt, dndp, sn)
    tag = 'default pruning' if ns is None else f'nsmall_rel={ns}'
    print(f"{tag:18s} species={v['n_species']:3d}  mean_corr={v['mean_corr']:.4f}  "
          f"rms_err={v['rms_error']:.6f}  (signal rms {v['signal_rms']:.4f})")
dndt, dndp, sn = recover_derivatives(D, E.hefesto_params, PD, nsmall_rel=0.0)
write_fort42('/tmp/fort.42.recovered', sn, dndt, dndp)
import os; print(f"\nwrote /tmp/fort.42.recovered ({os.path.getsize('/tmp/fort.42.recovered')/1e6:.2f} MB, "
                 f"original {os.path.getsize(D+'/fort.42')/1e6:.2f} MB)")
