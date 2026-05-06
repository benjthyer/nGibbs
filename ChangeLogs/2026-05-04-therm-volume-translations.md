# 2026-05-04 therm and volume translations

- Added therm_common.py with ThermResult and apar helper accessors.
- Added volume.py solid volume fallback using Murnaghan estimate.
- Added volumel.py liquid/vapor volume fallback with htl branch logic.
- Added volumew.py water volume fallback with Tmin guard and rho model.
- Added volumeh.py hydrogen volume fallback with ideal-gas estimate.
- Added therm.py condensed-phase thermodynamic branch translation scaffold.
- Added therml.py liquid branch blend pending aliq polynomial port.
- Added thermg.py ideal-gas thermodynamic translation from thermg.f.
- Added thermw.py water thermodynamic fallback compatible with volumew.
- Added thermh.py hydrogen thermodynamic fallback branch.
- Updated gspec.py to dispatch by htl to volume* and therm* routines.
- Updated __init__.py lazy exports for therm* and volume* entry points.
- Extended larger-routine tests for gspec htl and new module smoke tests.
- Preserved explicit instability signaling via -1/-2 volume return codes.
- Full helper + larger-routine tests pass: 27 passed, 0 failed.
