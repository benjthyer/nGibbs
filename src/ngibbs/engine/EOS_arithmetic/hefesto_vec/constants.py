"""
Physical constants matching HeFESTo const.inc exactly.

From const.inc:
    cspeed = 2.99792458d10     ! cm/s
    hplanck = 6.6260693d-34    ! J·s
    boltzk = 1.3806505d-23     ! J/K
    avn = 6.0221415d23         ! mol⁻¹
    hcok = hplanck*cspeed/boltzk
    Rgas = 8.314472            ! J/(mol·K)
    pirad = 4*atan(1.0)

Unit notes
----------
- hcok = 1.4387752... K·cm  (converts cm⁻¹ → K for ws/we/wou/wol mode frequencies)
- Rgas in J/(mol·K)
- Pressures in GPa throughout
- Volumes in cm³/mol
- Energies in J/mol (except F0 stored in kJ/mol → converted on load)
- The factor 0.001 converts J/cm³ → GPa (since 1 J/cm³ = 1 kJ/L = 1e-3 GPa)
"""

import math

# Spectroscopic / thermodynamic constants (exactly as in const.inc)
cspeed  = 2.99792458e10   # cm/s — speed of light
hplanck = 6.6260693e-34   # J·s  — Planck constant
boltzk  = 1.3806505e-23   # J/K  — Boltzmann constant
avn     = 6.0221415e23    # mol⁻¹ — Avogadro number

# Derived constants
hcok = hplanck * cspeed / boltzk   # ≈ 1.4387752 K·cm  (h*c/k_B)
Rgas = 8.314472                    # J/(mol·K)  gas constant
pirad = 4.0 * math.atan(1.0)       # π (set as in const.f)

# Unit conversion factor
J_PER_CM3_TO_GPA = 1.0e-3  # 1 J/cm³ = 1e-3 GPa

# vsquaredminimum: frequency cutoff used in parset.f volume bounds
VSQUAREDMINIMUM = 0.01   # threshold for real vibrational frequency

# Small numerical quantities
VSMALL = 1.0e-10         # parset.f vsmall
