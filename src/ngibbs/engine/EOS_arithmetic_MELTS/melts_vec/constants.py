"""
Physical constants and unit conventions matching MAGMA's sources/gibbs.c
exactly (see the `static const double r/tr/pr/trl` declarations at the top
of `gibbs()`).

Unit notes
----------
- Temperature in K.
- Pressure in **bars** (not GPa, not Pa) -- this differs from HeFESTo's
  EOS_arithmetic, which works in GPa throughout. 1 GPa = 10000 bars.
- Enthalpy/Gibbs energy in J/mol, entropy/Cp in J/(mol K).
- Volume in J/bar. Because dG/dP = V and P is in bars while G is in J,
  V must be in J/bar to keep units consistent -- numerically this equals
  10 x (volume in cm^3/mol). i.e. V[cm^3/mol] = 10 * V[J/bar].
- Reference state: Tr = 298.15 K, Pr = 1 bar (Berman 1988 standard state).
- Trl = 1673 K is the reference temperature for the *liquid* volume model
  (Kress/Ghiorso-Kress partial molar volumes) -- unrelated to Tr, kept here
  only because it appears in gibbs.c's shared constants and will be needed
  once melts_vec grows a liquid_volume module.
"""

Rgas = 8.3143     # J/(mol K) -- gas constant, exactly as used in gibbs.c
                  # (note: NOT CODATA 8.314472 -- this is the older MELTS-era
                  # value baked into the reference-state fits; using the more
                  # modern constant would introduce a small systematic
                  # mismatch against alphaMELTS output.)

Tr  = 298.15      # K    -- reference temperature (Berman 1988 standard state)
Pr  = 1.0         # bar  -- reference pressure
Trl = 1673.0      # K    -- liquid volume-model reference temperature

BARS_PER_GPA = 10000.0
CM3_PER_MOL_PER_J_PER_BAR = 10.0   # V[cm^3/mol] = 10 * V[J/bar]

# EOS/Cp model type codes (mirrors the #define values in includes/silmin.h;
# kept as small ints here purely for vectorized branch-selection convenience
# -- melts_vec.params translates the source's string tags into these).
EOS_BERMAN = 0
EOS_VINET  = 1
EOS_SAXENA = 2

CP_BERMAN = 0
CP_SAXENA = 1
