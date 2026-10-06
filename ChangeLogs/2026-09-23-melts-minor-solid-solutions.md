2026-09-23
- melts_vec: new solid-solution models, verbatim from MAGMA: garnet.py
  (garnet.c), leucite.py (leucite.c), biotite.py (biotite.c),
  hornblende.py (hornblende.c), nepheline.py (nepheline.c, incl. its K-Na
  ordering parameter and order() solve). One entry point:
  solid_solutions.compute_solution(phase, T, P, X, solid_params).
- biotiteTaj.c not translated: no MELTS solid table uses it. kalsilite and
  melilite are out of nGibbs's scope.
- compute.py: vc-/ca-nepheline endmember special cases (gibbs.c: tabulated
  entry averaged with a Vinet beta-nepheline reference).
- feldspar_disorder.py: albite displacive + Al-Si ordering (albite.c,
  2-parameter Newton), added to albite in compute(). This was the
  feldspar Cp gap (up to ~11% for pure albite).
- feldspar_disorder: shared ordering_response / equilibrium_properties
  (Cp, dCp/dT, V derivatives along an ordering equilibrium).
- nepheline: where MELTS skips the ordering solve (no K, no Na, or
  vacancy-saturated), ds/dT = ds/dP = 0 is used; MAGMA reuses a stale LU
  there, so its Cp/dVdT/dVdP are undefined for those rows.
- API: garnet, leucite, biotite, hornblende, nepheline added to
  _MELTS_SOLUTION_PHASES. Bulk sums ignore rows with zero phase moles, so
  placeholder compositions there cannot inject inf/NaN; a non-finite
  property on a row with nonzero moles now raises ValueError.
- melts_comparison: the new phases (GT .tbl endmember columns and emulator
  bridge) and muscovite as a pure phase.
- MAGMA oracle now built with -DRHYOLITE_ADJUSTMENTS, as alphaMELTS is
  (sanidine H was 3400 J off without it); liquid/fluid values unchanged.
  New ACT/SOLI/SS/SOLLIST oracle commands; frozen
  solid_solutions_oracle.json + benchmark_minor_solutions_test.py.
- All five models and the endmember cases match MAGMA to ~1e-15 relative.
  GT assemblage + EOS on the 120 standards: all bulk mean errors <=0.07%
  (BishopTuff Cp -0.33% -> 0.001%); 102 BishopTuff leucite exact.
- Open: alphaMELTS's reported plagioclase H/S (BishopTuff) differ from its
  own library by ~T*1.6 J/K/mol with G agreeing to 0.004% (not the EOS);
  tridymite H +0.31% vs alphaMELTS output (library matches exactly).
