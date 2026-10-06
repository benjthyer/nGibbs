2026-09-29
- New prototype: engine/cumulate_inversion.py. Inverts a closed isothermal
  MELTS ContinuousModel for the liquid, P and T a cumulate crystallised
  from, by gradient descent on the network inputs (weights frozen). No
  existing module was changed.
- Parameters per node: P and T through a sigmoid onto the training range,
  bulk through a softmax over Elkeys (closed, positive; elements can be
  pinned to 0 with InversionConfig.zero_elements).
- Loss per node: assemblage hinges on a = g_phi / T0 (required cumulus
  phases + liquid: hinge(1 - a); every other phase: hinge(a + 1); fluid
  free), + cumulus-phase oxide wt% misfit (sigma 1 wt%), + (1 - f_liq)
  with f_liq the element-mole fraction held by liquid (raw moles, no mass
  balance), + squared excursion of the bulk outside the training oxide
  range. hinge is a one-sided Huber (0 once met, d^2, then 2d - 1).
  w_liq ramps in over the first 25% of steps.
- Nodes are independent (Adam is elementwise), run in chunks of
  batch_size; the best state (lowest full-weight loss) of every node is
  kept. Starts: random P, GEOROC compositions with log-normal noise
  (sigma 0.1 per oxide) and a random Fe3+/sum Fe in [0.03, 0.25], clipped
  into the training range, each at its own liquidus.
- find_liquidi: vectorised liquidus for many (composition, P) pairs.
  Runs only network_component_moles (no mass balance, no phase tables);
  descending grid (48) then bisection (10) on the highest crossing of
  max_solid(g) = 0, fluid ignored. Returns T, a status (0 bracketed,
  +1 solid at Tmax, -1 no solid by Tmin) and the liquidus phase.
  NN_MELTS.find_liquidus / find_liquidi are untouched and still call the
  removed forward_binary.
- Bundle labels store pyroxene/spinel in the native MELTS endmember basis
  (project with ml_indexer.compToOxLoad); the chem heads emit the
  PxSp-transformed basis (compToOx). Using compToOx on labels gives cpx
  with negative MgO/Fe2O3 and a median cpx misfit of 8.5 wt%;
  _Geometry.phase_oxwt(native=True) handles labels.
- InversionResult: per-node table (P, T, loss terms, f_liq, comp RMSE,
  predicted liquid, bulk and start oxides), accepted() filters,
  summary(), weighted covariance()/correlation, density2d() (numpy
  Gaussian KDE, Scott bandwidth, full covariance) and plot_density()
  (SiO2-MgO KDE with marginals + P-T KDE, truth and best node marked).
- scripts/cumulate_inversion_demo.py: strips the liquid off bundle rows
  for 7 assemblages, inverts each, writes nodes/summary/covariance/density
  per target and recovery.csv; --no-truth-check skips the control run
  seeded at the true (liquid, P, T).
- tests/unit_tests/test_cumulate_inversion.py: bundle loader, oxide
  round trip, label vs network basis, liquidus bracketing, bundle target,
  inversion smoke test (skipped without the 102 NoCr NPT checkpoint).
- Results (1024 nodes, 600 steps, CPU): see docs/METHODS_cumulate_inversion.md.
- Revision (same day), from review:
  - Extra phases are free by default (InversionConfig.extra_phases='free'):
    phases outside the cumulate are not penalised, only what they take from
    f_liq; 'forbid' restores the old hinge. Nodes report n_extra_phases,
    extra_phases and extra_fraction.
  - w_liquid 1 -> 2, no warm-up ramp. f_liq at the best state barely moves
    (0.94-0.97, as before): the present_margin hinge (1 T0 per cumulus phase)
    is what binds. present_margin 0.25 gives f_liq 0.985 on 102 ol+cpx+pl
    (256 nodes); default left at 1.0.
  - Chemical basis: oxides >= min_oxide_wt (0.01 wt%) in some cumulus phase
    are optimised; H2O/CO2 are carried at their prior value; everything else
    is exactly 0. x = x_fixed + (1 - sum x_fixed) * softmax(z over optimised
    elements). Truth liquids are renormalised onto the basis for comparison;
    recovery RMSEs are over the optimised oxides only.
  - Prior replaces GEOROC starts: Dirichlet cumulus proportions -> bulk,
    volatiles ~ N(mean, std) of the training data clamped to [0, training
    max] and zeroed for 1/3 of nodes, P ~ U(training), T ~ U(Tmin, 1600);
    forward model, T -= 100 while superliquidus (T += 100 while no liquid),
    up to 5 times; the predicted liquid is the start, at T - 5. The liquid's
    volatiles are reset to the drawn values (a crystal-rich mix otherwise
    hands the liquid ~1/F times the H2O).
  - constants.CUMULATE_VOLATILE_PRIORS: H2O/CO2 wt% stats from
    102Isothermal_NoCr_Train (H2O 0.41 +- 0.31 over hydrous rows), 120OpenOx_NoCr_Train
    (H2O 1.20 +- 1.36, CO2 0.31 +- 0.22) and, as a stand-in for the SedIg
    training set, the 120SedIgClosed test subset (CO2 5.9 +- 7.2).
  - Reported liquid: liq_* (chem head at the best state) and liqMB_* (after
    NN_MELTS iterative mass balance).
  - find_liquidi: tol (C) sets the bisection count; n_grid default 24;
    return_phase=False skips the extra pass; 'passes' is returned.
  - tqdm bar over steps (InversionConfig.progress; auto when stderr is a tty),
    flushed step lines otherwise.
  - InversionResult.plot_prior / plot_drift / drift(); the demo writes
    <tag>_prior.png, <tag>_prior_accepted.png, <tag>_drift.png;
    scripts/cumulate_inversion_replot.py redraws them from saved nodes files.
  - scripts/run_cumulate_inversion_gpu.bat: env check, tests, 2^15-node
    validation on 102 and 120 NoCr (RTX 3060 Laptop: ~3-5 min per target).
- Revision 3:
  - Presence margins are the inverter's own convention (inference decides
    presence by g_phi > 0; T0 is used by the network only in training). They
    are now 0.05 T0 (was 1 T0), with the hinge scaled by hinge_width = 0.1 T0 so
    the constraint still bites. On 102 ol+cpx+pl / cpx+pl (256 nodes, 400
    steps): best-node f_liq 0.937 -> 0.968 / 0.968 -> 0.986, best liquid RMSE
    4.5 -> 2.5 / 2.5 -> 1.7 wt%.
  - recovery.csv: role / truth / best / accepted-median / prior-median and the
    three errors for EVERY model oxide ('--' outside the target's basis;
    H2O and CO2 are 'carried'); prior baselines for P, T and the liquid
    (median |error| per node, and the RMSE of the median).
  - InversionResult.plot_prior_vs_final (prior starts above accepted finals on
    shared axes, median |error| in each title), plot_convergence, history with
    per-term sums, per-node loss quantiles and best_min. Composition axes switch
    to SiO2 vs molar (Na2O+K2O+2CaO)/(2Al2O3) when the (true) liquid has < 4 wt%
    MgO (liq_NKC2A etc. derived columns).
  - Demo draws --n-assemblages (20) random rows with >= 2 solids (fluid ignored),
    one per distinct solid assemblage; --fixed-assemblages keeps the old list.
  - v2 GPU runs (margin 1 T0), prior vs accepted, median |error|: T 78-450 C ->
    22-84 C in all 14 targets; P better in 10/14; per-node liquid RMSE better in
    12/14 (worse: 102 opx+cpx+pl, 120 ol+cpx).
  - Emulator baseline: CumulateTarget.from_bundle stores truth['emulator'], the
    forward model at the row's true (P, T, bulk) -- liquid (raw and mass
    balanced), predicted solids, f_liq, cumulus-phase compositions. Plotted as a
    hollow teal diamond ("nGibbs at true P, T, bulk") next to the truth star in
    plot_density / plot_prior_vs_final (it sits on the star in P-T by
    construction); summary() gains an nGibbs_at_truth column; recovery.csv gains
    emu_liq_rmse_wt, emu_<ox>, emu_err_<ox>, emu_assemblage_match,
    emu_solids_predicted, emu_f_liquid, emu_cumulus_rmse_wt. On the 14 v2 GPU
    targets the emulator liquid is 0.08-0.54 wt% RMS from MELTS (constrained
    oxides), solids match in 12/14, while best-node inversion errors are 0.7-13 wt%.
  - w_assemblage default 0.05 (Ben's edit on disk, kept).
  - Prior: after the Dirichlet mix of cumulus phases, each oxide is multiplied
    by an independent U(0.7, 1.3) factor and the bulk re-closed
    (InversionConfig.prior_oxide_noise; None turns it off), before volatiles
    are added and P, T drawn.
  - v3 GPU 102 (20 random assemblages, margin 0.05 T0, w_assemblage 0.05),
    medians: emulator-at-truth liquid 0.23 wt% RMS from MELTS; inversion best
    node 1.71, accepted nodes 2.78, prior 6.44 wt%; |dT| accepted 34 C vs prior
    160 C; |dP| accepted 1.4 kbar vs prior 4.5 kbar.
  - Prior: accessory phases (apatite, whitlockite, muscovite, graphite, calcite;
    InversionConfig.prior_trace_phases) get U(0, 10%) of the cumulus mix each
    (prior_trace_max) instead of a Dirichlet share; the other cumulus phases
    split the remainder ~ Dirichlet. All-accessory cumulates keep the Dirichlet.
