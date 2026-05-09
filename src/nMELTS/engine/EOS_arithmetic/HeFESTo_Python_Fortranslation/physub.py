"""
Python translation of physub.f (HeFESTo subroutine)

External functions/routines called — these must be imported/connected:
  - nform(nnew, n, n1, q2, nspec, nnull)
  - dgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
      -> wraps scipy.linalg.blas.dgemv or numpy equivalent
  - hessfunc(nnew, hespro)
  - svdsub(m, n, A, lda, ldb, b, vwork1, vwork2, x, nnulls)
  - parset(ispec, apar, fn, zu, wm, To, Fo, Vo, Ko, Kop, Kopp,
           wd1, wd2, wd3, ws1, ws2, ws3,
           we1, qe1, we2, qe2, we3, qe3, we4, qe4, wou, wol,
           gam, qo, be, ge, q2A2, htl, ibv, ied, izp, Go, Gop, Got)
  - cp(ispec, n, chempot, rsum, volsum, sconf, smag)
  - gspec(ispec)  -> returns scalar
  - ddot(n, x, incx, y, incy) -> wraps scipy.linalg.blas.ddot or numpy.dot
  - dlacpy(uplo, m, n, A, lda, B, ldb) -> copies matrix
  - dgetrf(m, n, A, lda, ipiv, info) -> LU factorization (LAPACK)
  - dgetri(n, A, lda, ipiv, work, lwork, info) -> matrix inverse (LAPACK)
  - dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  - thetacal(x, result)  -> returns result via out-param; translate as return value
  - tlindeman(vol, wmagg, fnagg, thet) -> returns scalar
  - asqrt(x) -> safe square root (returns scalar)
  - depth(Pi)  -> returns depth from pressure
  - qr19(depth, Ti, qs, qp) -> returns qs, qp
  - vred(qs, qp, vsred, vpred) -> returns vsred, vpred

Global state accessed (from Fortran COMMONs/includes — must be provided as a State object or module):
  - Pi, Ti                              (/state/ common)
  - apar[nspecp, nparp]                 (/state/ common)
  - vol, Cap, Cv, gamma, K, Ks, alp, Ftot, ph, ent, deltas, tcal, zeta, Gsh,
    uth, uto, thet, qq, etas, dGdT, pzp, Vdeb, gamdeb  (/prop/ common)
  - cpa[nspecp]                         (/chempot/ common)
  - icfe                                (/mag/ common)
  - stox, wox, wcomp, stcomp, atom, comp  (/atomc/ common)
  - tlast, vtarg, starg, phugo, vhugo, ehugo, dvdpmol, dsdtmol,
    dhdpmol, dhdtmol, wmaggc, chcalc, adcalc, hucalc, nvet, nvep  (/tfindc/ common)
  - iphyflag                            (/phycom/ common)
  - spinod[nspecp], spinph[nphasep]     (/spinc/ common)
  - phname[nphasep], sname[nspecp]      (/names/ common)
  - n[nspecp], n1, q2[nspecp,nspecp], nnull, nnulls  (from include files)
  - nspec, nph, nc, nco, nspecp, nphasep, ncompp, nparp  (dimension parameters)
  - f[nphasep, nspecp]   (phase membership matrix)
  - s[ncompp, nspecp]    (stoichiometry)
  - absents[nspecp]      (logical array)
  - sspeca[nspecp], vspeca[nspecp]  (species entropy/volume derivatives)
  - iophase[nphasep], mophase[nphasep]  (phase species ranges)
  - iphase[nphasep]     (representative species index per phase)
  - lagc[ncompp]        (Lagrange multipliers / chemical potentials of components)
  - b[ncompp]           (composition vector)
  - Rgas                (gas constant)
  - Ksp, Gshp           (pressure derivatives of K and Gsh)
  - one, zero, ione, zervec  (BLAS constants; ione=1, one=1.0, zero=0.0)
  - fn                  (total moles, updated by parset/cp calls)
  - zu                  (updated by parset)

File unit outputs (write statements) — these are translated to calls on a
file_handles dict with integer keys mapping to file objects or logging targets.
If you do not need the output files, replace with pass or logging calls.
"""

import math
import time
import numpy as np

from .param_state import asqrt
from .state import HeFESToState
from .extern import HeFESToExtern
import tqdm

# ---------------------------------------------------------------------------
# Constants (Fortran PARAMETERs inside physub)
# ---------------------------------------------------------------------------
RHOMANTLE = 4422.76       # kg/m^3
GMANTLE   = 10.0          # m/s^2
HMANTLE   = 2891000.0     # m
PMANTLE   = RHOMANTLE * GMANTLE * HMANTLE / 1.0e9  # GPa

TSMALL    = 1.0e-5
ASMALL    = 1.0e-15
NSMALL    = 1.0e-15
DFDPSMALL = 1.0e-13
IADDMAX   = 5

# Standard oxide oxygen counts and component labels (indexed 0-based)
NOXIDES   = 7
NOXYGEN   = [2.0, 1.0, 1.0, 1.0, 1.5, 0.5, 1.5]         # SiO2 MgO FeO CaO AlO1.5 NaO0.5 CrO1.5
COMPSTAND = ['Si', 'Mg', 'Fe', 'Ca', 'Al', 'Na', 'Cr']

# O2 thermodynamic coefficients (used for oxygen fugacity output)
_AO2   =  47.255
_BO2   = -4.550e-4
_CO2   =  4.402e5
_DO2   = -393.5
_TRO2  =  298.15
_STRO2 =  205.147


# ---------------------------------------------------------------------------
# Module-level persistent state (mimics Fortran SAVE / DATA statements)
# ---------------------------------------------------------------------------
_ncall   = 0
_gphysub = 0.0


def physub(nnew, rho, wmagg, freeagg, 
           state,          # object/namespace carrying all COMMON variables
           extern,
           iprint=1):
    """
    Python translation of the Fortran subroutine physub.

    Parameters
    ----------
    nnew : ndarray, shape (nspecp,)
        Input species amounts (double precision).
    rho : float
        Density (output, returned in result dict).
    wmagg : float
        Aggregate molar mass (output).
    freeagg : float
        Aggregate Helmholtz / Gibbs free energy (output).
    iprint : int
        Verbosity flag (1 = write extended output).
    state : namespace/object
        Carries all COMMON-block variables listed in the module docstring.
    extern : namespace/object
        Carries all external functions listed in the module docstring.

    Returns
    -------
    dict with keys: 'rho', 'wmagg', 'freeagg', and updated state fields.
    """

    global _ncall, _gphysub

    # ------------------------------------------------------------------
    # Timing / bookkeeping
    # ------------------------------------------------------------------
    start = time.process_time()
    _ncall += 1

    # Convenience aliases for state variables
    s   = state        # shorthand
    fh  = getattr(s, 'file_handles', {})
    ex  = extern

    # Reset scalar state
    s.ent      = 0.0
    s.iphyflag = 1

    # ------------------------------------------------------------------
    # Warn about spinodally unstable species
    # ------------------------------------------------------------------
    for ispec in range(s.nspec):
        if s.spinod[ispec] and not s.absents[ispec]:
            # _write(fh, 31,
            print(f'WARNING: Spinodally unstable species present '
                   f'{s.Pi} {s.Ti} {ispec} {s.absents[ispec]} '
                   f'{s.spinod[ispec]} {s.n[ispec]}')

    # ------------------------------------------------------------------
    # Initialise aggregate accumulators
    # ------------------------------------------------------------------
    entagg     = 0.0
    freeagg    = 0.0
    volagg     = 0.0
    wmagg      = 0.0
    cpagg      = 0.0
    cvagg      = 0.0
    baggr      = 0.0
    btaggr     = 0.0
    gaggr      = 0.0
    baggv      = 0.0
    btaggv     = 0.0
    gaggv      = 0.0
    alpagg     = 0.0
    delagg     = 0.0
    fnagg      = 0.0
    vabsagg    = 0.0
    vabsoagg   = 0.0
    dgdtaggv   = 0.0
    dgdtaggr   = 0.0
    smix       = 0.0
    epsilon1   = 0.0
    epsilon2   = 0.0
    vdebagg    = 0.0
    Gwu        = 0.0
    Gironmin   = +1.0e15
    Giron      = np.zeros(3)

    # ------------------------------------------------------------------
    # Form n from nnew
    # ------------------------------------------------------------------
    ex.nform(nnew, s.n, s.n1, s.q2, s.nspec, s.nnull)

    # ------------------------------------------------------------------
    # Species initialisation — fix absent species and set dm/dT, dm/dP
    # ------------------------------------------------------------------
    dmdt = np.zeros(s.nspecp)
    dmdp = np.zeros(s.nspecp)
    for ispec in range(s.nspec):
        if s.absents[ispec]:
            s.n[ispec] = NSMALL / 10.0
        dmdt[ispec] =  s.sspeca[ispec] / 1000.0
        dmdp[ispec] = -s.vspeca[ispec]

    # ------------------------------------------------------------------
    # Project dm/dT and dm/dP into null space, solve for dn^P
    # ------------------------------------------------------------------
    dmdtp = np.zeros(s.nspecp)
    dmdpp = np.zeros(s.nspecp)
    dndtp = np.zeros(s.nspecp)
    dndpp = np.zeros(s.nspecp)
    dndt  = np.zeros(s.nspecp)
    dndp  = np.zeros(s.nspecp)

    print('Projecting dm/dT and dm/dP into null space...')
    """

    ex.dgemv('Transpose q2', s.nspec, s.nnull, s.one, s.q2, s.nspecp,
             dmdt, s.ione, s.zero, dmdtp, s.ione)
    ex.dgemv('Transpose q2', s.nspec, s.nnull, s.one, s.q2, s.nspecp,
             dmdp, s.ione, s.zero, dmdpp, s.ione)

    hespro = np.zeros((s.nspecp, s.nspecp))
    ex.hessfunc(nnew, hespro)

    vwork = np.zeros((s.nspecp, s.nspecp))
    ex.svdsub(s.nnull, s.nnull, hespro, s.nspecp, s.nspecp,
              dmdtp, vwork, vwork, dndtp, s.nnulls)
    ex.svdsub(s.nnull, s.nnull, hespro, s.nspecp, s.nspecp,
              dmdpp, vwork, vwork, dndpp, s.nnulls)

    ex.nform(dndtp, dndt, s.zervec, s.q2, s.nspec, s.nnull)
    ex.nform(dndpp, dndp, s.zervec, s.q2, s.nspec, s.nnull)


    # Consistency checks
    checkt = ex.ddot(s.nspec, s.cpa, s.ione, dndt, s.ione)
    checkp = ex.ddot(s.nspec, s.cpa, s.ione, dndp, s.ione)

    # ------------------------------------------------------------------
    # Alternative dndt via full projected Hessian inverse (validation)
    # ------------------------------------------------------------------
    swork = np.zeros((s.nspecp, s.nspecp))
    ex.dlacpy('All', s.nnull, s.nnull, hespro, s.nspecp, swork, s.nspecp)
    ipiv = np.zeros((s.nspecp, s.nspecp), dtype=int)
    info = [0]
    ex.dgetrf(s.nnull, s.nnull, swork, s.nspecp, ipiv, info)
    wwork = np.zeros(s.nspecp)
    ex.dgetri(s.nnull, swork, s.nspecp, ipiv, wwork, s.nspecp, info)

    ex.dgemm('Normal', 'Transpose q2', s.nnull, s.nspec, s.nnull,
             s.one, swork, s.nspecp, s.q2, s.nspecp, s.zero, vwork, s.nspecp)
    ex.dgemm('Normal q2', 'Normal', s.nspec, s.nspec, s.nnull,
             s.one, s.q2, s.nspecp, vwork, s.nspecp, s.zero, swork, s.nspecp)

    dndttest = np.zeros(s.nspecp)
    ex.dgemv('Normal', s.nspec, s.nspec, s.one, swork, s.nspecp,
             dmdt, s.ione, s.zero, dndttest, s.ione)

    # dn_atom/dP and phase proportion derivatives
    dnatomdp = np.zeros(s.nspecp)
    for ispec in range(s.nspec):
        dnatomdp[ispec] = s.apar[ispec, 0] * dndp[ispec]   # apar(ispec,1) -> index 0

    dfdp = np.zeros(s.nphasep)
    ex.dgemv('Normal', s.nph, s.nspec, s.one, s.f, s.nphasep,
             dnatomdp, s.ione, s.zero, dfdp, s.ione)"""

    # ------------------------------------------------------------------
    # Compute fast (intra-phase) dn/dT and dn/dP
    # ------------------------------------------------------------------
    dndpfast      = np.zeros(s.nspecp)
    dndtfast      = np.zeros(s.nspecp)
    dndpfastphase = np.zeros(s.nspecp)
    dndtfastphase = np.zeros(s.nspecp)
    q2save        = np.copy(s.q2)

    nfast    = 0
    nfastold = 0

    for iph in tqdm.tqdm(range(s.nph), 'Computing fast terms'):
        isign = -1.0
        for ispec in range(s.nspec):
            s.q2[ispec, 0] = 0.0          # zero first null vector column

        for ispec in range(s.nspec):
            if s.f[iph, ispec] == 0.0:
                continue
            lo = s.iophase[iph]
            hi = s.iophase[iph] + s.mophase[iph] - 1  # inclusive upper bound (1-based in F)
            # Convert to 0-based: ispec (0-based here) in [lo-1, hi-1]
            if lo - 1 <= ispec <= hi - 1:
                nfast  += 1
                isign   = -1.0 * isign
                s.q2[ispec, 0] = float(isign) / math.sqrt(2.0)

        nnullfast = 1
        if nfast == nfastold:
            # restore q2 and skip
            s.q2[:, :] = q2save
            continue
        nfastold = nfast

        _write(fh, 31, f"Compute fast terms for phase {s.phname[iph]:10s}"
                       f"{nfast:5d}{nnullfast:5d}")

        ex.dgemv('Transpose q2fast', s.nspec, nnullfast, s.one, s.q2, s.nspecp,
                 dmdt, s.ione, s.zero, dmdtp, s.ione)
        ex.dgemv('Transpose q2fast', s.nspec, nnullfast, s.one, s.q2, s.nspecp,
                 dmdp, s.ione, s.zero, dmdpp, s.ione)

        ex.hessfunc(nnew, hespro)

        ex.svdsub(nnullfast, nnullfast, hespro, s.nspecp, s.nspecp,
                  dmdtp, vwork, vwork, dndtp, s.nnulls)
        ex.svdsub(nnullfast, nnullfast, hespro, s.nspecp, s.nspecp,
                  dmdpp, vwork, vwork, dndpp, s.nnulls)

        ex.nform(dndtp, dndtfastphase, s.zervec, s.q2, s.nspec, s.nnull)
        ex.nform(dndpp, dndpfastphase, s.zervec, s.q2, s.nspec, s.nnull)

        for ispec in range(s.nspec):
            dndtfast[ispec] += dndtfastphase[ispec]
            dndpfast[ispec] += dndpfastphase[ispec]

        s.q2[:, :] = q2save   # restore

    s.q2[:, :] = q2save        # final restore (handles loop-end case)

    # ------------------------------------------------------------------
    # Per-phase property accumulation (main phase loop)
    # ------------------------------------------------------------------
    ntemp    = np.zeros(s.nspecp)

    # Per-phase output arrays
    rhopha  = np.zeros(s.nphasep)
    vbpha   = np.zeros(s.nphasep)
    vspha   = np.zeros(s.nphasep)
    vppha   = np.zeros(s.nphasep)
    volpha  = np.zeros(s.nphasep)
    fnpha   = np.zeros(s.nphasep)
    entpha  = np.zeros(s.nphasep)
    henpha  = np.zeros(s.nphasep)
    gibpha  = np.zeros(s.nphasep)
    entopha = np.zeros(s.nphasep)
    henopha = np.zeros(s.nphasep)
    gibopha = np.zeros(s.nphasep)
    vabs    = np.zeros(s.nphasep)
    vabso   = np.zeros(s.nphasep)
    gpha    = np.zeros(s.nphasep)
    bpha    = np.zeros(s.nphasep)
    btpha   = np.zeros(s.nphasep)
    wmpha   = np.zeros(s.nphasep)

    for iph in range(s.nph):
        # Initialise per-phase accumulators
        volph  = 0.0
        volpho = 0.0
        rhoph  = 0.0
        bukph  = 0.0
        buktph = 0.0
        gshph  = 0.0
        alpph  = 0.0
        cpph   = 0.0
        delph  = 0.0
        wmph   = 0.0
        fnph   = 0.0
        entph  = 0.0
        henph  = 0.0
        gibph  = 0.0
        entoph = 0.0
        henoph = 0.0
        giboph = 0.0
        gsolph = 0.0
        hsolph = 0.0
        dgdtph = 0.0
        rhopha[iph] = 0.0
        vbpha[iph]  = 0.0
        vspha[iph]  = 0.0
        vppha[iph]  = 0.0
        volpha[iph] = 0.0
        fnpha[iph]  = 0.0
        entpha[iph] = 0.0
        henpha[iph] = 0.0
        gibpha[iph] = 0.0
        wmpha[iph]  = 0.0
        vabs[iph]   = 0.0
        vabso[iph]  = 0.0
        gpha[iph]   = 0.0
        bpha[iph]   = 0.0

        # ---- Inner species loop ---- #VECTORIZATION TARGET?
        for ispec in range(s.nspec):
            if s.f[iph, ispec] == 0:
                continue

            # Call parset to populate state thermodynamic variables
            (fn, zu, wm, To, Fo, Vo, Ko, Kop, Kopp,
             wd1, wd2, wd3, ws1, ws2, ws3,
             we1, qe1, we2, qe2, we3, qe3, we4, qe4, wou, wol,
             gam, qo, be, ge, q2A2,
             htl, ibv, ied, izp, Go, Gop, Got) = ex.parset(
                ispec, s.apar)

            ntemp[:] = 0.0
            ntemp[ispec] = 1.0
            chempot, rsum, volsum, sconf, smag = ex.cp(ispec, ntemp)
            sconf = -chempot / s.Ti if s.Ti > 0.0 else 0.0

            fdumm = ex.gspec(ispec)
            chempot, rsum, volsum, smixi, smag = ex.cp(ispec, s.n)

            if not s.absents[ispec]:
                _write(fh, 31,
                       f"{'Chemical potential and activity':31s}{ispec+1:5d} "
                       f"{s.sname[ispec]:4s}"
                       f"{chempot/1000.:16.8f}{math.exp(chempot/(s.Rgas*s.Ti)):16.8f}"
                       f"{rsum:16.8f}")

            smix    += s.n[ispec] * smixi
            s.Ftot  -= s.Ti * smag
            s.ent   += smag
            entagg  += s.n[ispec] * s.sspeca[ispec]
            freeagg += s.n[ispec] * (fdumm + chempot)
            volxs    = s.vol + volsum
            volagg  += s.n[ispec] * s.vspeca[ispec]
            wmagg   += s.n[ispec] * wm
            wmph    += s.n[ispec] * wm

            # Reuss averages (harmonic in modulus)
            bukph  += s.n[ispec] * s.vol / s.Ks
            buktph += s.n[ispec] * s.vol / s.K
            gshph  += s.n[ispec] * s.vol / s.Gsh
            volph  += s.n[ispec] * volxs
            volpho += s.n[ispec] * Vo
            fnph   += s.n[ispec]

            if not s.absents[ispec]:
                cpagg  += s.n[ispec] * s.Cap
            cvagg  += s.n[ispec] * s.Cv
            if not s.absents[ispec]:
                alpagg += s.n[ispec] * s.vol * s.alp
            delagg  += s.n[ispec] * s.vol * s.alp * (1.0 + s.deltas) / s.Ks
            vdebagg += s.n[ispec] * s.vol * s.Vdeb

            if not s.absents[ispec]:
                alpph += s.n[ispec] * s.vol * s.alp
            if not s.absents[ispec]:
                cpph  += s.n[ispec] * s.Cap
            delph   += s.n[ispec] * s.vol * s.alp * (1.0 + s.deltas) / s.Ks

            Gtot = s.Ftot
            Htot = Gtot + s.Ti * s.ent
            Etot = Htot - 1000.0 * s.Pi * s.vol

            entph  += s.n[ispec] * (s.ent + smixi)
            henph  += (s.n[ispec] * s.cpa[ispec]
                       + s.Ti * s.n[ispec] * (s.ent + smixi) / 1000.0)
            gibph  += s.n[ispec] * s.cpa[ispec]
            entoph += s.n[ispec] * s.ent
            henoph += s.n[ispec] * Htot / 1000.0
            giboph += s.n[ispec] * Gtot / 1000.0
            gsolph += s.n[ispec] * (Gtot + chempot)
            dgdtph += s.n[ispec] * s.vol / s.Gsh**2 * s.dGdT

            epsilon1 += s.n[ispec] * wm
            epsilon2 += s.n[ispec] * wm**2

            # Track specific iron-phase Gibbs energies
            if s.sname[ispec].strip() == 'wu':
                Gwu = Gtot / 2.0
            if s.sname[ispec].strip() == 'fea':
                Giron[0] = Gtot
            if s.sname[ispec].strip() == 'feg':
                Giron[1] = Gtot
            if s.sname[ispec].strip() == 'fee':
                Giron[2] = Gtot
            for iiron in range(3):
                if Giron[iiron] != 0.0:
                    Gironmin = min(Gironmin, Giron[iiron])

        # ---- Phase-wise metamorphic (fast) contribution ----
        alpmet = 0.0
        cpmet  = 0.0
        bmet   = 0.0
        for ispec in range(s.nspec):
            lo = s.iophase[iph] - 1      # convert to 0-based
            hi = s.iophase[iph] + s.mophase[iph] - 2   # inclusive 0-based
            if lo <= ispec <= hi:
                alpmet += dndtfast[ispec] * s.vspeca[ispec]
                cpmet  += s.Ti * dndtfast[ispec] * s.sspeca[ispec]
                bmet   += dndpfast[ispec] * dmdp[ispec]
                _write(fh, 31,
                       f"bmet calc ispec,bmet,dndpfast,dmdp "
                       f"{ispec+1} {bmet} {dndpfast[ispec]} {dmdp[ispec]}")

        # Reuss averages -> moduli
        bukphreuss  = volph / bukph  if bukph  != 0.0 else 0.0
        buktphtot   = volph / (buktph + bmet) if (buktph + bmet) != 0.0 else 0.0
        buktph      = volph / buktph  if buktph  != 0.0 else 0.0
        gshph       = volph / gshph   if gshph   != 0.0 else 0.0
        hsolph      = gsolph + s.Ti * entph
        alpphtot    = alpph / volph + alpmet / volph
        alpph       = alpph / volph  if volph != 0.0 else 0.0
        cpphtot     = cpph / wmph + cpmet / wmph  if wmph != 0.0 else 0.0
        cvphtot     = (cpphtot - s.Ti * volph * alpphtot**2 * buktphtot / wmph * 1000.0
                       if wmph != 0.0 else 0.0)
        gamphtot    = (volph * alpphtot * buktphtot / (cvphtot * wmph) * 1000.0
                       if (cvphtot > 0.0 and wmph != 0.0) else 0.0)
        bukph       = buktphtot * (1.0 + alpphtot * gamphtot * s.Ti)


        if alpph != 0.0 and volph != 0.0:
            delph = delph * bukph / (volph * alpph) - 1.0
        rhoph       = wmph / volph  if volph != 0.0 else 0.0
        dlnvbdtph   = alpph / 2.0 * (delph - 1.0)
        if volph != 0.0:
            dgdtph = gshph**2 * dgdtph / volph

        rhopha[iph] = rhoph
        vbpha[iph]  = asqrt(bukph / rhoph) if rhoph > 0.0 else 0.0
        vspha[iph]  = asqrt(gshph / rhoph) if rhoph > 0.0 else 0.0
        vppha[iph]  = asqrt((bukph + 4.0/3.0 * gshph) / rhoph) if rhoph > 0.0 else 0.0
        volpha[iph] = volph / fnph  if fnph > 0.0 else 0.0

        if fnph > TSMALL:
            entpha[iph]  = entph  / fnph
            henpha[iph]  = henph  / fnph / 1000.0
            gibpha[iph]  = gibph  / fnph / 1000.0
            entopha[iph] = entoph / fnph
            henopha[iph] = henoph / fnph / 1000.0
            gibopha[iph] = giboph / fnph / 1000.0
            fnpha[iph]   = fn * fnph

        wmpha[iph]  = wmph
        vabs[iph]   = volph
        vabso[iph]  = volpho
        gpha[iph]   = gshph
        bpha[iph]   = bukph
        btpha[iph]  = buktph
        dgdtph      = dgdtph * gshph**2   # re-scale (matches line 471)

        # Accumulate into aggregate arrays
        fnagg    += fnpha[iph]
        vabsagg  += vabs[iph]
        vabsoagg += vabso[iph]
        baggv    += volph * bukph
        baggr    += volph / bukph  if bukph  != 0.0 else 0.0
        btaggv   += volph * buktph
        btaggr   += volph / buktph if buktph != 0.0 else 0.0
        gaggv    += volph * gshph
        gaggr    += volph / gshph  if gshph  != 0.0 else 0.0
        dgdtaggv += volph * dgdtph
        dgdtaggr += (volph / gshph**2 * dgdtph) if gshph != 0.0 else 0.0

    # ------------------------------------------------------------------
    # Hashin-Shtrikman bounds
    # ------------------------------------------------------------------
    gmin  = +1.0e15;  gmax  = -1.0e15
    bmin  = +1.0e15;  bmax  = -1.0e15
    vsmin = +1.0e15;  vsmax = -1.0e15
    vpmin = +1.0e15;  vpmax = -1.0e15

    for iph in range(s.nph):
        if vabs[iph] == 0.0:
            continue
        gmin  = min(gmin,  gpha[iph])
        gmax  = max(gmax,  gpha[iph])
        bmin  = min(bmin,  bpha[iph])
        bmax  = max(bmax,  bpha[iph])
        vsmin = min(vsmin, vspha[iph])
        vsmax = max(vsmax, vspha[iph])
        vpmin = min(vpmin, vppha[iph])
        vpmax = max(vpmax, vppha[iph])

    flambdap = 0.0;  flambdam = 0.0
    fgammap  = 0.0;  fgammam  = 0.0
    ximin = gmin / 6.0 * ((9.0*bmin + 8.0*gmin) / (bmin + 2.0*gmin)) if (bmin + 2.0*gmin) != 0.0 else 0.0
    ximax = gmax / 6.0 * ((9.0*bmax + 8.0*gmax) / (bmax + 2.0*gmax)) if (bmax + 2.0*gmax) != 0.0 else 0.0

    for iph in range(s.nph):
        if vabs[iph] == 0.0:
            continue
        frac = vabs[iph] / volagg if volagg != 0.0 else 0.0
        flambdap += frac / (bpha[iph] + 4.0/3.0 * gmax)
        flambdam += frac / (bpha[iph] + 4.0/3.0 * gmin)
        fgammap  += frac / (ximax + gpha[iph])
        fgammam  += frac / (ximin + gpha[iph])

    bhashp = 1.0/flambdap - 4.0/3.0*gmax if flambdap != 0.0 else 0.0
    bhashm = 1.0/flambdam - 4.0/3.0*gmin if flambdam != 0.0 else 0.0
    ghashp = 1.0/fgammap  - ximax         if fgammap  != 0.0 else 0.0
    ghashm = 1.0/fgammam  - ximin         if fgammam  != 0.0 else 0.0

    s.wmaggc = wmagg
    rho      = wmagg / volagg if volagg != 0.0 else 0.0

    baggv  = baggv  / volagg if volagg != 0.0 else 0.0
    baggr  = volagg / baggr  if baggr  != 0.0 else 0.0
    btaggv = btaggv / volagg if volagg != 0.0 else 0.0
    btaggr = volagg / btaggr if btaggr != 0.0 else 0.0
    gaggv  = gaggv  / volagg if volagg != 0.0 else 0.0
    gaggr  = volagg / gaggr  if gaggr  != 0.0 else 0.0

    baggh  = (baggv + baggr) / 2.0
    btaggh = (btaggv + btaggr) / 2.0
    gaggh  = (gaggv + gaggr) / 2.0

    Vbr = ex.asqrt(baggr / rho) if rho > 0.0 else 0.0
    Vsr = ex.asqrt(gaggr / rho) if rho > 0.0 else 0.0
    Vpr = ex.asqrt((baggr + 4.0/3.0*gaggr) / rho) if rho > 0.0 else 0.0
    Vbv = ex.asqrt(baggv / rho) if rho > 0.0 else 0.0
    Vsv = ex.asqrt(gaggv / rho) if rho > 0.0 else 0.0
    Vpv = ex.asqrt((baggv + 4.0/3.0*gaggv) / rho) if rho > 0.0 else 0.0
    Vbh = (Vbv + Vbr) / 2.0
    Vsh = (Vsv + Vsr) / 2.0
    Vph = (Vpv + Vpr) / 2.0

    Vdeb3   = 2.0/3.0/Vsh**3 + 1.0/3.0/Vph**3 if (Vsh != 0.0 and Vph != 0.0) else 0.0
    Vdebold = (1.0 / Vdeb3**(1.0/3.0) if Vdeb3 > 0.0
               else 1.0 / abs(Vdeb3)**(1.0/3.0) if Vdeb3 < 0.0 else 0.0)

    dlnvsdlnv  = (0.5 * (1.0 - btaggh / gaggh * s.Gshp)
                  if gaggh != 0.0 else 0.0)
    dlnvpdlnv  = (0.5 * (1.0 - btaggh / (baggh + 4.0/3.0*gaggh) * (s.Ksp + 4.0/3.0*s.Gshp))
                  if (baggh + 4.0/3.0*gaggh) != 0.0 else 0.0)
    dlnvdebdlnv = (2.0/3.0*(s.Vdeb/Vsh)**3*dlnvsdlnv
                   + 1.0/3.0*(s.Vdeb/Vph)**3*dlnvpdlnv
                   if (Vsh != 0.0 and Vph != 0.0) else 0.0)
    gamdeb = -(dlnvdebdlnv - 1.0/3.0)
    gamp   = -(dlnvpdlnv   - 1.0/3.0)

    Vshashp = ex.asqrt(ghashp / rho) if rho > 0.0 else 0.0
    Vshashm = ex.asqrt(ghashm / rho) if rho > 0.0 else 0.0
    Vshasha = ex.asqrt(0.5*(ghashp + ghashm) / rho) if rho > 0.0 else 0.0
    Vphashp = ex.asqrt((bhashp + 4.0/3.0*ghashp) / rho) if rho > 0.0 else 0.0
    Vphashm = ex.asqrt((bhashm + 4.0/3.0*ghashm) / rho) if rho > 0.0 else 0.0
    Vphasha = ex.asqrt(0.5*(bhashp + 4.0/3.0*ghashp + bhashm + 4.0/3.0*ghashm) / rho) if rho > 0.0 else 0.0

    alpagg  = alpagg / volagg if volagg != 0.0 else 0.0
    if alpagg != 0.0 and volagg != 0.0:
        delagg = delagg * baggr / (volagg * alpagg) - 1.0
    dlnvbdt  = alpagg / 2.0 * (delagg - 1.0)
    cpagg    = cpagg  / wmagg if wmagg != 0.0 else 0.0
    cvagg    = cvagg  / wmagg if wmagg != 0.0 else 0.0
    gruagg   = (1000.0 * baggh * alpagg / (rho * cpagg)
                if (rho != 0.0 and cpagg != 0.0) else 0.0)
    dgdtaggv = dgdtaggv / volagg if volagg != 0.0 else 0.0
    dgdtaggr = dgdtaggr / volagg * gaggr**2 if volagg != 0.0 else 0.0
    dgdtagg  = 0.5 * (dgdtaggv + dgdtaggr)
    dlnvsdt  = 0.5 * (dgdtagg / gaggh + alpagg) if gaggh != 0.0 else 0.0
    dlnvpdt  = (0.5 * ((-delagg*alpagg*baggh + 4.0/3.0*dgdtagg)
                        / (baggh + 4.0/3.0*gaggh) + alpagg)
                if (baggh + 4.0/3.0*gaggh) != 0.0 else 0.0)
    rhoo     = wmagg / vabsoagg if vabsoagg != 0.0 else 0.0
    vdebagg  = vdebagg / volagg if volagg != 0.0 else 0.0

    if s.Ti == TSMALL:
        delagg  = 0.0
        dlnvbdt = 0.0

    s.ent  = entagg
    enthagg = freeagg + s.Ti * entagg
    eintagg = freeagg + s.Ti * entagg - 1000.0 * s.Pi * volagg
    enth    = enthagg

    # ------------------------------------------------------------------
    # Aggregate metamorphic contributions
    # ------------------------------------------------------------------
    alpmet = 0.0
    cpmet  = 0.0
    bmet   = 0.0
    for ispec in range(s.nspec):
        alpmet += dndt[ispec] * s.vspeca[ispec] / volagg if volagg != 0.0 else 0.0
        cpmet  += s.Ti * dndt[ispec] * s.sspeca[ispec]
        bmet   += dndp[ispec] * dmdp[ispec]

    btot   = volagg / (volagg/btaggr + bmet) if (volagg/btaggr + bmet) != 0.0 else 0.0
    alptot = alpagg + alpmet
    cptot  = cpagg + cpmet / wmagg if wmagg != 0.0 else 0.0
    cvtot  = cptot - s.Ti * volagg * alptot**2 * btot / wmagg * 1000.0 if wmagg != 0.0 else 0.0
    gamtot = (volagg * alptot * btot / (cvtot * wmagg) * 1000.0
              if (cvtot != 0.0 and wmagg != 0.0) else 0.0)
    bstot  = btot * (1.0 + alptot * gamtot * s.Ti)

    # Isomorphic (non-metamorphic) contributions
    btiso  = btaggr
    cpiso  = cpagg
    alpiso = alpagg
    cviso  = cpiso - s.Ti * volagg * alpiso**2 * btiso / wmagg * 1000.0 if wmagg != 0.0 else 0.0
    gamiso = (volagg * alpiso * btiso / (cviso * wmagg) * 1000.0
              if (cviso != 0.0 and wmagg != 0.0) else 0.0)
    bsiso  = btiso * (1.0 + alpiso * gamiso * s.Ti)
    if s.Ti <= 0.0:
        bsiso = btiso

    # Phase transition properties
    deltavol = ex.ddot(s.nspec, dndp, s.ione, s.vspeca, s.ione)
    deltaent = ex.ddot(s.nspec, dndp, s.ione, s.sspeca, s.ione)

    dfdpabstotal   = 0.0
    dfdptotal      = 0.0
    fnaggtransform = 0.0
    for iph in range(s.nph):
        if dfdp[iph] > DFDPSMALL:
            dfdptotal += dfdp[iph]
        dfdpabstotal   += abs(dfdp[iph])
        fnaggtransform += 2.0 * fnpha[iph] * abs(dfdp[iph])

    if dfdpabstotal != 0.0:
        fnaggtransform /= dfdpabstotal
    if fnaggtransform != 0.0:
        dfdptotal /= fnaggtransform

    phasebuoyancyparameter = 0.0
    if dfdptotal > 0.0:
        phasebuoyancyparameter = alpmet / alpiso / dfdptotal / PMANTLE if (alpiso != 0.0 and PMANTLE != 0.0) else 0.0
    ClapeyronSlope = 0.0
    if abs(deltavol) > ASMALL and phasebuoyancyparameter != 0.0:
        ClapeyronSlope = deltaent / deltavol

    _write(fh, 69, _fmt_f25(s.Pi, ex.depth(s.Pi), s.Ti,
                             phasebuoyancyparameter, ClapeyronSlope,
                             dfdptotal, deltaent, deltavol,
                             alpagg, alpmet,
                             alpmet/alpagg if alpagg != 0.0 else 0.0,
                             alptot,
                             alptot/alpagg if alpagg != 0.0 else 0.0,
                             cvtot, bstot))

    _write(fh, 31, f"alpmet,alpagg,alptot "
                   f"{alpmet} {alpagg} {alptot} "
                   f"{phasebuoyancyparameter} {dfdptotal} {PMANTLE}")
    _write(fh, 31, f"Delta S, Delta V, Gamma = {deltaent} {deltavol} {ClapeyronSlope}")
    _write(fh, 31, f"Pi,X_Fe,bmet,btaggh,btot,bstot "
                   f"{s.Pi} {s.b[1]} {bmet} {btaggh} {btot} {bstot}")
    _write(fh, 31, f"cpmet,cpagg,cptot,cvtot,gamtot "
                   f"{cpmet/wmagg if wmagg != 0.0 else 0.0} "
                   f"{cpagg} {cptot} {cvtot} {gamtot}")

    dhdtmol = cpagg * wmagg / 1000.0   # note: Fortran overwrites with cpagg version
    dsdtmol = cpagg * wmagg / s.Ti     if s.Ti != 0.0 else 0.0
    s.dhdtmol = dhdtmol
    s.dsdtmol = dsdtmol

    # ------------------------------------------------------------------
    # Anelastic velocity reduction
    # ------------------------------------------------------------------
    qs, qp       = ex.qr19(ex.depth(s.Pi), s.Ti)
    vsred, vpred = ex.vred(qs, qp)

    # ------------------------------------------------------------------
    # Extended output block (iprint == 1)
    # ------------------------------------------------------------------
    if iprint == 1:
        fnphamax = -1.0e15
        iimax    = s.nphasep - 1   # 0-based index of last (dummy) phase
        for ii in range(s.nph):
            if fnpha[ii] > fnphamax:
                fnphamax = fnpha[ii]
                iimax    = ii

        tmelt    = ex.tlindeman(s.vol, wmagg, fnagg, s.thet)
        wmav     = wmagg / fnph if fnph != 0.0 else 0.0
        epsilon  = ((epsilon2 - 2.0*wmav*epsilon1 + fnph*wmav**2)
                    / (fnph * wmav**2)
                    if (fnph != 0.0 and wmav != 0.0) else 0.0)

        tcalagg_val = ex.thetacal(cvagg * wmagg / (3.0 * fnagg * s.Rgas)
                                  if fnagg != 0.0 else 0.0)
        tcalagg_val *= s.Ti

        """
        # fort.56
        _write(fh, 56, _fmt_f25(s.Pi, ex.depth(s.Pi), s.Ti, rho,
                                 Vbh, Vsh, Vph, Vsh*vsred, Vph*vpred,
                                 enthagg/wmagg/1000. if wmagg != 0.0 else 0.0,
                                 entagg/wmagg        if wmagg != 0.0 else 0.0,
                                 1.0e5*alptot, cptot, bstot, btot, qp, rhoo,
                                 s.phname[iimax]))

        # fort.59
        _write(fh, 59, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti, volagg,
                                  bsiso, btiso, 1.0e5*alpiso, cpiso,
                                  s.thet, gamiso, s.qq, s.Vdeb,
                                  s.ph, s.pzp, tmelt))

        # fort.57
        _write(fh, 57, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti, volagg,
                                  s.fn, s.zu, 1000.*Vdebold, gruagg, s.qq, wmav,
                                  epsilon, cvagg*wmagg/(3.*fnagg*s.Rgas) if fnagg != 0.0 else 0.0,
                                  tcalagg_val, gamdeb, s.fn, fnph,
                                  fnagg, s.Cv/(3*s.fn*s.Rgas) if s.fn != 0.0 else 0.0,
                                  s.thet))

        # fort.58
        _write(fh, 58, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti, rho,
                                  baggh, gaggh, Vbh, Vsh, Vph,
                                  baggr, gaggr, Vbr, Vsr, Vpr,
                                  baggv, gaggv, Vbv, Vsv, Vpv))

        # Phase-property files (61-68)
        _write(fh, 61, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                  *[math.exp(math.log(max(rhopha[iph], ASMALL+1e-300)))
                                    for iph in range(s.nph)]))
        _write(fh, 62, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                  *[math.exp(math.log(max(vbpha[iph], ASMALL+1e-300)))
                                    for iph in range(s.nph)]))
        _write(fh, 63, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                  *[math.exp(math.log(max(vspha[iph], ASMALL+1e-300)))
                                    for iph in range(s.nph)]))
        _write(fh, 64, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                  *[math.exp(math.log(max(vppha[iph], ASMALL+1e-300)))
                                    for iph in range(s.nph)]))
        _write(fh, 65, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                  *[math.exp(math.log(max((wmpha[iph]-ASMALL)/fnpha[iph], ASMALL+1e-300)))
                                    if fnpha[iph] > 0.0 else 0.0
                                    for iph in range(s.nph)]))
        _write(fh, 68, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                  *[math.exp(math.log(max(volpha[iph], ASMALL+1e-300)))
                                    for iph in range(s.nph)]))

        # Phase proportions (66, 661, 67)
        _write(fh, 66,  _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                   *[fnpha[iph]/fnagg if fnagg != 0.0 else 0.0 for iph in range(s.nph)]))
        _write(fh, 661, _fmt_f12(s.Pi, ex.depth(s.Pi), s.Ti,
                                   *[dfdp[iph]/fnagg if fnagg != 0.0 else 0.0 for iph in range(s.nph)]))
        _write(fh, 67,  _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                   *[vabs[iph]/vabsagg if vabsagg != 0.0 else 0.0 for iph in range(s.nph)]))

        _write(fh, 31, '\nPhase proportions: atomic, volume, mass')
        _write(fh, 31, ''.join(f"{'Pi':>12}{'depth':>12}{'Ti':>12}") +
               ''.join(f"{s.phname[iph][:8]:>12}" for iph in range(s.nph)))
        _write(fh, 31, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                  *[fnpha[iph]/fnagg if fnagg != 0.0 else 0.0 for iph in range(s.nph)]))
        _write(fh, 31, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                  *[vabs[iph]/vabsagg if vabsagg != 0.0 else 0.0 for iph in range(s.nph)]))
        _write(fh, 31, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti,
                                  *[wmpha[iph]/wmagg if wmagg != 0.0 else 0.0 for iph in range(s.nph)]))

        _write(fh, 31, '\nModuli and Velocities')
        _write(fh, 31,
               f"{'Pi':>6}{'depth':>8}{'Ti':>8}"
               + ''.join(f"{h:>12}" for h in
                         ['rho','KS','G ','VBh','VSh','VPh','VBr','VSr','VPr','VBv','VSv','VPv']))
        _write(fh, 31, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti, rho,
                                  baggh, gaggh, Vbh, Vsh, Vph, Vbr, Vsr, Vpr, Vbv, Vsv, Vpv))

        _write(fh, 31, '\nPhysical Properties')
        _write(fh, 31,
               f"{'Pi':>6}{'depth':>8}{'Ti':>8}"
               + ''.join(f"{h:>12}" for h in
                         ['rho','KS','KT','alpha','cP','g','S(J/g)','H(kJ/g)']))
        _write(fh, 31, _fmt_f16(s.Pi, ex.depth(s.Pi), s.Ti, rho,
                                  bstot, btot, 1.0e5*alptot, cptot,
                                  gamtot,
                                  entagg/wmagg  if wmagg != 0.0 else 0.0,
                                  enthagg/wmagg/1000. if wmagg != 0.0 else 0.0,
                                  wmagg))

        _write(fh, 31, '\nPhysical properties of individual phases')
        _write(fh, 31,
               f"{'phase':>5}" +
               ''.join(f"{h:>13}" for h in
                       ['rho','VB','VS','VP','G','vol','ent','H ','G',
                        'S0','H0','G0','volo','KT','KS']))
        for iph in range(s.nph):
            if fnpha[iph] < 1.0e-12:
                continue
            _write(fh, 31,
                   f"{s.phname[iph]:5s}" +
                   ''.join(f"{v:13.7f}" for v in [
                       rhopha[iph], vbpha[iph], vspha[iph], vppha[iph],
                       gpha[iph], volpha[iph], entpha[iph], henpha[iph],
                       gibpha[iph], entopha[iph], henopha[iph], gibopha[iph],
                       vabso[iph]/fnpha[iph]*5 if fnpha[iph] != 0.0 else 0.0,
                       btpha[iph], bpha[iph]]))"""

        # Oxygen fugacity
        for ic in range(s.nc):
            if s.comp[ic].strip() == 'O':
                #_write(fh, 31, f"oxygen chemical potential = {s.lagc[ic]}")
                Ti = s.Ti
                ho2gas = (_AO2*(Ti-_TRO2) + 0.5*_BO2*(Ti**2-_TRO2**2)
                          - _CO2*(1./Ti - 1./_TRO2)
                          + 2.*_DO2*(math.sqrt(Ti) - math.sqrt(_TRO2)))
                so2gas = (205.15 + _AO2*math.log(Ti/_TRO2) + _BO2*(Ti-_TRO2)
                          - 0.5*_CO2*(1./Ti**2 - 1./_TRO2**2)
                          - 2.*_DO2*(1./math.sqrt(Ti) - 1./math.sqrt(_TRO2)))
                go2gas = (ho2gas - Ti*so2gas + _STRO2*_TRO2) / 1000.0
                fug = (1000.0*(2.*s.lagc[ic] - go2gas) / (s.Rgas*Ti)) / math.log(10.0)

                #_write(fh, 31, f"oxygen gas enthalpy, entropy, gibbs (kJ/mol) = "
                #               f"{Ti} {ho2gas} {so2gas} {go2gas}")
                #_write(fh, 31, f"2.*lagc(ic) - go2gas = {2.*s.lagc[ic] - go2gas}")
                #_write(fh, 31, f"log_10 oxygen fugacity = {fug}")

        # Phase compositions — cations
        #_write(fh, 31, '\nPhase Compositions - Cations')
        #_write(fh, 31, ''.join(f"{s.comp[i]:>12}" for i in range(s.nco)) +
        #       f"{'XFe':>12}{'Fe3+':>12}")

        img = s.ncompp - 1   # 0-based sentinel
        ife = s.ncompp - 1
        iox = s.ncompp - 1
        ip  = [s.ncompp - 1] * s.ncompp

        for ic in range(s.ncompp):
            if s.comp[ic].strip() == 'Mg':
                img = ic
            if s.comp[ic].strip() == 'Fe':
                ife = ic
            if s.comp[ic].strip() == 'O':
                iox = ic
            for jc in range(NOXIDES):
                if s.comp[ic].strip() == COMPSTAND[jc]:
                    ip[ic] = jc

        st_arr  = np.zeros(s.ncompp)
        wco_arr = np.zeros((s.nphasep, s.ncompp))

        for iph in range(s.nph):
            stfox  = 0.0
            st_arr[:] = 0.0
            wco_arr[iph, :] = 0.0

            if fnpha[iph] < 1.0e-12:
                continue

            ssum = 0.0
            for ispec in range(s.nspec):
                if s.f[iph, ispec] == 0:
                    continue
                wsum_loc = 0.0
                for ic in range(s.nco):
                    st_arr[ic] += s.s_stoich[ic, ispec] * s.n[ispec]
                    if ip[ic] != 2 and ip[ic] < NOXIDES:   # ip(ic) .ne. 3 (1-based) -> != 2 (0-based)
                        stfox += s.s_stoich[ic, ispec] * s.n[ispec] * NOXYGEN[ip[ic]]
                    if ispec == s.iphase[iph] - 1:         # iphase is 1-based in Fortran
                        ssum += s.s_stoich[ic, ispec]
                    if s.comp[ic].strip() != 'O':
                        wco_arr[iph, ic] += (s.s_stoich[ic, ispec] * s.n[ispec]
                                             * s.wcomp[ic] / s.stcomp[ic]
                                             if s.stcomp[ic] != 0.0 else 0.0)

            stfox = st_arr[iox] - stfox
            stsum = np.sum(st_arr[:s.nco])
            wsum  = sum(wco_arr[iph, ic] for ic in range(s.nco)
                        if s.comp[ic].strip() != 'O')

            if stsum != 0.0:
                for ic in range(s.nco):
                    st_arr[ic] = st_arr[ic] / stsum * ssum
                    if s.comp[ic].strip() != 'O':
                        wco_arr[iph, ic] = wco_arr[iph, ic] / wsum if wsum != 0.0 else 0.0
                stfox = stfox / stsum * ssum

            xfe = 0.0
            fe3 = 0.0
            if ife < s.ncompp and img < s.ncompp:
                denom = st_arr[ife] + st_arr[img]
                if denom > 0.0:
                    xfe = st_arr[ife] / denom
            if ife < s.ncompp and iox < s.ncompp:
                if st_arr[ife] > 0.0 and st_arr[iox] > 0.0:
                    fe3 = st_arr[ife] * 2.0 * (stfox / st_arr[ife] - 1.0)

            #_write(fh, 31,
            #       f"{s.phname[iph]:5s}" +
            #       ''.join(f"{st_arr[ic]:12.5f}" for ic in range(s.nco)) +
            #       f"{xfe:12.5f}{fe3:12.5f}")

        # Phase compositions — mass standard oxides
        #_write(fh, 31, '\nPhase Compositions - Mass Standard Oxides (%)')
        #_write(fh, 31, ''.join(f"{s.comp[i]:>12}" for i in range(s.nco)))
        #for iph in range(s.nph):
        #    if fnpha[iph] < 1.0e-12:
        #        continue
        #    _write(fh, 31,
        #           f"{s.phname[iph]:5s}" +
        #           ''.join(f"{100.0*wco_arr[iph, ic]:12.5f}" for ic in range(s.nco)))

    # ------------------------------------------------------------------
    # Finalise and save all computed properties to state
    # ------------------------------------------------------------------
    s.iphyflag = 0

    # Save aggregate properties to state object for downstream access
    s.vol = volagg
    s.Cap = cpagg
    s.Cv = cvagg
    s.alp = alpagg
    s.gamma = gruagg
    s.K = baggh
    s.Ks = btaggh
    s.Gsh = gaggh
    s.deltas = delagg if volagg > 0.0 else 0.0
    s.ent = entagg
    s.Ftot = freeagg
    s.Ksp = bstot
    s.Gshp = gamtot
    s.fn = fnagg
    s.zu = wmagg
    s.Vdeb = vdebagg
    s.gamdeb = gamdeb
    s.phugo = enth
    
    # Also save velocity/modulus tensors (Hill averages for bulk returns)
    s.tcal = ex.thetacal(cvagg * wmagg / (3.0 * fnagg * s.Rgas)
                         if fnagg != 0.0 else 0.0) * s.Ti if s.Ti > 0.0 else 0.0
    s.zeta = vdebagg
    s.uth = (cvagg * s.Ti) if cvagg > 0.0 else 0.0
    s.uto = cvagg if cvagg > 0.0 else 0.0

    finish    = time.process_time()
    _gphysub += finish - start
    #_write(fh, 31, f"cumulative time in physub {_gphysub} {_ncall}")

    # Return comprehensive bulk property dictionary
    return {
        'rho': rho,
        'wmagg': wmagg,
        'freeagg': freeagg,
        'vol': volagg,
        'Cp': cpagg,
        'Cv': cvagg,
        'alpha': alpagg,
        'gamma': gruagg,
        'K_Hill': baggh,
        'K_S': btaggh,
        'G_Hill': gaggh,
        'Vb_Hill': Vbh,
        'Vs_Hill': Vsh,
        'Vp_Hill': Vph,
        'Vb_Reuss': Vbr,
        'Vs_Reuss': Vsr,
        'Vp_Reuss': Vpr,
        'Vb_Voigt': Vbv,
        'Vs_Voigt': Vsv,
        'Vp_Voigt': Vpv,
        'K_Reuss': baggr,
        'K_Voigt': baggv,
        'G_Reuss': gaggr,
        'G_Voigt': gaggv,
        'entropy': entagg,
        'enthalpy': enth,
        'theta_Debye': s.tcal,
        'gamma_Debye': gamdeb,
        'Vs_anelastic': vsred,
        'Vp_anelastic': vpred,
        'Q_shear': qs,
        'Q_pressure': qp,
    }


# ---------------------------------------------------------------------------
# Internal formatting helpers (replace Fortran FORMAT statements)
# ---------------------------------------------------------------------------

def _write(fh, unit, text):
    """Write text to the file handle mapped to `unit`, if present."""
    if unit in fh and fh[unit] is not None:
        fh[unit].write(str(text) + '\n')


def _fmt_f25(*args):
    """Format floats in 25-wide fields (format 500 / 700 equivalent)."""
    parts = []
    for a in args:
        if isinstance(a, float):
            parts.append(f"{a:25.16f}")
        else:
            parts.append(f"{str(a):>25}")
    return ''.join(parts)
