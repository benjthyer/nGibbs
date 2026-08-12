"""
Metamorphic (latent-heat / phase-change) contributions to the aggregate
thermodynamic derivatives, vectorised over a batch of assemblages.

WHAT THIS MODULE ADDS
---------------------
The isomorphic EOS in ``compute.py`` evaluates every property at *fixed*
species abundances ``n``.  HeFESTo additionally accounts for the fact that, as
P or T is varied, the equilibrium assemblage itself changes -- ``dn/dT`` and
``dn/dP`` are non-zero -- and this reaction term contributes latent heat to
Cp, softens K_T, and inflates alpha.  In ``physub.f`` these are::

    alpmet = sum_i dndt_i * V_i / volagg          (physub.f:591)
    cpmet  = sum_i Ti * dndt_i * S_i              (physub.f:592)  <-- LATENT HEAT
    bmet   = sum_i dndp_i * dmdp_i                (physub.f:593)

and then (physub.f:596-601)::

    alptot = alpagg + alpmet
    cptot  = cpagg  + cpmet / wmagg
    btot   = volagg / (volagg/btaggr + bmet)
    cvtot  = cptot - Ti*volagg*alptot^2*btot/wmagg*1000
    gamtot = volagg*alptot*btot/(cvtot*wmagg)*1000
    bstot  = btot*(1 + alptot*gamtot*Ti)

``fort.56`` reports ``alptot``, ``cptot``, ``bstot``, ``btot`` -- i.e. the
*metamorphic-inclusive* values.  ``fort.59`` reports the isomorphic
counterparts ``alpiso``, ``cpiso``, ``bsiso``, ``btiso``, which is what a
fixed-``n`` EOS such as ``compute.py`` reproduces.  ``fort.69`` reports
``alpagg, alpmet, alptot, cvtot, bstot`` at full precision.

HOW dn/dT AND dn/dP ARE OBTAINED
--------------------------------
Not by finite differences.  HeFESTo solves a linear system (``physub.f``
lines 156-178)::

    dmdt_i = S_i / 1000        (kJ/K/mol)   ! sign pre-flipped, see physub.f:161
    dmdp_i = -V_i              (kJ/GPa/mol)

    H^P = Q2^T H Q2            projected Hessian   (hessfunc.f)
    H^P x = Q2^T dmdt          solved by SVD pseudo-inverse (svdsub.f)
    dn/dT = Q2 x

where ``H_ij = d2G/dn_i dn_j`` and ``Q2`` is an orthonormal basis for the null
space of the bulk-composition constraint (i.e. the directions in species space
that conserve bulk chemistry).

Two facts make this cheap and exactly reproducible:

1.  **H contains no EOS second derivatives.**  End-member Gibbs energy is
    linear in ``n``, so it drops out of ``d2G/dn dn`` entirely.  ``hessian.f``
    shows H comes purely from the *mixing model*: the ideal configurational
    term (site occupancies + the ``iastate`` ordering mask) and the
    asymmetric-regular-solution excess term (``wreg``/``vreg`` + Van Laar size
    parameters).  No volume solve, no Debye integrals.

2.  **H is block-diagonal by phase.**  Every term in ``hessian.f`` carries an
    ``f(iph,jspec)`` factor, so species in different phases never couple.  We
    therefore never materialise a (B, nspec, nspec) tensor; we keep per-phase
    (B, m, m) blocks with m <= 7.

Absent species are excluded exactly the way ``sform.f`` does it: an extra
constraint row ``n_j = 0`` per absent species, which is equivalent to taking
the null space of the stoichiometry matrix restricted to *active* columns.
Since the active set is piecewise-constant across a batch, we group rows by
their active-species pattern and build one Q2 per distinct pattern.

NOT IMPLEMENTED (deliberate, flagged for later)
-----------------------------------------------
* **Anelastic attenuation** (``qr19.f`` / ``vred.f``).  Applies only to the
  ``VSQ``/``VPQ`` columns of fort.56; see ``qr19.py``, already ported.
* **Landau composition-dependent term** (``cp.f`` lines 109-138).  It is
  computed but *not* added to ``chempot`` in the Fortran (line 138 is commented
  out), so it does not enter H either.
* ``dfdp`` / ``phasebuoyancyparameter`` / ``ClapeyronSlope`` (physub.f:610-631).
  These are free by-products of ``dndp``; see ``clapeyron_terms``.

Units
-----
    n, dn        mol
    T            K
    P            GPa
    H            kJ / mol^2
    S_i          J / mol / K
    V_i          cm^3/mol  == kJ / (mol GPa)
    dndt         mol / K
    dndp         mol / GPa
    alp*         1/K
    cp*          J / g / K
    b* (moduli)  GPa
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import Rgas
from .params import HeFESToParams

# ── Parity constants, lifted verbatim from the Fortran ──────────────────────
SMALL_SVD = 1.0e-6    # svdsub.f:14   singular-value cutoff (ABSOLUTE, not relative)
NSMALL    = 1.0e-16   # physub.f:160  nsmall/10, the absent-species floor on n
TINY      = 1.0e-30   # guard for logs / divisions, matching aggregate.py


# ════════════════════════════════════════════════════════════════════════════
# Static tables
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MetamorphicTables:
    """Pre-computed, batch-independent tables.  Build once per parameter set."""

    nspec         : int
    elements      : List[str]                 # element symbols, incl. O
    s_mat         : np.ndarray                # (n_el, nspec) element stoichiometry
    phase_members : List[List[int]]           # phase_members[iph] = [global ispec, ...]
    lphase        : np.ndarray                # (nspec,)

    # site_tables[iph][kst] = dict(q=(m, n_el_site), incl=(m, m) bool, elems=[str])
    #   q[i, e]  : stoichiometric coefficient of element e for member i at site kst
    #   incl[i,j]: False where iastate(i,j,kst) is True (j excluded from i's nikp)
    site_tables   : List[List[dict]]

    wreg          : np.ndarray                # (nspec, nspec) kJ/mol   W_ij and O_ji
    vreg          : np.ndarray                # (nspec, nspec) cm^3/mol
    size          : np.ndarray                # (nspec,) Van Laar size, apar[:, 40]

    # per phase: local (a, b) index pairs with any non-zero W/O/V -- the rest
    # contribute nothing and are skipped.
    excess_pairs  : List[List[Tuple[int, int]]]

    # per phase: local indices of the order-disorder ("fast") species block,
    # equivalent to iophase(iph) .. iophase(iph)+mophase(iph)-1 (readin.f:222-241).
    od_members    : List[List[int]]

    spec_names    : List[str] = field(default_factory=list)


_ELEM_RE = re.compile(r'([A-Z][a-z]*)_(\d+)')


def _parse_formula_elements(formula: str) -> Dict[str, int]:
    """Total element counts for a HeFESTo formula string, oxygen included.

    ``'(Na_2Mg_1)Si_1Si_3O_12'`` -> ``{'Na': 2, 'Mg': 1, 'Si': 4, 'O': 12}``
    """
    counts: Dict[str, int] = {}
    for el, n in _ELEM_RE.findall(formula):
        counts[el] = counts.get(el, 0) + int(n)
    return counts


def _read_formulas(param_dir: str, snames: Sequence[str]) -> List[str]:
    """First whitespace token of the first line of each species parameter file."""
    formulas: List[str] = []
    for sname in snames:
        fpath = os.path.join(param_dir, sname)
        with open(fpath, 'r') as fh:
            first = fh.readline().split()
        formulas.append(first[0] if first else '')
    return formulas


def load_regular_solution(param_dir: str,
                          params: HeFESToParams) -> Tuple[np.ndarray, np.ndarray]:
    """Port of ``regread.f``.

    Reads ``<param_dir>/PHASE/<phase_name>`` for every multi-species phase and
    returns dense ``(nspec, nspec)`` matrices.  Element ``[i, j]`` with i<j in
    control-file order is the symmetric interaction ``W_ij``; ``[j, i]`` is the
    asymmetric term ``O_ij`` (HeFESTo's "o" parameter).

    Missing phase files mean ideal mixing (regread.f:59), which is not an error.
    """
    S = params.nspec
    wreg = np.zeros((S, S), dtype=np.float64)
    vreg = np.zeros((S, S), dtype=np.float64)

    phase_dir = os.path.join(param_dir, 'PHASE')

    for iph, ph_name in enumerate(params.phase_names):
        members = params.phase_members[iph]
        if len(members) <= 1:          # regread.f:62  -- mphase <= 1, skip
            continue

        fpath = os.path.join(phase_dir, ph_name)
        if not os.path.isfile(fpath):
            continue                    # regread.f:59  "Assuming ideal mixing"

        with open(fpath, 'r') as fh:
            raw = [ln for ln in fh.read().splitlines() if ln.strip()]
        if not raw:
            continue

        header = raw[0].split()
        ncmax = len(header)

        # W block: ncmax rows immediately after the header
        w_rows = [list(map(float, raw[1 + i].split())) for i in range(ncmax)]
        w_blk = np.asarray(w_rows, dtype=np.float64)

        # Optional volume block, introduced by a label line (regread.f:72)
        v_blk = np.zeros_like(w_blk)
        vstart = 1 + ncmax
        if len(raw) > vstart and not _is_number_row(raw[vstart]):
            v_rows = [list(map(float, raw[vstart + 1 + i].split()))
                      for i in range(ncmax)]
            v_blk = np.asarray(v_rows, dtype=np.float64)

        # Map header labels -> global species indices.  regread.f:87-94 scans
        # forward from the last match, so labels must appear in control order.
        ip: List[int] = []
        cursor = 0
        for label in header:
            found = -1
            for k in range(cursor, len(members)):
                if params.snames[members[k]][:4].strip() == label[:4].strip():
                    found = members[k]
                    cursor = k + 1
                    break
            ip.append(found)

        for ia in range(ncmax):
            if ip[ia] < 0:
                continue
            for ib in range(ncmax):
                if ia == ib or ip[ib] < 0:
                    continue
                wreg[ip[ia], ip[ib]] = w_blk[ia, ib]
                vreg[ip[ia], ip[ib]] = v_blk[ia, ib]

        # Backwards compatibility (regread.f:108-125): a symmetric W stored in
        # both triangles means O = 0, not O = W.
        for ia in range(ncmax - 1):
            for ib in range(ia + 1, ncmax):
                A, Bx = ip[ia], ip[ib]
                if A < 0 or Bx < 0:
                    continue
                if wreg[A, Bx] == wreg[Bx, A] and wreg[A, Bx] != 0.0:
                    wreg[Bx, A] = 0.0
                    vreg[Bx, A] = 0.0

    return wreg, vreg


def _is_number_row(line: str) -> bool:
    try:
        [float(tok) for tok in line.split()]
        return True
    except ValueError:
        return False


def build_tables(params: HeFESToParams, param_dir: str) -> MetamorphicTables:
    """Build every batch-independent table needed by this module."""
    S = params.nspec
    formulas = _read_formulas(param_dir, params.snames)

    # ── Element stoichiometry matrix s(ic, ispec) ────────────────────────────
    per_spec = [_parse_formula_elements(f) for f in formulas]
    elements = sorted({el for d in per_spec for el in d})
    s_mat = np.zeros((len(elements), S), dtype=np.float64)
    for j, d in enumerate(per_spec):
        for el, cnt in d.items():
            s_mat[elements.index(el), j] = float(cnt)

    # ── Per-phase, per-site occupancy + iastate exclusion mask ───────────────
    # Reuses aggregate.py's two-layer iastate reconstruction verbatim so the
    # Hessian and the mixing entropy stay mutually consistent.
    site_tables: List[List[dict]] = []
    for members in params.phase_members:
        if len(members) < 1:
            site_tables.append([])
            continue

        member_sites = [params.site_data[s] for s in members]
        n_sites = max((len(sp) for sp in member_sites), default=0)
        m = len(members)

        # Layer 1 -- species with identical site formulas are quantum states of
        # the same compound (wu/wuls, hepv/hlpv); excluded at every site.
        formula_keys = [str(sp) for sp in member_sites]
        global_excl = [
            frozenset(j for j in range(m)
                      if formula_keys[j] == formula_keys[i] and j != i)
            for i in range(m)
        ]

        per_site: List[dict] = []
        for kst in range(n_sites):
            spec_comps = [
                (sp[kst] if kst < len(sp) else []) for sp in member_sites
            ]

            # Layer 2 -- iron co-occupancy on this site (readin.f:255-279).
            has_fe = [any(el == 'Fe' for el, _ in c) for c in spec_comps]
            excl = np.zeros((m, m), dtype=bool)
            for i in range(m):
                for j in global_excl[i]:
                    excl[i, j] = True
            for i in range(m):
                for j in range(m):
                    if i == j or excl[i, j]:
                        continue
                    if has_fe[i] and has_fe[j]:
                        excl[i, j] = True

            site_elems = sorted({el for c in spec_comps for el, _ in c})
            q = np.zeros((m, len(site_elems)), dtype=np.float64)
            for i, comps in enumerate(spec_comps):
                for el, cnt in comps:
                    q[i, site_elems.index(el)] += float(cnt)

            per_site.append(dict(q=q, incl=~excl, elems=site_elems))

        site_tables.append(per_site)

    # ── Regular-solution parameters ─────────────────────────────────────────
    wreg, vreg = load_regular_solution(param_dir, params)
    size = params.apar[:, 40].copy()   # 1-indexed col 41 == Van Laar size

    excess_pairs: List[List[Tuple[int, int]]] = []
    for members in params.phase_members:
        idx = np.asarray(members, dtype=np.int64)
        pairs: List[Tuple[int, int]] = []
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                ga, gb = idx[a], idx[b]
                if (wreg[ga, gb] or wreg[gb, ga]
                        or vreg[ga, gb] or vreg[gb, ga]):
                    pairs.append((a, b))
        excess_pairs.append(pairs)

    # ── Order-disorder ("fast") blocks, readin.f:222-241 ────────────────────
    od_members: List[List[int]] = []
    for members in params.phase_members:
        iophase_local = -1
        mophase = 0
        for a in range(len(members) - 1):
            for b in range(a + 1, len(members)):
                if np.array_equal(s_mat[:, members[a]], s_mat[:, members[b]]):
                    if mophase == 0:
                        mophase = 1
                        iophase_local = a
                    mophase += 1
        if mophase > 0:
            od_members.append(
                [members[k] for k in
                 range(iophase_local, min(iophase_local + mophase, len(members)))]
            )
        else:
            od_members.append([])

    return MetamorphicTables(
        nspec=S,
        elements=elements,
        s_mat=s_mat,
        phase_members=params.phase_members,
        lphase=np.asarray(params.lphase),
        site_tables=site_tables,
        wreg=wreg,
        vreg=vreg,
        size=size,
        excess_pairs=excess_pairs,
        od_members=od_members,
        spec_names=list(params.snames),
    )


# ════════════════════════════════════════════════════════════════════════════
# Mixing quantities (partial molar) -- port of cp.f
# ════════════════════════════════════════════════════════════════════════════

def mixing_terms(n: np.ndarray,
                 P: np.ndarray,
                 tables: MetamorphicTables,
                 *,
                 active: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Partial molar ideal-mixing entropy and excess volume (``cp.f``).

    Returns
    -------
    smix : (B, S) J/mol/K
        ``smixi`` in cp.f:105.  ``sspeca = sspeco + smixi + smag``  (func.f:92).
    volsum : (B, S) cm^3/mol
        ``volsum`` in cp.f:95.  ``vspeca = vspeco + volsum``        (func.f:93).

    Note
    ----
    The ``len(nikp_i) < 2`` short-circuit is kept for bit-parity with
    ``aggregate.py``'s entropy path (there is no such guard in cp.f; it is a
    no-op except on sites where ``iastate`` masking leaves a species seeing a
    single element while ``nkp`` still counts the masked partners).
    """
    B = n.shape[0]
    S = tables.nspec
    if active is None:
        active = n > 0.0
    n = np.maximum(n, 0.0)          # fort.99 carries -0.0 / tiny negatives
    smix = np.zeros((B, S), dtype=np.float64)
    volsum = np.zeros((B, S), dtype=np.float64)

    for iph, members in enumerate(tables.phase_members):
        if len(members) < 2:
            continue
        idx = np.asarray(members, dtype=np.int64)
        X = n[:, idx]                                    # (B, m)
        m = len(members)

        # ── ideal configurational part ──────────────────────────────────────
        for site in tables.site_tables[iph]:
            q, incl = site['q'], site['incl']            # (m, ne), (m, m)
            if q.shape[1] == 0:
                continue
            nkp = X @ q.sum(axis=1)                      # (B,)
            snkp = np.maximum(nkp, TINY)
            # nikp[b, i, e] = sum_j incl[i,j] * q[j,e] * X[b,j]
            nikp = np.einsum('ij,je,bj->bie', incl.astype(np.float64), q, X)
            snikp = np.maximum(nikp, TINY)

            contrib = np.einsum('ie,bie->bi', q,
                                np.log(snkp[:, None, None] / snikp))
            # parity guard (see docstring)
            n_elems_seen = (np.einsum('ij,je->ie',
                                      incl.astype(np.float64), q) > 0).sum(axis=1)
            contrib[:, n_elems_seen < 2] = 0.0
            smix[:, idx] += Rgas * contrib

        # ── excess (asymmetric regular solution) volume, cp.f:95 ────────────
        pairs = tables.excess_pairs[iph]
        if not pairs:
            continue
        sz = tables.size[idx]                            # (m,)
        msum = X @ sz
        nsum = X @ np.abs(sz)
        ok = nsum > 0.0
        snsum = np.where(ok, nsum, 1.0)
        ratio = np.where(ok, msum / snsum, 0.0)          # (B,)

        for (a, b) in pairs:
            ga, gb = idx[a], idx[b]
            qa = np.abs(sz[a]) * X[:, a] / snsum         # cp.f:81 uses abs()
            qb = np.abs(sz[b]) * X[:, b] / snsum
            scale = 2.0 * np.abs(sz) / (abs(sz[a]) + abs(sz[b]))   # (m,)

            v_ab = tables.vreg[ga, gb]
            v_ba = tables.vreg[gb, ga]
            vregsz = (v_ab + v_ba * (qb - qa))[:, None] * scale[None, :]

            da = np.eye(m)[a][None, :] - qa[:, None]     # (B, m)
            db = np.eye(m)[b][None, :] - qb[:, None]
            volsum[:, idx] -= np.where(ok[:, None], 1.0, 0.0) * \
                da * db * vregsz * ratio[:, None]

    # Absent species can pick up a divergent log ratio (nikp -> 0 while nkp
    # does not).  They carry dndt = dndp = 0 by construction, but zeroing here
    # keeps 0 * inf out of the downstream sums.
    smix = np.where(active, smix, 0.0)
    volsum = np.where(active, volsum, 0.0)
    return smix, volsum


# ════════════════════════════════════════════════════════════════════════════
# Hessian -- port of hessian.f, kept block-diagonal by phase
# ════════════════════════════════════════════════════════════════════════════

def hessian_blocks(n: np.ndarray,
                   T: np.ndarray,
                   P: np.ndarray,
                   tables: MetamorphicTables) -> List[np.ndarray]:
    """``d2G/dn_i dn_j`` in kJ/mol^2, as one (B, m, m) block per phase.

    Port of ``hessian.f``.  Two blocks:

    * ideal configurational: ``-Ti*Rgas/1000 * [ -sum2a + sum1*sum1a/nkp ]``
    * asymmetric regular solution: ``rsuma``, evaluated on site 1 only
      (hessian.f:69 ``if (kst .eq. 1)``).

    Parity notes
    ------------
    * ``hessian.f`` uses *signed* size parameters for ``qa``/``qb`` and omits the
      ``ratio = msum/nsum`` factor, whereas ``cp.f`` uses ``abs()`` and includes
      ``ratio``.  That asymmetry is in the Fortran, not here; it is reproduced.
    * ``hessian.f`` restores ``wreg`` to its pressure-free value at line 88-89
      *before* the ``do 7`` loop, so the ``O`` appearing at line 95 carries no
      ``P*vreg`` term even though ``wregsz``/``oregsz`` (lines 86-87) do.
      Reproduced verbatim below as ``o_raw``.
    """
    B = n.shape[0]
    blocks: List[np.ndarray] = []

    for iph, members in enumerate(tables.phase_members):
        m = len(members)
        if m == 0:
            blocks.append(np.zeros((B, 0, 0)))
            continue

        idx = np.asarray(members, dtype=np.int64)
        # physub.f:160 -- absent species are floored, not removed, so that the
        # 1/nikp terms diverge and freeze them rather than producing NaN.
        X = np.maximum(n[:, idx], NSMALL)
        H = np.zeros((B, m, m), dtype=np.float64)

        # ── ideal configurational term ──────────────────────────────────────
        cfg = np.zeros((B, m, m), dtype=np.float64)
        for site in tables.site_tables[iph]:
            q, incl = site['q'], site['incl']
            if q.shape[1] == 0:
                continue
            inclf = incl.astype(np.float64)

            sum1 = q.sum(axis=1)                                   # (m,)
            nkp = X @ sum1                                          # (B,)
            nikp = np.einsum('ij,je,bj->bie', inclf, q, X)          # (B, m, ne)

            # sum2a[b,i,j] = sum_e incl[i,j]*q[i,e]*q[j,e]/nikp[b,i,e]
            inv = np.where(nikp > 0.0, 1.0 / np.maximum(nikp, TINY), 0.0)
            sum2a = np.einsum('ie,je,bie->bij', q, q, inv) * inclf[None, :, :]

            cfg -= sum2a
            good = nkp > 0.0
            cfg += np.where(good[:, None, None], 1.0, 0.0) * \
                (sum1[None, :, None] * sum1[None, None, :]) / \
                np.where(good, nkp, 1.0)[:, None, None]

        H += -T[:, None, None] * Rgas * cfg / 1000.0

        # ── asymmetric regular solution term (site 1 only) ──────────────────
        pairs = tables.excess_pairs[iph]
        if pairs and tables.site_tables[iph]:
            sz = tables.size[idx]                                   # (m,)
            nsum = X @ sz                                           # signed! (B,)
            ok = nsum > 0.0                                         # hessian.f:74
            snsum = np.where(ok, nsum, 1.0)
            eye = np.eye(m)

            rsuma = np.zeros((B, m, m), dtype=np.float64)
            for (a, b) in pairs:
                ga, gb = idx[a], idx[b]
                qa = sz[a] * X[:, a] / snsum                        # (B,)
                qb = sz[b] * X[:, b] / snsum
                scale = 2.0 * np.abs(sz) / (abs(sz[a]) + abs(sz[b]))  # (m,)

                w_ab = tables.wreg[ga, gb] + P * tables.vreg[ga, gb]   # (B,)
                o_ab = tables.wreg[gb, ga] + P * tables.vreg[gb, ga]   # (B,)
                o_raw = tables.wreg[gb, ga]                            # see docstring

                wregsz = (w_ab + o_ab * (qb - qa))[:, None] * scale[None, :]
                oregsz = o_ab[:, None] * scale[None, :]               # (B, m)

                # d q_a / d n_j  and  d q_b / d n_j          (hessian.f:91-92)
                dqa = sz[a] * (eye[a][None, :] * snsum[:, None]
                               - X[:, a][:, None] * sz[None, :]) / snsum[:, None]**2
                dqb = sz[b] * (eye[b][None, :] * snsum[:, None]
                               - X[:, b][:, None] * sz[None, :]) / snsum[:, None]**2

                da = eye[a][None, :] - qa[:, None]                    # (B, m) index i
                db = eye[b][None, :] - qb[:, None]

                # hessian.f:93-96
                t1 = (-dqb[:, None, :] * da[:, :, None]
                      - dqa[:, None, :] * db[:, :, None]) * wregsz[:, :, None]
                t2 = (da[:, :, None] * db[:, :, None] * o_raw
                      * (dqb[:, None, :] - dqa[:, None, :])
                      * scale[None, :, None])
                rsuma -= (t1 + t2)

                # hessian.f:97-98
                t3 = oregsz[:, :, None] * (
                    (dqa[:, None, :] * qb[:, None, None]
                     + dqb[:, None, :] * qa[:, None, None])
                    * (db - da)[:, :, None]
                    + qa[:, None, None] * qb[:, None, None]
                    * (dqa - dqb)[:, None, :]
                )
                rsuma += t3

            H += np.where(ok[:, None, None], rsuma, 0.0)

        blocks.append(H)

    return blocks


# ════════════════════════════════════════════════════════════════════════════
# Null space of the bulk-composition constraint
# ════════════════════════════════════════════════════════════════════════════

def _nullspace(s_active: np.ndarray) -> np.ndarray:
    """Orthonormal basis for ``null(s_active)`` using svdsub.f's 1e-6 cutoff.

    ``s_active`` is (n_el, k).  Returns (k, nnull).

    The absolute (not relative) threshold is deliberate: ``svdsub.f:14`` uses
    ``small = 1.e-6`` on the raw singular values, and the stoichiometry matrix
    entries are small integers, so the scale is fixed and the choice is safe.
    """
    _, sv, Vt = np.linalg.svd(s_active, full_matrices=True)
    k = s_active.shape[1]
    rank = int(np.sum(sv >= SMALL_SVD))
    return np.ascontiguousarray(Vt[rank:].T)     # (k, k - rank)


def _pinv_solve(A: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Batched SVD pseudo-inverse solve reproducing ``svdsub.f``.

    ``A`` is (B, k, k), ``rhs`` is (B, k, r).  Singular values below
    ``SMALL_SVD`` are zeroed (svdsub.f:41-51), giving the minimum-norm least
    squares solution.  Because the basis passed in is orthonormal, the lifted
    result ``Q2 @ x`` is invariant to the choice of null-space basis even when
    ``A`` is rank deficient.
    """
    U, sv, Vt = np.linalg.svd(A)
    inv = np.where(sv >= SMALL_SVD, 1.0 / np.where(sv >= SMALL_SVD, sv, 1.0), 0.0)
    y = np.einsum('bki,bkr->bir', U, rhs)
    y = y * inv[:, :, None]
    return np.einsum('bik,bir->bkr', Vt, y)


def _group_by_active(active: np.ndarray) -> Dict[bytes, np.ndarray]:
    """Group batch rows by identical active-species pattern."""
    packed = np.packbits(active, axis=1)
    groups: Dict[bytes, List[int]] = {}
    for b in range(active.shape[0]):
        groups.setdefault(packed[b].tobytes(), []).append(b)
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


# ════════════════════════════════════════════════════════════════════════════
# dn/dT and dn/dP
# ════════════════════════════════════════════════════════════════════════════

def molar_derivatives(n: np.ndarray,
                      T: np.ndarray,
                      P: np.ndarray,
                      S_spec: np.ndarray,
                      V_spec: np.ndarray,
                      tables: MetamorphicTables,
                      *,
                      active: np.ndarray | None = None,
                      restrict_to: List[List[int]] | None = None,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for ``dn/dT|_P`` and ``dn/dP|_T`` at fixed bulk composition.

    Parameters
    ----------
    n       : (B, S) species moles
    T       : (B,)   K
    P       : (B,)   GPa
    S_spec  : (B, S) partial molar entropy ``sspeca`` (J/mol/K)
    V_spec  : (B, S) partial molar volume  ``vspeca`` (cm^3/mol)
    active  : (B, S) bool, defaults to ``n > 0``.  Inactive species are pinned
              to zero exactly as ``sform.f:20-26`` does with its extra rows.
    restrict_to : optional per-phase list of *global* species indices.  When
              given, the null space is additionally restricted to exchange
              directions *within* each listed block -- this reproduces the
              ``dndtfast``/``dndpfast`` calculation of physub.f:242-288, which
              uses only the order-disorder species of each phase.

    Returns
    -------
    dndt : (B, S) mol/K
    dndp : (B, S) mol/GPa
    """
    B, S = n.shape
    if active is None:
        active = n > 0.0

    # physub.f:161-163.  Signs are pre-flipped in the Fortran so that the same
    # linear solve yields +dn/dT and +dn/dP (triple-product rule).
    dmdt = S_spec / 1000.0        # kJ/K/mol
    dmdp = -V_spec                # kJ/GPa/mol

    H_blocks = hessian_blocks(n, T, P, tables)

    dndt = np.zeros((B, S), dtype=np.float64)
    dndp = np.zeros((B, S), dtype=np.float64)

    for _, rows in _group_by_active(active).items():
        act = active[rows[0]]
        act_idx = np.where(act)[0]
        if act_idx.size == 0:
            continue

        Q2 = _build_q2(tables, act_idx, restrict_to)
        if Q2 is None or Q2.shape[1] == 0:
            continue
        nnull = Q2.shape[1]
        nb = rows.size

        # H^P = Q2^T H Q2, accumulated over phase blocks (H is block diagonal)
        Hp = np.zeros((nb, nnull, nnull), dtype=np.float64)
        for iph, members in enumerate(tables.phase_members):
            if len(members) == 0:
                continue
            Qb = Q2[np.asarray(members, dtype=np.int64), :]        # (m, nnull)
            if not Qb.any():
                continue
            Hb = H_blocks[iph][rows]                               # (nb, m, m)
            Hp += np.einsum('mk,bmn,nl->bkl', Qb, Hb, Qb)

        rhs = np.stack([Q2.T @ dmdt[rows].T, Q2.T @ dmdp[rows].T], axis=-1)
        rhs = np.transpose(rhs, (1, 0, 2))                         # (nb, nnull, 2)

        x = _pinv_solve(Hp, rhs)                                   # (nb, nnull, 2)

        dndt[rows] = (Q2 @ x[:, :, 0].T).T
        dndp[rows] = (Q2 @ x[:, :, 1].T).T

    return dndt, dndp


def _build_q2(tables: MetamorphicTables,
              act_idx: np.ndarray,
              restrict_to: List[List[int]] | None) -> np.ndarray | None:
    """Orthonormal (S, nnull) basis of bulk-composition-conserving directions."""
    S = tables.nspec

    if restrict_to is None:
        Nb = _nullspace(tables.s_mat[:, act_idx])                  # (k, nnull)
        if Nb.shape[1] == 0:
            return None
        Q2 = np.zeros((S, Nb.shape[1]), dtype=np.float64)
        Q2[act_idx, :] = Nb
        return Q2

    # "Fast" path: only exchange within each order-disorder block.  physub.f
    # builds a single +-1/sqrt(2) direction per block; we take the full null
    # space of the block's stoichiometry, which coincides with that for a pair
    # and generalises correctly to longer blocks.
    cols: List[np.ndarray] = []
    act_set = set(act_idx.tolist())
    for block in restrict_to:
        sub = [g for g in block if g in act_set]
        if len(sub) < 2:
            continue
        Nb = _nullspace(tables.s_mat[:, sub])
        for c in range(Nb.shape[1]):
            col = np.zeros(S, dtype=np.float64)
            col[np.asarray(sub, dtype=np.int64)] = Nb[:, c]
            cols.append(col)
    if not cols:
        return None
    return np.stack(cols, axis=1)


# ════════════════════════════════════════════════════════════════════════════
# Aggregate totals
# ════════════════════════════════════════════════════════════════════════════

def metamorphic_terms(dndt: np.ndarray,
                      dndp: np.ndarray,
                      S_spec: np.ndarray,
                      V_spec: np.ndarray,
                      T: np.ndarray,
                      volagg: np.ndarray) -> Dict[str, np.ndarray]:
    """physub.f:590-594.  Returns ``alpmet`` (1/K), ``cpmet`` (J/mol/K... see below),
    and ``bmet`` (cm^3/mol/GPa, a compliance).

    ``cpmet`` is extensive here (J/K); divide by ``wmagg`` to get J/g/K, exactly
    as physub.f:598 does.
    """
    safe_vol = np.maximum(volagg, TINY)
    alpmet = (dndt * V_spec).sum(axis=1) / safe_vol
    cpmet = T * (dndt * S_spec).sum(axis=1)
    bmet = (dndp * (-V_spec)).sum(axis=1)          # dmdp = -V  (physub.f:163)
    return dict(alpmet=alpmet, cpmet=cpmet, bmet=bmet)


def combine_totals(*,
                   alpagg: np.ndarray,
                   cpagg: np.ndarray,
                   btaggr: np.ndarray,
                   volagg: np.ndarray,
                   wmagg: np.ndarray,
                   T: np.ndarray,
                   alpmet: np.ndarray | None = None,
                   cpmet: np.ndarray | None = None,
                   bmet: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """physub.f:595-608.  Isomorphic totals when the met terms are omitted.

    Returns ``alp``, ``cp``, ``cv``, ``gam``, ``KT``, ``KS`` -- matching
    ``alptot/cptot/cvtot/gamtot/btot/bstot`` (or the ``*iso`` variants).
    """
    zeros = np.zeros_like(alpagg)
    alpmet = zeros if alpmet is None else alpmet
    cpmet = zeros if cpmet is None else cpmet
    bmet = zeros if bmet is None else bmet

    safe_wm = np.maximum(wmagg, TINY)
    safe_bt = np.where(np.abs(btaggr) > TINY, btaggr, TINY)

    compliance = volagg / safe_bt + bmet
    KT = np.where(np.abs(compliance) > TINY, volagg / compliance, 0.0)

    alp = alpagg + alpmet
    cp = cpagg + cpmet / safe_wm
    cv = cp - T * volagg * alp**2 * KT / safe_wm * 1000.0
    safe_cv = np.where(np.abs(cv) > 1.0e-10, cv, 1.0e-10)
    gam = volagg * alp * KT / (safe_cv * safe_wm) * 1000.0
    KS = KT * (1.0 + alp * gam * T)
    KS = np.where(T > 0.0, KS, KT)                 # physub.f:609

    return dict(alp=alp, cp=cp, cv=cv, gam=gam, KT=KT, KS=KS)


# Default active-set smallness threshold, as a fraction of total system moles. 1E-6
# See prune_active_set() for why this exists and how to choose it.
NSMALL_REL = 1E-3#4E-4 #1E-3


def compute_within_phase_frac(n: np.ndarray, phase_members: Sequence[Sequence[int]]) -> np.ndarray:
    """Each species' mole fraction *within its own phase* (0 where that phase
    is entirely absent).

    This is the quantity HeFESTo's own ideal-configurational-entropy penalty
    actually scales with (``~RT/x``, x = within-phase mole fraction) -- not
    absolute abundance. A species sitting at within_phase_frac ~= 1 is
    receiving essentially no regularisation regardless of how small the whole
    phase is, which is the root cause prune_active_set's single-component
    sweep (below) targets. See its docstring for the full mechanism.

    Works in whatever column space ``phase_members`` groups: pass full
    (B, nspec) species moles with HeFESToParams.phase_members for the
    ground-truth/internal-EOS path, or (B, C) ML-component moles with
    ml_indexer.label_indices.values() for anything upstream of PropertyIDX.
    """
    n = np.maximum(np.asarray(n, dtype=np.float64), 0.0)
    frac = np.zeros_like(n)
    for members in phase_members:
        if len(members) == 0:
            continue
        idx = np.asarray(members, dtype=np.int64)
        tot = n[:, idx].sum(axis=1, keepdims=True)
        frac[:, idx] = np.where(tot > 0, n[:, idx] / np.maximum(tot, 1e-300), 0.0)
    return frac


def prune_active_set(n: np.ndarray,
                     tables: MetamorphicTables,
                     nsmall_rel: float = NSMALL_REL,
                     within_phase_frac: Optional[np.ndarray] = None,
                     single_component_dominance: float = 0.98,
                     single_component_nsmall_rel: Optional[float] = None) -> np.ndarray:
    """Zero out species that are present only at trace level.

    Why this is necessary
    ---------------------
    HeFESTo's regularisation of absent species (physub.f:160 sets ``n = 1e-16``)
    works through the *ideal configurational* term of the Hessian, which
    penalises a species by ``~RT/x`` where ``x`` is its mole fraction **within
    its phase** -- not its absolute abundance.  That is fine inside HeFESTo,
    whose minimiser never puts a phase in the active set at trace level.

    It breaks when the composition comes from a neural network, which does not
    emit exact zeros.  A species that is the sole occupant of an otherwise
    absent phase has mole fraction ~1, so it gets **no penalty at all**, however
    tiny its absolute amount.  Its Hessian diagonal is O(1) and the dn/dT solve
    treats it as a full-blown reactant that can absorb arbitrary entropy.

    Observed on JiChingSims/model_000298: leaking anorthite ('an', the partner
    of the only active plagioclase species) in at 1e-8 mol gives

        plg Hessian block = [[3.3e+00, -3.3e+08],
                             [-3.3e+08,  3.3e+16]]

    -- the 1e16 penalty lands on 'ab' (mole fraction ~0), while 'an' sits at
    3.3, i.e. unregularised.  The solve then returns ``dn_an/dT = 2.7e-2``
    mol/K for a species holding 1e-8 mol: it would deplete in 4e-7 K.  That one
    number is the largest ``dndt`` in the system and it propagates straight into
    ``cpmet = sum_i T dndt_i S_i``, inflating Cp by ~50%.

    Because the offending species carries no mass, no volume and no entropy, it
    is invisible in rho / Vp / Vs / phase proportions -- which is exactly the
    reported symptom: "phase proportions agree but Cp has a spike that isn't in
    the ground truth".  A lingering trace tail on a species that should have
    phased out also drags the latent-heat peak to higher pressure ("the
    transition looks late") and inflates it.

    Choosing the threshold
    ----------------------
    ``nsmall_rel`` is relative to total system moles, so it is composition- and
    scale-independent.  It must sit **above the noise floor of whatever produced
    n** and below any abundance you care about.  Emulator output with an
    absolute noise floor of ~1e-6 mol needs ``nsmall_rel`` around 1e-5; ground
    truth read from fort.99 (exact zeros) is insensitive to anything in
    1e-12..1e-4.  A threshold *below* the noise floor is worse than useless --
    it leaves the trace species active and pays the cost of the check.

    Both a per-species and a per-phase test are applied: a phase whose *total*
    is trace level is dropped entirely, which catches the case where the trace
    amount is split across several members so that no single member's mole
    fraction is small.

    The single-component sweep
    ---------------------------
    The two tests above only look at a phase's share of the *whole system*.
    That misses a distinct case that turned out to be at least as common in
    practice: a phase that mostly reacts out and leaves ONE surviving member
    behind (e.g. a spinel-group phase where sp/hc/picr all go to exactly zero
    but a magnetite-spinel endmember lingers at 0.1-0.2% of the system for
    several GPa/simulation steps -- see the nGibbs Simulation754 Cp
    investigation). That survivor is precisely the "sole occupant of an
    otherwise absent phase" pathology described above: with no phase-mates to
    be diluted against, its within-phase mole fraction sits near 1 and it gets
    essentially zero configurational regularisation, however small the phase's
    total is -- and its total can easily sit ABOVE the ordinary nsmall_rel
    floor (which was tuned for numerical noise, not for real, slowly-decaying
    residual phases), so the two sweeps above don't catch it.

    ``within_phase_frac`` (see :func:`compute_within_phase_frac`) makes this
    testable directly instead of inferring it from smallness alone: a species
    with within_phase_frac >= single_component_dominance IS that lone
    survivor, regardless of how large or small its phase happens to be. Such a
    species is only pruned if its phase is ALSO small relative to
    ``single_component_nsmall_rel`` -- a deliberately looser bar than
    ``nsmall_rel`` (default: 5x), since a lone-survivor phase is dangerous at
    abundances an ordinarily-mixed phase would be fine at.

    Parameters
    ----------
    n : (B, S) array
        Species moles.  Negative values are clipped to zero first.
    nsmall_rel : float
        Threshold as a fraction of each row's total moles.  Set to 0 to disable
        (reproducing the raw behaviour, e.g. to reproduce an old result).
    within_phase_frac : (B, S) array, optional
        Each species' mole fraction within its own phase (see
        :func:`compute_within_phase_frac`). When given, enables the
        single-component sweep described above. When omitted, only the plain
        total-system-fraction tests run (previous behaviour).
    single_component_dominance : float
        A species counts as a lone phase-survivor when its within_phase_frac
        is at least this (default 0.98).
    single_component_nsmall_rel : float, optional
        Smallness threshold (fraction of total system moles) applied only to
        phases flagged by single_component_dominance. Defaults to
        ``5 * nsmall_rel`` when not given.

    Returns
    -------
    (B, S) array
        A copy of ``n`` with trace species set to exactly zero.
    """
    n = np.maximum(np.asarray(n, dtype=np.float64), 0.0)
    if not nsmall_rel:
        return n

    total = n.sum(axis=1, keepdims=True)
    thresh = nsmall_rel * total                       # (B, 1)

    n = np.where(n < thresh, 0.0, n)

    # Phase-level sweep: a phase that is collectively trace level goes entirely,
    # even if its members individually clear the per-species bar.
    for members in tables.phase_members:
        if len(members) == 0:
            continue
        idx = np.asarray(members, dtype=np.int64)
        ph_total = n[:, idx].sum(axis=1, keepdims=True)
        n[:, idx] = np.where(ph_total < thresh, 0.0, n[:, idx])

    # Single-component sweep: a phase's lone surviving member gets no
    # configurational dilution regardless of absolute size, so hold it to a
    # looser (but still finite) smallness bar than an ordinarily-mixed phase.
    if within_phase_frac is not None:
        single_thresh = (
            single_component_nsmall_rel if single_component_nsmall_rel is not None
            else 5.0 * nsmall_rel
        ) * total
        wpf = np.asarray(within_phase_frac, dtype=np.float64)
        for members in tables.phase_members:
            if len(members) == 0:
                continue
            idx = np.asarray(members, dtype=np.int64)
            ph_total = n[:, idx].sum(axis=1, keepdims=True)
            lone_survivor = (wpf[:, idx] >= single_component_dominance).any(axis=1, keepdims=True)
            drop = lone_survivor & (ph_total > 0) & (ph_total < single_thresh)
            n[:, idx] = np.where(drop, 0.0, n[:, idx])

    return n


def trace_species_leverage(dndt: np.ndarray,
                           n: np.ndarray,
                           S_spec: np.ndarray,
                           T: np.ndarray,
                           nsmall_rel: float = NSMALL_REL) -> Dict[str, np.ndarray]:
    """Diagnose whether trace species are driving the latent-heat term.

    Use this to *choose* ``nsmall_rel`` from evidence instead of guessing.  The
    pathology described in :func:`prune_active_set` has a clean signature: a
    species holding almost nothing carries a large share of
    ``cpmet = sum_i T dndt_i S_i``.  A physically sensible species changes by a
    small fraction of its own abundance per kelvin; a pathological one would
    deplete itself in microkelvins.

    Returns
    -------
    dict
        ``leverage``   (B,)  fraction of |cpmet| carried by species below the
                             threshold.  Should be ~0.  Anything above ~0.05 in
                             a row means ``nsmall_rel`` is too low for this
                             input -- raise it above the emulator's noise floor.
        ``worst_frac`` (B,)  largest single-species share of |cpmet|.
        ``worst_spec`` (B,)  index of that species.
        ``depletion_K`` (B,) smallest |n_i / dndt_i| over all present species,
                             in K: how long the fastest-changing species would
                             survive at its current rate.  This one is
                             **threshold-free**, so it is the metric to watch:
                             ground truth gives tens of K, a pathological trace
                             species gives 1e-3 K or less.
    """
    contrib = T[:, None] * dndt * S_spec                 # (B, S), sums to cpmet
    tot = np.abs(contrib).sum(axis=1)
    tot = np.where(tot > 0, tot, 1.0)

    total_n = n.sum(axis=1, keepdims=True)
    trace = n < (nsmall_rel * total_n)

    leverage = np.abs(np.where(trace, contrib, 0.0)).sum(axis=1) / tot
    share = np.abs(contrib) / tot[:, None]
    worst_spec = share.argmax(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        life = np.where((n > 0) & (np.abs(dndt) > 0),
                        np.abs(n / np.where(dndt != 0, dndt, 1.0)), np.inf)
    return dict(
        leverage=leverage,
        worst_frac=share.max(axis=1),
        worst_spec=worst_spec,
        depletion_K=life.min(axis=1),
    )


def add_metamorphic(result: Dict[str, np.ndarray],
                    n: np.ndarray,
                    T: np.ndarray,
                    P: np.ndarray,
                    tables: MetamorphicTables,
                    *,
                    include_fast: bool = False,
                    nsmall_rel: float = NSMALL_REL,
                    within_phase_frac: Optional[np.ndarray] = None,
                    single_component_dominance: float = 0.98,
                    single_component_nsmall_rel: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Augment a ``compute()`` result with metamorphic (phase-change) totals.

    Parameters
    ----------
    result : dict
        Output of ``hefesto_vec.compute()``.  Must carry ``_S``, ``V``,
        ``KTr``, ``alpagg``, ``cpagg``, ``volagg``, ``wmagg``.
    n : (B, S)
        Species moles -- the same ``X`` that was passed to ``compute()``.
    nsmall_rel : float
        Active-set smallness threshold, as a fraction of total moles.  Trace
        species are excluded before the dn/dT solve — see
        :func:`prune_active_set`, which explains why this is load-bearing when
        ``n`` comes from an emulator rather than from HeFESTo's own minimiser.
        Pass 0 to disable.
    within_phase_frac, single_component_dominance, single_component_nsmall_rel
        Forwarded to :func:`prune_active_set`'s single-component sweep -- see
        there for what they do. ``within_phase_frac`` omitted disables that
        sweep (previous behaviour).
    include_fast : bool
        Also compute the intra-phase order-disorder ("fast") terms that
        physub.f:414-431 folds into the *per-phase* adiabatic modulus, and
        hence into Vp/Vb.  See the note on velocities below.

    Returns
    -------
    dict with (all shape (B,))
        ``alpiso, cpiso, cviso, gamiso, KTiso, KSiso``  -- fort.59
        ``alptot, cptot, cvtot, gamtot, KTtot, KStot``  -- fort.56 / fort.69
        ``alpmet, cpmet, bmet``                         -- fort.69
        ``dndt, dndp``  (B, S)                          -- fort.42
        plus ``deltaent``, ``deltavol``, ``ClapeyronSlope``.

    Note on velocities
    ------------------
    fort.56's ``VP``/``VB`` are *not* purely isomorphic: physub.f:414-431
    applies a per-phase ``bmet`` built only from the order-disorder species
    (cation ordering, Fe spin-state pairs), on the argument that intra-phase
    ordering relaxes at seismic frequency while cross-phase reactions do not.
    ``Vs`` is unaffected -- the shear modulus carries no metamorphic term.
    Set ``include_fast=True`` to get ``bmet_fast`` per phase for that path.
    """
    n = prune_active_set(
        n, tables, nsmall_rel=nsmall_rel,
        within_phase_frac=within_phase_frac,
        single_component_dominance=single_component_dominance,
        single_component_nsmall_rel=single_component_nsmall_rel,
    )
    active = n > 0.0

    S_spec_vib = result['_S']          # (B, S) J/mol/K, no mixing term
    V_vib = result['V']                # (B, S) cm^3/mol, no excess term

    smix, volsum = mixing_terms(n, P, tables, active=active)
    S_spec = S_spec_vib + smix         # sspeca  (func.f:92; smag omitted, = 0)
    V_spec = V_vib + volsum            # vspeca  (func.f:93)

    dndt, dndp = molar_derivatives(n, T, P, S_spec, V_spec, tables,
                                   active=active)

    volagg = result['volagg']
    wmagg = result['wmagg']
    met = metamorphic_terms(dndt, dndp, S_spec, V_spec, T, volagg)

    iso = combine_totals(alpagg=result['alpagg'], cpagg=result['cpagg'],
                         btaggr=result['KTr'], volagg=volagg, wmagg=wmagg, T=T)
    tot = combine_totals(alpagg=result['alpagg'], cpagg=result['cpagg'],
                         btaggr=result['KTr'], volagg=volagg, wmagg=wmagg, T=T,
                         alpmet=met['alpmet'], cpmet=met['cpmet'],
                         bmet=met['bmet'])

    out: Dict[str, np.ndarray] = dict(
        dndt=dndt, dndp=dndp,
        sspeca=S_spec, vspeca=V_spec,
        alpmet=met['alpmet'], cpmet=met['cpmet'], bmet=met['bmet'],
        alpiso=iso['alp'], cpiso=iso['cp'], cviso=iso['cv'],
        gamiso=iso['gam'], KTiso=iso['KT'], KSiso=iso['KS'],
        alptot=tot['alp'], cptot=tot['cp'], cvtot=tot['cv'],
        gamtot=tot['gam'], KTtot=tot['KT'], KStot=tot['KS'],
    )
    out.update(clapeyron_terms(dndp, S_spec, V_spec))

    if include_fast:
        dndt_f, dndp_f = molar_derivatives(
            n, T, P, S_spec, V_spec, tables, active=active,
            restrict_to=[b for b in tables.od_members if len(b) >= 2],
        )
        out['dndt_fast'] = dndt_f
        out['dndp_fast'] = dndp_f

        # SEISMIC-FREQUENCY path, per phase (physub.f:414-431).  Same Hessian
        # machinery as the aggregate terms, but restricted to the order-disorder
        # species of each phase.  All THREE accumulators are needed, and they
        # are the per-phase analogues of the aggregate ones at physub.f:590-594:
        #
        #   bmet_ph   = sum_i dndpfast_i * dmdp_i  = -sum_i dndpfast_i V_i
        #   alpmet_ph = sum_i dndtfast_i V_i / volph
        #   cpmet_ph  = sum_i T dndtfast_i S_i / wmph      (specific, J/g/K)
        #
        # bmet alone softens K_T too much: it accounts for the whole 0.43%
        # high-pressure Vb bias and then overshoots to -0.17%.  alpmet_ph feeds
        # gamma and pushes Ks back up; with both, plus cpmet_ph, the residual
        # Vp/Vb bias becomes flat in P and equal to the isomorphic rho/Vs
        # baseline -- i.e. no velocity-specific error is left.
        volph_by_phase, wmph_by_phase = {}, {}
        for c in result.get('phase_cache') or []:
            volph_by_phase[c['iph']] = c['volph']
            wmph_by_phase[c['iph']] = c['wmph']

        zero = np.zeros_like(T)
        bmet_f: List[np.ndarray] = []
        alpmet_f: List[np.ndarray] = []
        cpmet_f: List[np.ndarray] = []
        for iph, members in enumerate(tables.phase_members):
            if len(members) == 0:
                bmet_f.append(zero); alpmet_f.append(zero); cpmet_f.append(zero)
                continue
            idx = np.asarray(members, dtype=np.int64)
            volph = volph_by_phase.get(iph)
            wmph = wmph_by_phase.get(iph)
            bmet_f.append((dndp_f[:, idx] * (-V_spec[:, idx])).sum(axis=1))
            if volph is None:
                alpmet_f.append(zero); cpmet_f.append(zero)
                continue
            safe_v = np.where(volph > 1.0e-30, volph, 1.0e-30)
            safe_w = np.where(wmph > 1.0e-30, wmph, 1.0e-30)
            alpmet_f.append((dndt_f[:, idx] * V_spec[:, idx]).sum(axis=1) / safe_v)
            cpmet_f.append(
                (T[:, None] * dndt_f[:, idx] * S_spec[:, idx]).sum(axis=1) / safe_w
            )

        out['bmet_fast_phases'] = bmet_f
        out['alpmet_fast_phases'] = alpmet_f
        out['cpmet_fast_phases'] = cpmet_f

        # Apply it: overwrite the isomorphic velocities with the softened ones.
        # Vs is recomputed too but is unchanged by construction (no metamorphic
        # term in the shear modulus) -- a useful free regression check.
        if result.get('phase_cache'):
            from .aggregate import apply_fast_metamorphic
            out['Vp_iso'] = result['Vp']
            out['Vb_iso'] = result['Vb']
            out['Kh_iso'] = result['Kh']
            out.update(apply_fast_metamorphic(
                result, T,
                bmet_fast_phases=bmet_f,
                alpmet_fast_phases=alpmet_f,
                cpmet_fast_phases=cpmet_f,
            ))

    return out


def clapeyron_terms(dndp: np.ndarray,
                    S_spec: np.ndarray,
                    V_spec: np.ndarray) -> Dict[str, np.ndarray]:
    """physub.f:610-627.  Free by-products of ``dndp``.

    ``deltaent`` (J/K), ``deltavol`` (cm^3/GPa), and the Clapeyron slope
    ``deltaent/deltavol``.  Not needed for rho/Vp/Vs; provided for diagnostics.
    """
    deltaent = (dndp * S_spec).sum(axis=1)
    deltavol = (dndp * V_spec).sum(axis=1)
    slope = np.where(np.abs(deltavol) > 1.0e-15, deltaent / np.where(
        np.abs(deltavol) > 1.0e-15, deltavol, 1.0), 0.0)
    return dict(deltaent=deltaent, deltavol=deltavol, ClapeyronSlope=slope)
