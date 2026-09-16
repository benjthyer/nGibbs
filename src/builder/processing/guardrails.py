"""
Guardrails for operations that are unsafe while thermodynamic labels are attached.

Two separate hazards, both silent.

Abundance resampling breaks the isentropic premise
--------------------------------------------------
`retrieve_component_moles` draws `phase_multipliers` **inside** its `for phase in
phases` loop, so every phase in every row gets an independent random factor. That does
not merely rescale a system -- it changes the relative proportions of the assemblage,
and therefore the bulk composition and the specific entropy, while the stored
`S(J/g/K)` column keeps the value HeFESTo computed for the *un*resampled assemblage.
An isentropic emulator trained on that is being taught a wrong S for its own
assemblage.

A uniform factor applied to every phase at once would be harmless -- specific entropy
is intensive -- but that is not what the code does.

Lifting this requires recomputing bulk entropy from the weighted phase-wise entropies
(collected for MELTS, not yet wired).

The hazard above is specifically an entropy-column hazard, not a resampling hazard:
resampling never touches Pressure or Temperature, and specific entropy is the only
attached label that abundance-weighted perturbation silently invalidates. So
`check_resampling_config` below runs once, at the top of `process_for_ML`, before any
table is opened: it raises `ResamplingNotSupported` only when an entropy column
('S(System_main)', 'S(System_main) / mass(System_main)', 'S(J/g/K)(System_main)') is a
configured feature or free output, and otherwise just prints a loud warning and lets
the configured resampling proceed. The deep call sites inside `BigMetaTable`/
`MLexporter` that used to hard-block any non-identity multiplier no longer do -- this
upfront check is the single place that decision gets made, before any data is read.

Derivative sidecars and column-moving operations
------------------------------------------------
`separate_analcime` and `separate_k_feldspar` move columns. Any attached dn/dP or
dn/dT table would have to follow the same moves or it silently misaligns against its
own species. Both are MELTS-only, and there is no dn/dP or dn/dT source for MELTS yet,
so the guard refuses rather than pretending to handle it.
"""
from __future__ import annotations

from typing import Any, Sequence

IDENTITY = (1, 1)


class ResamplingNotSupported(RuntimeError):
    """Raised when abundance resampling is requested while entropy labels are live."""


class DerivativeSidecarConflict(RuntimeError):
    """Raised when a column-moving operation would desynchronise a derivative table."""


def _is_identity(bounds: Sequence[float]) -> bool:
    try:
        lo, hi = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError, IndexError):
        return False
    return lo == 1.0 and hi == 1.0


def assert_identity_multipliers(bounds: Any, caller: str) -> None:
    """Reject anything but [1, 1].

    `bounds` may be a single pair or a list of pairs (`resample_bounds`).
    """
    if bounds is None:
        return
    pairs = bounds
    if len(bounds) == 2 and not hasattr(bounds[0], '__len__'):
        pairs = [bounds]
    bad = [p for p in pairs if not _is_identity(p)]
    if not bad:
        return
    raise ResamplingNotSupported(
        f"{caller}: abundance multipliers must be [1, 1]; got {bad}.\n"
        "Independent per-phase multipliers change the assemblage's relative\n"
        "proportions, hence its bulk composition and specific entropy, while the\n"
        "stored S(J/g/K) column still holds the value computed for the unresampled\n"
        "assemblage. Training an isentropic emulator on that teaches it a wrong S.\n"
        "Prefer generating a genuinely larger dataset.\n"
        "To lift this, recompute bulk entropy as the mass-weighted sum of phase-wise\n"
        "entropies after resampling, then relax the guard here."
    )


ENTROPY_LABELS = frozenset({
    'S(System_main)',
    'S(System_main) / mass(System_main)',
    'S(J/g/K)(System_main)',
})


def entropy_labels_present(feature_names: Any, free_outputs: Any) -> bool:
    """True if an entropy column is a configured feature or free output."""
    names = set(feature_names or []) | set(free_outputs or [])
    return bool(names & ENTROPY_LABELS)


def check_resampling_config(upsample_enabled: bool, upsample_cfg: dict, resampling_cfg: dict,
                             feature_names: Any, free_outputs: Any) -> None:
    """Validate abundance-resampling config before any table is opened.

    Mirrors exactly which bounds `process_for_ML` will actually use:
    `upsampling.phases`/`test_set_phases` and `resampling.train_bounds` only take
    effect when `upsample_enabled`; `resampling.test_bounds` is always applied.
    Non-identity bounds among those are only a hazard while an entropy column (see
    `ENTROPY_LABELS`) is a configured feature or free output -- resampling doesn't
    touch Pressure or Temperature either way. So: raise if entropy is live, else warn
    loudly and return so the caller can proceed.
    """
    bounds_by_source = []
    if upsample_enabled:
        for section in ('phases', 'test_set_phases'):
            for phase, phase_cfg in ((upsample_cfg or {}).get(section) or {}).items():
                bounds_by_source.append((
                    f"upsampling.{section}.{phase}.multiplier_bounds",
                    (phase_cfg or {}).get('multiplier_bounds'),
                ))
        for bounds in (resampling_cfg or {}).get('train_bounds') or []:
            bounds_by_source.append(('resampling.train_bounds', bounds))
    for bounds in (resampling_cfg or {}).get('test_bounds') or []:
        bounds_by_source.append(('resampling.test_bounds', bounds))

    non_identity = [(name, b) for name, b in bounds_by_source if not _is_identity(b)]
    if not non_identity:
        return

    listing = "\n".join(f"  - {name}: {b}" for name, b in non_identity)

    if entropy_labels_present(feature_names, free_outputs):
        raise ResamplingNotSupported(
            "Abundance resampling is configured with non-identity multiplier bounds "
            "while an entropy column is a configured feature or free output:\n"
            f"{listing}\n"
            "Independent per-phase multipliers change the assemblage's relative\n"
            "proportions, hence its bulk composition and specific entropy, while the\n"
            "stored S(J/g/K) column still holds the value computed for the unresampled\n"
            "assemblage. Training an isentropic emulator on that teaches it a wrong S.\n"
            "Remove the entropy column from featureNames/free_outputs, or pin the\n"
            "bounds above back to [1, 1], before running this recipe.\n"
            "To lift this properly, recompute bulk entropy as the mass-weighted sum of\n"
            "phase-wise entropies after resampling, then relax this guard in\n"
            "builder/processing/guardrails.py."
        )

    print("!" * 78)
    print("[GUARDRAIL WARNING] Abundance resampling is configured with non-identity "
          "multiplier bounds:")
    print(listing)
    print("Independent per-phase multipliers change the relative proportions -- and\n"
          "therefore the bulk composition -- of every resampled row. No entropy column\n"
          f"({', '.join(sorted(ENTROPY_LABELS))}) is currently a\n"
          "configured feature or free output, so nothing downstream is taught a stale S;\n"
          "Pressure and Temperature are untouched by resampling either way. If an\n"
          "entropy column is added to featureNames/free_outputs later, this check will\n"
          "raise instead -- re-verify these bounds before overriding it.")
    print("!" * 78)


def assert_alias_safe(bounds: Any, model: str, caller: str) -> None:
    """Aliasing `table1` onto `table` is only safe with identity multipliers.

    `resampling_to_datasets` normally copies the whole table so the resampling loop can
    mutate mass columns without touching the original -- a real cost at 90 GB. With
    multipliers pinned to [1, 1] that loop's only remaining effect is renormalising mass
    columns to 100, and for HeFESTo nothing downstream reads them: `masslabels` are
    built from `phaseMass` in the HeFESTo branch, and `retrieve_component_moles` takes
    component moles from their own columns. So the copy buys nothing and the alias is
    exact.

    MELTS is different -- its `masslabels` read `table1[:, mass_indices]` directly and
    its molar path divides by phase mass -- so aliasing there would change results and
    is refused.
    """
    assert_identity_multipliers(bounds, caller)
    if str(model).upper() != 'HEFESTO':
        raise ResamplingNotSupported(
            f"{caller}: table1 aliasing is HeFESTo-only; model is {model!r}.\n"
            "MELTS reads mass columns out of table1 for both masslabels and the molar\n"
            "conversion, so aliasing would mutate the source table and change results."
        )


def assert_no_derivative_sidecars(table: Any, caller: str) -> None:
    """Refuse column-moving operations while dn/dP or dn/dT tables are attached."""
    attached = [n for n in ('dndp', 'dndt')
                if getattr(table, n, None) is not None]
    if not attached:
        return
    raise DerivativeSidecarConflict(
        f"{caller}: {', '.join(attached)} sidecar(s) attached.\n"
        "This operation moves columns, and the derivative tables would have to follow\n"
        "the same moves or they silently misalign against their own species. Both\n"
        "callers are MELTS-only and there is no dn/dP or dn/dT source for MELTS yet,\n"
        "so this refuses rather than pretending to handle it.\n"
        "Run column-moving operations before attaching derivatives, or drop them first."
    )
