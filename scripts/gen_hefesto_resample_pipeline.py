"""
Generate (print, do not run) a bash script that drives the full HeFESTo
double-resample + merge + bundle pipeline.

The pipeline the emitted script runs, in order:

  stage 0  import the original (already-run) workspace with composition
           derivatives, and export a phase-boundary CSV
             import_hefesto_subdirs_derivs.py --phase-change-dataname ...
  stage 1  build a phase-change resample tree from that CSV
             prepare_hefesto_tree_from_phase_changes.py
           -> run HeFESTo on it (local GNU parallel, or SLURM + wait barrier)
           -> re-import it, this time exporting *deep* phase-boundary bounds
             import_hefesto_subdirs_derivs.py --deep-phase-change-dataname ...
  stage 2  build a fine deep-resample tree from the deep bounds
             prepare_hefesto_tree_fine.py --deep
           -> run HeFESTo on it
           -> re-import it (no phase-boundary export this time)
  stage 3  merge the three imported BigMetaTables (stage 0 / 1 / 2 main
           tables only -- the phase-boundary CSVs used to seed the resample
           trees are deliberately NOT merged)
             merge_bigmetatables.py
  stage 4  tar + gzip the merged table, its derivative sidecars
           (dn/dP, dn/dT) and the per-stage derivative manifests into one
           <name>_merged_bundle.tar.gz

Derived directories follow the repo's existing convention:
  <output-root>/<name>_resample1   (phase-change tree)
  <output-root>/<name>_resample2   (fine deep tree)

Usage:
  python scripts/gen_hefesto_resample_pipeline.py \
      --workspace /path/to/OriginalWorkspace \
      --name EarthAdiabats \
      --output-root /scratch/hefesto \
      --control-dir src/builder/HeFESTo/batch/shallowHeFESTo \
      > run_earthadiabats_pipeline.sh

  # cluster execution instead of local GNU parallel:
  python scripts/gen_hefesto_resample_pipeline.py ... --runner cluster \
      --sbatch-time-limit 20 --squeue-filter hefesto_ > pipeline.sh
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_DIR = re.compile(r"^Simulation\d+$", re.IGNORECASE)


def count_simulation_dirs(root: Path) -> int:
    """Number of SimulationN directories anywhere under root."""
    n = 0
    for dirpath, _dirnames, _files in os.walk(root):
        if _SIM_DIR.match(os.path.basename(dirpath)):
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("inputs")
    g.add_argument("--workspace", required=True, type=Path,
                   help="Original, already-run HeFESTo workspace root (passed as "
                        "--root to the importer).")
    g.add_argument("--name", default=None,
                   help="Short base name for all outputs and derived directory "
                        "suffixes. Default: the workspace directory's own name.")
    g.add_argument("--output-root", required=True, type=Path,
                   help="Directory under which <name>_resample1 / _resample2 "
                        "trees are created.")
    g.add_argument("--control-dir", required=True, type=Path,
                   help="Control-template directory passed to both prepare "
                        "scripts (e.g. .../batch/shallowHeFESTo).")
    g.add_argument("--tables-dir", type=Path, default=None,
                   help="Directory for the imported CSV/NPY tables, the merged "
                        "output and the bundle. Default: <output-root>/<name>_tables.")

    g = p.add_argument_group("resample 1 -- phase-change tree")
    g.add_argument("--limit1", type=int, default=None,
                   help="--limit passed to prepare_hefesto_tree_from_phase_changes.py. "
                        "Default: 3x the SimulationN count in --workspace.")

    g = p.add_argument_group("resample 2 -- fine deep tree")
    g.add_argument("--limit2", type=int, default=None,
                   help="--limit passed to prepare_hefesto_tree_fine.py. "
                        "Default: 3x the SimulationN count in --workspace.")
    g.add_argument("--deep-axis", choices=("isotherm", "isobar", "both"),
                   default="isotherm",
                   help="Scan axis for both the deep bounds export and the fine "
                        "tree. 'both' is only valid for the import step; it is "
                        "narrowed to 'isotherm' for the fine-tree build.")
    g.add_argument("--deep-dp", type=float, default=0.01,
                   help="Pressure step (GPa) for the fine isotherm tree.")
    g.add_argument("--deep-dt", type=float, default=None,
                   help="Temperature step (K) for the fine isobar tree. Left "
                        "unset, prepare_hefesto_tree_fine.py derives it from "
                        "--clapeyron.")
    g.add_argument("--clapeyron", type=float, default=None,
                   help="Clapeyron slope magnitude (MPa/K) used to derive "
                        "--deep-dt when that is unset.")

    g = p.add_argument_group("HeFESTo execution")
    g.add_argument("--runner", choices=("local", "cluster"), default="local",
                   help="local: run_hefesto_parallel.py (GNU parallel, blocks). "
                        "cluster: run_many_sbatches_grouped.py + a squeue wait "
                        "barrier that blocks until every job is DONE.")
    g.add_argument("--jobs", type=int, default=None,
                   help="[local] concurrent GNU parallel workers.")
    g.add_argument("--hefesto-cmd", default="$HOME/HeFESTo/HeFESToRepository/main",
                   help="Command invoked inside each SimulationN directory.")
    g.add_argument("--sbatch-time-limit", type=int, default=30,
                   help="[cluster] --time-limit (minutes per 10-sim worker).")
    g.add_argument("--sbatch-max-queued", type=int, default=9500,
                   help="[cluster] --max-queued for the submitter.")
    g.add_argument("--poll-interval", type=int, default=180,
                   help="[cluster] seconds between wait-barrier squeue checks.")
    g.add_argument("--squeue-filter", default="hefesto_",
                   help="[cluster] substring matched against `squeue -o %%j` job "
                        "names; the barrier proceeds once no matching job remains.")

    g = p.add_argument_group("misc")
    g.add_argument("--python", default="python",
                   help="Python interpreter used in the emitted script.")
    g.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                   help="nGibbs repo root (to locate the helper scripts).")
    g.add_argument("--verify", type=int, default=2,
                   help="--verify N forwarded to every derivative import.")
    g.add_argument("--import-args", default="",
                   help="Extra arguments appended verbatim to every "
                        "import_hefesto_subdirs_derivs.py call "
                        "(e.g. \"--no-recover --param-dir /x\").")
    g.add_argument("--no-clean", action="store_true",
                   help="Do not delete pre-existing table/sidecar/manifest files "
                        "before each import. Default is to clean so imports never "
                        "append onto stale rows.")
    return p.parse_args()


def q(value) -> str:
    """Shell-quote, leaving $VAR / ${VAR} refs usable by not quoting bare names."""
    return shlex.quote(str(value))


def main() -> None:
    a = parse_args()

    workspace = a.workspace.resolve()
    output_root = a.output_root.resolve()
    control_dir = a.control_dir.resolve()
    repo_root = a.repo_root.resolve()

    name = a.name or workspace.name

    n_sims = count_simulation_dirs(workspace)
    if n_sims == 0:
        print(f"WARNING: no SimulationN directories found under {workspace}; "
              f"cannot infer --limit1/--limit2. Pass them explicitly.",
              file=sys.stderr)
    limit1 = a.limit1 if a.limit1 is not None else (3 * n_sims or None)
    limit2 = a.limit2 if a.limit2 is not None else (3 * n_sims or None)

    tables_dir = (a.tables_dir.resolve() if a.tables_dir
                  else output_root / f"{name}_tables")

    scripts = repo_root / "scripts"
    import_py = scripts / "import_hefesto_subdirs_derivs.py"
    prep_pc_py = scripts / "prepare_hefesto_tree_from_phase_changes.py"
    prep_fine_py = scripts / "prepare_hefesto_tree_fine.py"
    merge_py = scripts / "merge_bigmetatables.py"
    run_local_py = scripts / "run_hefesto_parallel.py"
    run_grouped_py = scripts / "run_many_sbatches_grouped.py"

    rs1 = output_root / f"{name}_resample1"
    rs2 = output_root / f"{name}_resample2"

    t0 = tables_dir / f"{name}_orig.csv"
    t1 = tables_dir / f"{name}_resample1.csv"
    t2 = tables_dir / f"{name}_resample2.csv"
    merged = tables_dir / f"{name}_merged"
    bundle = tables_dir / f"{name}_merged_bundle.tar.gz"

    pc1 = tables_dir / f"{name}_phasebounds1.csv"
    pc2 = tables_dir / f"{name}_phasebounds2_deep.csv"

    fine_axis = "isotherm" if a.deep_axis == "both" else a.deep_axis

    extra = f" {a.import_args}" if a.import_args.strip() else ""
    verify = f" --verify {a.verify}"

    L: list[str] = []
    w = L.append

    w("#!/usr/bin/env bash")
    w("#")
    w(f"# HeFESTo double-resample pipeline for: {name}")
    w("# Generated by scripts/gen_hefesto_resample_pipeline.py -- edit the")
    w("# CONFIG block below or regenerate with different flags.")
    w(f"# Original workspace SimulationN count: {n_sims}   "
      f"resample --limit = 3x = {3 * n_sims if n_sims else 'n/a'}")
    w("#")
    w("# PREREQUISITE: the HeFESTo runtime environment must already be active in")
    w("# this shell (Benv activated, LD_LIBRARY_PATH / LIBRARY_PATH set, as in")
    w("# scripts/ClusterHeFESTo.sh). The prepare/import/merge steps only need the")
    w("# nGibbs Python env; the run steps need HeFESTo itself.")
    w("")
    w("set -euo pipefail")
    w("")
    w("# ---------------------------------------------------------------- CONFIG")
    w(f"PYTHON={q(a.python)}")
    w(f"REPO_ROOT={q(repo_root)}")
    w(f"WORKSPACE={q(workspace)}")
    w(f"OUTPUT_ROOT={q(output_root)}")
    w(f"CONTROL_DIR={q(control_dir)}")
    w(f"TABLES_DIR={q(tables_dir)}")
    w("")
    w(f"RS1={q(rs1)}                 # phase-change resample tree")
    w(f"RS2={q(rs2)}                 # fine deep resample tree")
    w("")
    w(f"T0={q(t0)}")
    w(f"T1={q(t1)}")
    w(f"T2={q(t2)}")
    w(f"PC1={q(pc1)}                 # phase boundaries -> RS1 (not merged)")
    w(f"PC2={q(pc2)}                 # deep phase boundaries -> RS2 (not merged)")
    w(f"MERGED={q(merged)}")
    w(f"BUNDLE={q(bundle)}")
    w("")
    w(f"RUNNER={q(a.runner)}")
    w(f"HEFESTO_CMD={q(a.hefesto_cmd)}")
    w(f"JOBS={q('' if a.jobs is None else a.jobs)}")
    w(f"SBATCH_TIME_LIMIT={a.sbatch_time_limit}")
    w(f"SBATCH_MAX_QUEUED={a.sbatch_max_queued}")
    w(f"POLL_INTERVAL={a.poll_interval}")
    w(f"SQUEUE_FILTER={q(a.squeue_filter)}")
    w("")
    w(f"IMPORT_PY={q(import_py)}")
    w(f"PREP_PC_PY={q(prep_pc_py)}")
    w(f"PREP_FINE_PY={q(prep_fine_py)}")
    w(f"MERGE_PY={q(merge_py)}")
    w(f"RUN_LOCAL_PY={q(run_local_py)}")
    w(f"RUN_GROUPED_PY={q(run_grouped_py)}")
    w("# --------------------------------------------------------------------- ")
    w("")
    w('mkdir -p "$TABLES_DIR" "$OUTPUT_ROOT"')
    w("")
    w("banner() { printf '\\n========== %s ==========\\n' \"$*\"; }")
    w("")

    # ---- helper: tolerant derivative import (rc 1 == chain-rule warning) -----
    w("# import_hefesto_subdirs_derivs.py exits 1 on a chain-rule verification")
    w("# failure and >=2 on a real error. Treat 1 as a warning so the pipeline")
    w("# still completes; anything higher aborts.")
    w("run_import() {")
    w('  set +e')
    w('  "$PYTHON" "$IMPORT_PY" "$@"')
    w('  local rc=$?')
    w('  set -e')
    w('  if [ "$rc" -ge 2 ]; then')
    w('    echo "ERROR: import failed (rc=$rc)" >&2; exit "$rc"')
    w('  elif [ "$rc" -eq 1 ]; then')
    w('    echo "WARNING: chain-rule verification failure during import (rc=1); continuing" >&2')
    w('  fi')
    w("}")
    w("")

    if not a.no_clean:
        w("# Remove any table/sidecar/manifest from a previous run of this base so")
        w("# the importer starts from empty rather than appending onto stale rows.")
        w("clean_table() {")
        w('  local base="${1%.csv}"')
        w('  rm -f "$base".csv "$base".npy "$base".txt \\')
        w('        "${base}_dndP".csv "${base}_dndP".npy \\')
        w('        "${base}_dndT".csv "${base}_dndT".npy \\')
        w('        "${base}_deriv_manifest".csv \\')
        w('        "${base}blurredbinaries".npy')
        w("}")
        w("")

    # ---- HeFESTo execution: local or cluster+barrier ------------------------
    w("run_hefesto_tree() {")
    w('  local tree="$1"')
    w('  banner "run HeFESTo on $tree ($RUNNER)"')
    w('  if [ "$RUNNER" = "local" ]; then')
    w('    local jflag=()')
    w('    [ -n "$JOBS" ] && jflag=(--jobs "$JOBS")')
    w('    "$PYTHON" "$RUN_LOCAL_PY" --base-dir "$tree" "${jflag[@]}" \\')
    w('        --hefesto-cmd "$HEFESTO_CMD"')
    w('  else')
    w('    ( cd "$tree" && mkdir -p logs && \\')
    w('      "$PYTHON" "$RUN_GROUPED_PY" --base-dir "$tree" \\')
    w('          --time-limit "$SBATCH_TIME_LIMIT" \\')
    w('          --max-queued "$SBATCH_MAX_QUEUED" \\')
    w('          --check-interval "$POLL_INTERVAL" )')
    w('    wait_for_slurm "$tree"')
    w('  fi')
    w("}")
    w("")
    w("# Block until no SLURM job whose name contains $SQUEUE_FILTER is left in")
    w("# the queue for this user, i.e. every submitted job has finished (not")
    w("# merely been submitted). Progress is also reported straight off the tree.")
    w("wait_for_slurm() {")
    w('  local tree="$1"')
    w('  echo "[barrier] waiting for SLURM jobs matching \'$SQUEUE_FILTER\' to finish..."')
    w('  while :; do')
    w("    local n")
    w("    n=$(squeue -u \"$USER\" -h -r -o '%j' 2>/dev/null | grep -c -- \"$SQUEUE_FILTER\" || true)")
    w("    local progress")
    w('    progress=$("$PYTHON" - "$tree" <<\'PY\'')
    w("import os, re, sys")
    w("base = sys.argv[1]")
    w(r"pat = re.compile(r'^Simulation\d+$', re.I)")
    w("total = done = 0")
    w("for root, _dirs, files in os.walk(base):")
    w("    if pat.match(os.path.basename(root)):")
    w("        total += 1")
    w("        if set(files) - {'control', 'ad.in'}:")
    w("            done += 1")
    w('print(f"{done}/{total} simulations have output")')
    w("PY")
    w(")")
    w('    echo "[barrier] $(date +%H:%M:%S)  queued/running: ${n:-?}   ${progress}"')
    w('    [ "${n:-0}" -eq 0 ] && break')
    w('    sleep "$POLL_INTERVAL"')
    w('  done')
    w('  echo "[barrier] SLURM queue drained for $tree"')
    w("}")
    w("")

    # ---- stage 0 ----------------------------------------------------------
    w('banner "stage 0 -- import original workspace + export phase boundaries"')
    if not a.no_clean:
        w('clean_table "$T0"')
    w(f'run_import --root "$WORKSPACE" --dataname "$T0" \\')
    w(f'    --phase-change-dataname "$PC1"{verify}{extra}')
    w("")

    # ---- stage 1 --------------------------------------------------------
    w('banner "stage 1a -- build phase-change resample tree -> RS1"')
    w('rm -rf "$RS1"')
    w(f'"$PYTHON" "$PREP_PC_PY" --directory "$RS1" \\')
    w('    --phase-path "$PC1" --control-dir "$CONTROL_DIR"'
      + ("" if limit1 is None else f' \\\n    --limit {limit1}'))
    w("")
    w('run_hefesto_tree "$RS1"')
    w("")
    w('banner "stage 1c -- re-import RS1 + export DEEP phase boundaries"')
    if not a.no_clean:
        w('clean_table "$T1"')
    w(f'run_import --root "$RS1" --dataname "$T1" \\')
    w(f'    --deep-phase-change-dataname "$PC2" --deep-axis {q(a.deep_axis)}{verify}{extra}')
    w("")

    # ---- stage 2 ------------------------------------------------------
    w('banner "stage 2a -- build fine deep resample tree -> RS2"')
    w('rm -rf "$RS2"')
    fine = [f'"$PYTHON" "$PREP_FINE_PY" --directory "$RS2"',
            '--phase-path "$PC2"', '--control-dir "$CONTROL_DIR"',
            '--deep', f'--deep-axis {q(fine_axis)}', f'--deep-dp {a.deep_dp}']
    if a.deep_dt is not None:
        fine.append(f'--deep-dt {a.deep_dt}')
    if a.clapeyron is not None:
        fine.append(f'--clapeyron {a.clapeyron}')
    if limit2 is not None:
        fine.append(f'--limit {limit2}')
    w(" \\\n    ".join(fine))
    w("")
    w('run_hefesto_tree "$RS2"')
    w("")
    w('banner "stage 2c -- re-import RS2 (no phase-boundary export)"')
    if not a.no_clean:
        w('clean_table "$T2"')
    w(f'run_import --root "$RS2" --dataname "$T2"{verify}{extra}')
    w("")

    # ---- stage 3 ----------------------------------------------------------
    w('banner "stage 3 -- merge the three imported tables"')
    w("# Extensionless bases. The phase-boundary CSVs (PC1/PC2) are intentionally")
    w("# not included -- only the three main imported tables are merged.")
    w(f'"$PYTHON" "$MERGE_PY" \\')
    w(f'    --tables "${{T0%.csv}}" "${{T1%.csv}}" "${{T2%.csv}}" \\')
    w(f'    --output "$MERGED" --csv-output header')
    w("")

    # ---- stage 4 --------------------------------------------------------
    w('banner "stage 4 -- tar + gzip the merged table, sidecars and manifests"')
    w('members=()')
    w('for f in \\')
    w('    "$(basename "$MERGED").npy" "$(basename "$MERGED").csv" "$(basename "$MERGED").txt" \\')
    w('    "$(basename "$MERGED")_dndP.npy" "$(basename "$MERGED")_dndP.csv" \\')
    w('    "$(basename "$MERGED")_dndT.npy" "$(basename "$MERGED")_dndT.csv" \\')
    w('    "$(basename "$MERGED")blurredbinaries.npy" \\')
    w('    "$(basename "${T0%.csv}")_deriv_manifest.csv" \\')
    w('    "$(basename "${T1%.csv}")_deriv_manifest.csv" \\')
    w('    "$(basename "${T2%.csv}")_deriv_manifest.csv" ; do')
    w('  [ -f "$TABLES_DIR/$f" ] && members+=("$f")')
    w('done')
    w('if [ "${#members[@]}" -eq 0 ]; then')
    w('  echo "ERROR: nothing to bundle -- merge outputs not found in $TABLES_DIR" >&2')
    w('  exit 1')
    w('fi')
    w('tar -czf "$BUNDLE" -C "$TABLES_DIR" "${members[@]}"')
    w('echo "bundled ${#members[@]} file(s):"')
    w('printf "  %s\\n" "${members[@]}"')
    w("")
    w('banner "done"')
    w('echo "merged table : $MERGED.npy"')
    w('echo "bundle       : $BUNDLE"')
    w("")

    print("\n".join(L))


if __name__ == "__main__":
    main()
