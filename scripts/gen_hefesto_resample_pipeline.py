"""
Generate a bash script that drives the full HeFESTo double-resample + merge +
bundle pipeline, and (by default) run it.

The pipeline the emitted script runs, in order:

  stage 0  import the original (already-run) workspace and export a
           phase-boundary CSV
             import_hefesto_subdirs.py --phase-change-dataname ...
  stage 1  build a phase-change resample tree from that CSV
             prepare_hefesto_tree_from_phase_changes.py
           -> run HeFESTo on it (local GNU parallel, or SLURM + wait barrier)
           -> re-import it, this time exporting *deep* phase-boundary bounds
             import_hefesto_subdirs.py --deep-phase-change-dataname ...
  stage 2  build a fine deep-resample tree from the deep bounds
             prepare_hefesto_tree_fine.py --deep
           -> run HeFESTo on it
           -> re-import it (no phase-boundary export this time)
  stage 3  merge the three imported BigMetaTables (stage 0 / 1 / 2 main
           tables only -- the phase-boundary CSVs used to seed the resample
           trees are deliberately NOT merged)
             merge_bigmetatables.py
  stage 4  tar + gzip the merged table into one <name>_merged_bundle.tar.gz

By default no composition derivatives (fort.42, dn/dP, dn/dT) are read,
reconstructed or saved anywhere in this pipeline -- every import stage runs
the derivative-free import_hefesto_subdirs.py. Pass --derivatives to switch
all three import stages to import_hefesto_subdirs_derivs.py instead, which
additionally reconstructs/verifies dn/dP and dn/dT and folds their shadow
tables and manifests into the stage-4 bundle.

Derived directories follow the repo's existing convention:
  <output-root>/<name>_resample1   (phase-change tree)
  <output-root>/<name>_resample2   (fine deep tree)

The emitted script is always written to disk first (default:
<output-root>/run_<name>_pipeline.sh) and its 5 stages (stage0..stage4) are
each a standalone shell function over a fixed CONFIG block, so a run that
gets interrupted can be resumed -- without re-running this generator -- by
copy-pasting the CONFIG block plus function definitions into a shell, or
simply re-invoking the saved script with the stage to resume from:

  bash /scratch/hefesto/run_earthadiabats_pipeline.sh stage2

Usage:
  python scripts/gen_hefesto_resample_pipeline.py \
      --workspace /path/to/OriginalWorkspace \
      --name EarthAdiabats \
      --output-root /scratch/hefesto \
      --control-dir src/builder/HeFESTo/batch/shallowHeFESTo
      # writes + runs .../run_earthadiabats_pipeline.sh

  # write the script and print it, but don't run it:
  python scripts/gen_hefesto_resample_pipeline.py ... --print-only > pipeline.sh

  # cluster execution instead of local GNU parallel:
  python scripts/gen_hefesto_resample_pipeline.py ... --runner cluster \
      --sbatch-time-limit 20 --squeue-filter hefesto_
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
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
    g.add_argument("--derivatives", action="store_true",
                   help="Use import_hefesto_subdirs_derivs.py instead of the "
                        "default derivative-free import_hefesto_subdirs.py for "
                        "every import stage, reconstructing/verifying dn/dP and "
                        "dn/dT from fort.42 and folding their shadow tables and "
                        "manifests into the stage-4 bundle. Off by default.")
    g.add_argument("--verify", type=int, default=2,
                   help="[--derivatives only] --verify N forwarded to every "
                        "derivative import (chain-rule check on N sims/workspace).")
    g.add_argument("--import-args", default="",
                   help="Extra arguments appended verbatim to every import call "
                        "(e.g. \"--phase-change-offset-only\", or, with "
                        "--derivatives, \"--no-recover --param-dir /x\").")
    g.add_argument("--no-clean", action="store_true",
                   help="Do not delete pre-existing table/sidecar/manifest files "
                        "before each import. Default is to clean so imports never "
                        "append onto stale rows.")

    g = p.add_argument_group("script output / execution")
    g.add_argument("--script-path", type=Path, default=None,
                   help="Where to write the generated bash script. Default: "
                        "<output-root>/run_<name>_pipeline.sh.")
    g.add_argument("--print-only", action="store_true",
                   help="Write the script to --script-path and print it to "
                        "stdout, but do not run it (the old default behavior).")
    g.add_argument("--start-stage", default="stage0",
                    choices=("stage0", "stage1", "stage2", "stage3", "stage4"),
                    help="Stage to start the run from when executing the script "
                         "(runs that stage and every one after it). Useful for "
                         "resuming a previously interrupted run without redoing "
                         "earlier, possibly expensive, stages. Ignored with "
                         "--print-only; pass it directly to the saved script "
                         "instead, e.g. `bash run_x_pipeline.sh stage2`.")
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
    import_py = scripts / ("import_hefesto_subdirs_derivs.py" if a.derivatives
                            else "import_hefesto_subdirs.py")
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
    verify = f" --verify {a.verify}" if a.derivatives else ""

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
    w("#")
    w("# USAGE:  bash $0 [STAGE]     STAGE in {stage0,stage1,stage2,stage3,stage4}")
    w("#   Runs STAGE and everything after it (default: stage0, i.e. everything).")
    w("#   If a run gets interrupted partway, resume it -- without rebuilding")
    w("#   earlier, possibly expensive, stages -- by re-running with the stage")
    w("#   name it stopped in, e.g.:")
    w("#     bash $0 stage2")
    w("#   Or copy-paste the CONFIG block below plus the helper/stageN functions")
    w("#   you need directly into an interactive shell.")
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

    if a.derivatives:
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
    else:
        w('run_import() { "$PYTHON" "$IMPORT_PY" "$@"; }')
    w("")

    if not a.no_clean:
        w("# Remove any table (and, with --derivatives, sidecar/manifest) from a")
        w("# previous run of this base so the importer starts from empty rather")
        w("# than appending onto stale rows.")
        w("clean_table() {")
        w('  local base="${1%.csv}"')
        if a.derivatives:
            w('  rm -f "$base".csv "$base".npy "$base".txt \\')
            w('        "${base}_dndP".csv "${base}_dndP".npy \\')
            w('        "${base}_dndT".csv "${base}_dndT".npy \\')
            w('        "${base}_deriv_manifest".csv \\')
            w('        "${base}blurredbinaries".npy')
        else:
            w('  rm -f "$base".csv "$base".npy "$base".txt "${base}blurredbinaries".npy')
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

    # Each stage below is a standalone function over the CONFIG block above,
    # so a run can be resumed after an interruption -- without re-running
    # this generator -- either by re-invoking this saved script with the
    # stage to resume from (e.g. `bash $0 stage2`), or by copy-pasting the
    # CONFIG block, the helper functions, and the one stageN function you
    # want straight into an interactive shell.

    # ---- stage 0 ----------------------------------------------------------
    w("stage0() {")
    w('  banner "stage 0 -- import original workspace + export phase boundaries"')
    if not a.no_clean:
        w('  clean_table "$T0"')
    w(f'  run_import --root "$WORKSPACE" --dataname "$T0" \\')
    w(f'      --phase-change-dataname "$PC1"{verify}{extra}')
    w("}")
    w("")

    # ---- stage 1 --------------------------------------------------------
    w("stage1() {")
    w('  banner "stage 1a -- build phase-change resample tree -> RS1"')
    w('  rm -rf "$RS1"')
    w(f'  "$PYTHON" "$PREP_PC_PY" --directory "$RS1" \\')
    w('      --phase-path "$PC1" --control-dir "$CONTROL_DIR"'
      + ("" if limit1 is None else f' \\\n      --limit {limit1}'))
    w("")
    w('  run_hefesto_tree "$RS1"')
    w("")
    w('  banner "stage 1c -- re-import RS1 + export DEEP phase boundaries"')
    if not a.no_clean:
        w('  clean_table "$T1"')
    w(f'  run_import --root "$RS1" --dataname "$T1" \\')
    w(f'      --deep-phase-change-dataname "$PC2" --deep-axis {q(a.deep_axis)}{verify}{extra}')
    w("}")
    w("")

    # ---- stage 2 ------------------------------------------------------
    w("stage2() {")
    w('  banner "stage 2a -- build fine deep resample tree -> RS2"')
    w('  rm -rf "$RS2"')
    fine = [f'"$PYTHON" "$PREP_FINE_PY" --directory "$RS2"',
            '--phase-path "$PC2"', '--control-dir "$CONTROL_DIR"',
            '--deep', f'--deep-axis {q(fine_axis)}', f'--deep-dp {a.deep_dp}']
    if a.deep_dt is not None:
        fine.append(f'--deep-dt {a.deep_dt}')
    if a.clapeyron is not None:
        fine.append(f'--clapeyron {a.clapeyron}')
    if limit2 is not None:
        fine.append(f'--limit {limit2}')
    w("  " + " \\\n      ".join(fine))
    w("")
    w('  run_hefesto_tree "$RS2"')
    w("")
    w('  banner "stage 2c -- re-import RS2 (no phase-boundary export)"')
    if not a.no_clean:
        w('  clean_table "$T2"')
    w(f'  run_import --root "$RS2" --dataname "$T2"{verify}{extra}')
    w("}")
    w("")

    # ---- stage 3 ----------------------------------------------------------
    w("stage3() {")
    w('  banner "stage 3 -- merge the three imported tables"')
    w("  # Extensionless bases. The phase-boundary CSVs (PC1/PC2) are")
    w("  # intentionally not included -- only the three main imported tables")
    w("  # are merged.")
    w(f'  "$PYTHON" "$MERGE_PY" \\')
    w(f'      --tables "${{T0%.csv}}" "${{T1%.csv}}" "${{T2%.csv}}" \\')
    w(f'      --output "$MERGED" --csv-output header')
    w("}")
    w("")

    # ---- stage 4 --------------------------------------------------------
    w("stage4() {")
    if a.derivatives:
        w('  banner "stage 4 -- tar + gzip the merged table, sidecars and manifests"')
    else:
        w('  banner "stage 4 -- tar + gzip the merged table"')
    w('  members=()')
    w('  for f in \\')
    w('      "$(basename "$MERGED").npy" "$(basename "$MERGED").csv" "$(basename "$MERGED").txt" \\')
    if a.derivatives:
        w('      "$(basename "$MERGED")_dndP.npy" "$(basename "$MERGED")_dndP.csv" \\')
        w('      "$(basename "$MERGED")_dndT.npy" "$(basename "$MERGED")_dndT.csv" \\')
        w('      "$(basename "$MERGED")blurredbinaries.npy" \\')
        w('      "$(basename "${T0%.csv}")_deriv_manifest.csv" \\')
        w('      "$(basename "${T1%.csv}")_deriv_manifest.csv" \\')
        w('      "$(basename "${T2%.csv}")_deriv_manifest.csv" ; do')
    else:
        w('      "$(basename "$MERGED")blurredbinaries.npy" ; do')
    w('    [ -f "$TABLES_DIR/$f" ] && members+=("$f")')
    w('  done')
    w('  if [ "${#members[@]}" -eq 0 ]; then')
    w('    echo "ERROR: nothing to bundle -- merge outputs not found in $TABLES_DIR" >&2')
    w('    exit 1')
    w('  fi')
    w('  tar -czf "$BUNDLE" -C "$TABLES_DIR" "${members[@]}"')
    w('  echo "bundled ${#members[@]} file(s):"')
    w('  printf "  %s\\n" "${members[@]}"')
    w("")
    w('  banner "done"')
    w('  echo "merged table : $MERGED.npy"')
    w('  echo "bundle       : $BUNDLE"')
    w("}")
    w("")

    # ---- dispatch: run STAGE (arg 1, default stage0) through stage4 -------
    w("# ------------------------------------------------------------- DISPATCH")
    w('STAGES=(stage0 stage1 stage2 stage3 stage4)')
    w('START_STAGE="${1:-stage0}"')
    w("")
    w('if [ "$START_STAGE" = "-h" ] || [ "$START_STAGE" = "--help" ]; then')
    w('  echo "Usage: $0 [STAGE]" >&2')
    w('  echo "  STAGE: one of ${STAGES[*]} (default: stage0)." >&2')
    w('  echo "  Runs STAGE and every stage after it -- pass the stage where a" >&2')
    w('  echo "  previous run was interrupted to resume without redoing earlier," >&2')
    w('  echo "  possibly expensive, stages." >&2')
    w('  exit 0')
    w('fi')
    w("")
    w('found=false')
    w('for s in "${STAGES[@]}"; do')
    w('  [ "$s" = "$START_STAGE" ] && found=true')
    w('  if $found; then "$s"; fi')
    w('done')
    w('if ! $found; then')
    w('  echo "ERROR: unknown stage \'$START_STAGE\' (expected one of ${STAGES[*]})" >&2')
    w('  exit 1')
    w('fi')
    w("")

    script_text = "\n".join(L) + "\n"

    script_path = (a.script_path.resolve() if a.script_path
                   else output_root / f"run_{name}_pipeline.sh")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_text)
    script_path.chmod(0o755)

    print(f"Wrote pipeline script to {script_path}", file=sys.stderr)
    print(f"Resume (or re-run from a given stage) with: "
          f"bash {q(script_path)} stageN", file=sys.stderr)

    if a.print_only:
        print(script_text)
        return

    result = subprocess.run(["bash", str(script_path), a.start_stage])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
