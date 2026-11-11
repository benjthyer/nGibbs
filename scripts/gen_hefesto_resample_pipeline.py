"""
Generate a bash script that drives the full HeFESTo double-resample + merge +
bundle pipeline, and (by default) run it.

The pipeline the emitted script runs, in order:

  stage 0  import the original (already-run) workspace and export a
           phase-boundary CSV
             import_hefesto_subdirs.py --phase-change-dataname ...
  stage 1a build a phase-change resample tree from that CSV
             prepare_hefesto_tree_from_phase_changes.py
  stage 1b run HeFESTo on it (local GNU parallel, or SLURM + wait barrier)
  stage 1c re-import it, this time exporting *deep* phase-boundary bounds
             import_hefesto_subdirs.py --deep-phase-change-dataname ...
  stage 2a build a fine deep-resample tree from the deep bounds
             prepare_hefesto_tree_fine.py --deep
  stage 2b run HeFESTo on it
  stage 2c re-import it (no phase-boundary export this time)
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

Pass --affinity (mutually exclusive with --derivatives) to switch all three
import stages to import_hefesto_subdirs_affinity.py instead, which keeps qout
and parses phase affinities / component chemical potentials from it into a
row-aligned <table>_affinity.csv shadow (+ _affinity_manifest.csv). Stage 3
then also concatenates the three affinity shadows, in the same order as the
main merge, into <merged>_affinity.csv (row parity with the merged table is
checked), and stage 4 folds it and the manifests into the bundle.

With --runner cluster, every import stage is also run on a compute node: it
is submitted as a single-node SLURM job via submit_import_hefesto_subdirs.py
--wait (with --importer derivs / affinity for --derivatives / --affinity),
which blocks until the import finishes (logs in <tables-dir>/logs).

Derived directories follow the repo's existing convention:
  <output-root>/<name>_resample1   (phase-change tree)
  <output-root>/<name>_resample2   (fine deep tree)

The emitted script is always written to disk first (default:
<output-root>/run_<name>_pipeline.sh) and its stages (stage0, stage1a-c,
stage2a-c, stage3, stage4) are each a standalone shell function over a fixed
CONFIG block, so a run that gets interrupted can be resumed -- without
re-running this generator -- by copy-pasting the CONFIG block plus function
definitions into a shell, or simply re-invoking the saved script with the
stage to resume from:

  bash /scratch/hefesto/run_earthadiabats_pipeline.sh stage1b

The build stages (1a, 2a) delete and rebuild their tree. The run stages (1b,
2b) are safe to re-run on a partly run tree: the cluster submitter skips
batches already listed in <tree>/submitted_jobs.log, and the local runner
skips simulations that already have output. So an interrupted run stage is
resumed by re-invoking that stage. `stage1` / `stage2` are accepted as
shorthand for stage1a / stage2a.

An optional second argument stops the run after that stage instead of going
on to stage4, e.g. to only import a workspace on one machine and continue the
pipeline on another:

  bash /scratch/hefesto/run_earthadiabats_pipeline.sh stage0 stage0

As an end stage, `stage1` / `stage2` mean stage1c / stage2c (the whole of
stage 1 / 2).

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
      --sbatch-time-limit 20 --squeue-filter hefesto_ --import-hours 8
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

STAGES = ("stage0", "stage1a", "stage1b", "stage1c",
          "stage2a", "stage2b", "stage2c", "stage3", "stage4")
# Pre-split stage names, kept so old resume commands still work
STAGE_ALIASES = {"stage1": "stage1a", "stage2": "stage2a"}
# ...and as an end stage, where they mean the whole of stage 1 / 2
END_STAGE_ALIASES = {"stage1": "stage1c", "stage2": "stage2c"}


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
                        "barrier that blocks until every job is DONE; imports are "
                        "submitted via submit_import_hefesto_subdirs.py --wait.")
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
    g.add_argument("--import-hours", type=int, default=6,
                   help="[cluster] time limit (hours) for each import SLURM job.")
    g.add_argument("--import-mem", default="16G",
                   help="[cluster] memory for each import SLURM job.")

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
    g.add_argument("--affinity", action="store_true",
                   help="Use import_hefesto_subdirs_affinity.py instead of the "
                        "default import_hefesto_subdirs.py for every import "
                        "stage, parsing phase affinities from qout into a "
                        "row-aligned <table>_affinity.csv shadow. Stage 3 also "
                        "concatenates the three shadows into <merged>_affinity.csv "
                        "and stage 4 bundles it with the manifests. Mutually "
                        "exclusive with --derivatives. Off by default.")
    g.add_argument("--verify", type=int, default=2,
                   help="[--derivatives only] --verify N forwarded to every "
                        "derivative import (chain-rule check on N sims/workspace).")
    g.add_argument("--import-args", default="",
                   help="Extra arguments appended verbatim to every import call "
                        "(e.g. \"--phase-change-offset-only\", or, with "
                        "--derivatives, \"--no-recover --param-dir /x\", or, with "
                        "--affinity, \"--delete-qout-after-parse\").")
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
                    choices=STAGES + tuple(STAGE_ALIASES),
                    help="Stage to start the run from when executing the script "
                         "(runs that stage and every one after it). Useful for "
                         "resuming a previously interrupted run without redoing "
                         "earlier, possibly expensive, stages. Ignored with "
                         "--print-only; pass it directly to the saved script "
                         "instead, e.g. `bash run_x_pipeline.sh stage1b`. "
                         "stage1/stage2 mean stage1a/stage2a.")
    g.add_argument("--end-stage", default="stage4",
                    choices=STAGES + tuple(END_STAGE_ALIASES),
                    help="Last stage to run when executing the script (default: "
                         "stage4, i.e. run to the end). E.g. --start-stage stage0 "
                         "--end-stage stage0 only imports the original workspace. "
                         "Ignored with --print-only; pass it as the saved script's "
                         "second argument instead, e.g. `bash run_x_pipeline.sh "
                         "stage0 stage0`. stage1/stage2 mean stage1c/stage2c.")
    a = p.parse_args()
    if a.derivatives and a.affinity:
        p.error("--derivatives and --affinity are mutually exclusive")
    return a


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
    # submit_import_hefesto_subdirs.py --importer name for each import script
    importer = ("derivs" if a.derivatives else
                "affinity" if a.affinity else "plain")
    import_py = scripts / {
        "plain": "import_hefesto_subdirs.py",
        "derivs": "import_hefesto_subdirs_derivs.py",
        "affinity": "import_hefesto_subdirs_affinity.py",
    }[importer]
    submit_import_py = scripts / "submit_import_hefesto_subdirs.py"
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
    w("# USAGE:  bash $0 [STAGE [END_STAGE]]")
    w(f"#   STAGE in {{{','.join(STAGES)}}}")
    w("#   (stage1 / stage2 = stage1a / stage2a). Runs STAGE and everything after")
    w("#   it (default: stage0, i.e. everything), stopping after END_STAGE if")
    w("#   given (default: stage4; as an end, stage1 / stage2 = stage1c / stage2c).")
    w("#   E.g. only import the original workspace:  bash $0 stage0 stage0")
    w("#   If a run gets interrupted partway, resume it -- without rebuilding")
    w("#   earlier, possibly expensive, stages -- by re-running with the stage")
    w("#   name it stopped in, e.g.:")
    w("#     bash $0 stage1b")
    w("#   1a/2a delete and rebuild their tree. 1b/2b are safe to re-run on a")
    w("#   partly run tree: already-submitted batches (submitted_jobs.log) or")
    w("#   already-run simulations are skipped.")
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
    w(f"IMPORT_JOB_NAME={q(f'imp_{name}')}")
    w(f"IMPORT_HOURS={a.import_hours}")
    w(f"IMPORT_MEM={q(a.import_mem)}")
    w("")
    if importer != "plain":
        w(f"IMPORTER={q(importer)}           # submit_import_hefesto_subdirs.py --importer")
    w(f"IMPORT_PY={q(import_py)}")
    w(f"SUBMIT_IMPORT_PY={q(submit_import_py)}")
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

    # Both run modes go through run_import. In cluster mode every import --
    # whichever importer -- is submitted as a single-node SLURM job and the
    # call blocks until it finishes; the submitter exits with the job's code.
    tolerant = None
    if a.derivatives:
        w("# import_hefesto_subdirs_derivs.py exits 1 on a chain-rule verification")
        w("# failure and >=2 on a real error. Treat 1 as a warning so the pipeline")
        w("# still completes; anything higher aborts.")
        tolerant = "chain-rule verification failure during import (rc=1); continuing"
    elif a.affinity:
        w("# import_hefesto_subdirs_affinity.py exits 1 when the affinity shadow's")
        w("# row count does not match the main table's, and 2 on a real error.")
        w("# Treat 1 as a warning so the main pipeline still completes (stage 3")
        w("# re-checks parity and skips the misaligned affinity merge); 2+ aborts.")
        tolerant = "affinity/main row parity mismatch during import (rc=1); continuing"
    w("# cluster: submit the import as a single-node SLURM job and block until")
    w("# it finishes (exit code is the job's); logs land in $TABLES_DIR/logs.")
    w("run_import() {")
    if tolerant:
        w('  set +e')
    w('  if [ "$RUNNER" = "cluster" ]; then')
    w('    ( cd "$TABLES_DIR" && "$PYTHON" "$SUBMIT_IMPORT_PY" --wait \\')
    if importer != "plain":
        w('        --importer "$IMPORTER" \\')
    w('        --job-name "$IMPORT_JOB_NAME" --hours "$IMPORT_HOURS" \\')
    w('        --mem "$IMPORT_MEM" "$@" )')
    w('  else')
    w('    "$PYTHON" "$IMPORT_PY" "$@"')
    w('  fi')
    if tolerant:
        w('  local rc=$?')
        w('  set -e')
        w('  if [ "$rc" -ge 2 ]; then')
        w('    echo "ERROR: import failed (rc=$rc)" >&2; exit "$rc"')
        w('  elif [ "$rc" -eq 1 ]; then')
        w(f'    echo "WARNING: {tolerant}" >&2')
        w('  fi')
    w("}")
    w("")

    if not a.no_clean:
        w("# Remove any table (and, with --derivatives, sidecar/manifest) from a")
        w("# previous run of this base so the importer starts from empty rather")
        w("# than appending onto stale rows. Also removes the importer's resume")
        w("# checkpoint -- left behind by an interrupted import, it would make the")
        w("# fresh import skip every simulation up to it -- and any extra files")
        w("# given (the phase-boundary CSV the import appends to).")
        w("clean_table() {")
        w('  local base="${1%.csv}"')
        if a.derivatives:
            w('  rm -f "$base".csv "$base".npy "$base".txt \\')
            w('        "${base}_dndP".csv "${base}_dndP".npy \\')
            w('        "${base}_dndT".csv "${base}_dndT".npy \\')
            w('        "${base}_deriv_manifest".csv \\')
            w('        "${base}blurredbinaries".npy')
        elif a.affinity:
            w('  rm -f "$base".csv "$base".npy "$base".txt \\')
            w('        "${base}_affinity".csv "${base}_affinity_manifest".csv \\')
            w('        "${base}blurredbinaries".npy')
        else:
            w('  rm -f "$base".csv "$base".npy "$base".txt "${base}blurredbinaries".npy')
        w('  rm -f "$base".csv.checkpoint.txt "${@:2}"')
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
    # stage to resume from (e.g. `bash $0 stage1b`), or by copy-pasting the
    # CONFIG block, the helper functions, and the one stageN function you
    # want straight into an interactive shell.

    # ---- stage 0 ----------------------------------------------------------
    w("stage0() {")
    w('  banner "stage 0 -- import original workspace + export phase boundaries"')
    if not a.no_clean:
        w('  clean_table "$T0" "$PC1"')
    w(f'  run_import --root "$WORKSPACE" --dataname "$T0" \\')
    w(f'      --phase-change-dataname "$PC1"{verify}{extra}')
    w("}")
    w("")

    # ---- stage 1 --------------------------------------------------------
    w("stage1a() {")
    w('  banner "stage 1a -- build phase-change resample tree -> RS1"')
    w('  rm -rf "$RS1"')
    w(f'  "$PYTHON" "$PREP_PC_PY" --directory "$RS1" \\')
    w('      --phase-path "$PC1" --control-dir "$CONTROL_DIR"'
      + ("" if limit1 is None else f' \\\n      --limit {limit1}'))
    w("}")
    w("")
    w("# Safe to re-run on a partly run RS1 (see the USAGE note at the top).")
    w("stage1b() {")
    w('  banner "stage 1b -- run HeFESTo on RS1"')
    w('  run_hefesto_tree "$RS1"')
    w("}")
    w("")
    w("stage1c() {")
    w('  banner "stage 1c -- re-import RS1 + export DEEP phase boundaries"')
    if not a.no_clean:
        w('  clean_table "$T1" "$PC2"')
    w(f'  run_import --root "$RS1" --dataname "$T1" \\')
    w(f'      --deep-phase-change-dataname "$PC2" --deep-axis {q(a.deep_axis)}{verify}{extra}')
    w("}")
    w("")

    # ---- stage 2 ------------------------------------------------------
    w("stage2a() {")
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
    w("}")
    w("")
    w("# Safe to re-run on a partly run RS2 (see the USAGE note at the top).")
    w("stage2b() {")
    w('  banner "stage 2b -- run HeFESTo on RS2"')
    w('  run_hefesto_tree "$RS2"')
    w("}")
    w("")
    w("stage2c() {")
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
    if a.affinity:
        w("")
        w('  banner "stage 3b -- concatenate the affinity shadows (same order)"')
        w("  # merge_bigmetatables.py only concatenates rows (it never drops or")
        w("  # reorders them), so concatenating the shadows in the same T0/T1/T2")
        w("  # order keeps them row-aligned with the merged table. Parity is")
        w("  # checked; on a mismatch the merged shadow is not written.")
        w('  rm -f "${MERGED}_affinity.csv"')
        w('  "$PYTHON" - "$MERGED" "${T0%.csv}_affinity.csv" "${T1%.csv}_affinity.csv" \\')
        w("      \"${T2%.csv}_affinity.csv\" <<'PY' || \\")
        w('    echo "WARNING: affinity merge skipped (see message above); main merge is unaffected" >&2')
        w("import os, sys")
        w("import numpy as np, pandas as pd")
        w("merged, srcs = sys.argv[1], sys.argv[2:]")
        w("missing = [s for s in srcs if not os.path.isfile(s)]")
        w("if missing:")
        w('    sys.exit(f"missing affinity shadow(s): {missing}")')
        w("frames = [pd.read_csv(s) for s in srcs]")
        w("cols = list(frames[0].columns)")
        w("for s, f in zip(srcs, frames):")
        w("    if list(f.columns) != cols:")
        w('        sys.exit(f"affinity header mismatch: {s}")')
        w("out = pd.concat(frames, ignore_index=True)")
        w("n_main = np.load(merged + '.npy', mmap_mode='r').shape[0]")
        w("if len(out) != n_main:")
        w('    sys.exit(f"affinity rows {len(out)} != merged rows {n_main}")')
        w("out.to_csv(merged + '_affinity.csv', index=False)")
        w('print(f"affinity shadow: {len(out)} rows -> {merged}_affinity.csv (parity MATCH)")')
        w("PY")
    w("}")
    w("")

    # ---- stage 4 --------------------------------------------------------
    w("stage4() {")
    if a.derivatives or a.affinity:
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
    elif a.affinity:
        w('      "$(basename "$MERGED")blurredbinaries.npy" \\')
        w('      "$(basename "$MERGED")_affinity.csv" \\')
        w('      "$(basename "${T0%.csv}")_affinity_manifest.csv" \\')
        w('      "$(basename "${T1%.csv}")_affinity_manifest.csv" \\')
        w('      "$(basename "${T2%.csv}")_affinity_manifest.csv" ; do')
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

    # ---- dispatch: run STAGE (arg 1, default stage0) through END_STAGE ----
    # (arg 2, default stage4)
    w("# ------------------------------------------------------------- DISPATCH")
    w(f'STAGES=({" ".join(STAGES)})')
    w('START_STAGE="${1:-stage0}"')
    w('END_STAGE="${2:-stage4}"')
    w("# pre-split stage names: the first sub-stage as a start, the last as an end")
    for old, new in STAGE_ALIASES.items():
        w(f'[ "$START_STAGE" = {old} ] && START_STAGE={new}')
    for old, new in END_STAGE_ALIASES.items():
        w(f'[ "$END_STAGE" = {old} ] && END_STAGE={new}')
    w("")
    w('if [ "$START_STAGE" = "-h" ] || [ "$START_STAGE" = "--help" ]; then')
    w('  echo "Usage: $0 [STAGE [END_STAGE]]" >&2')
    w('  echo "  STAGE, END_STAGE: one of ${STAGES[*]}" >&2')
    w('  echo "  (defaults: stage0, stage4)." >&2')
    w('  echo "  Runs STAGE through END_STAGE -- pass the stage where a previous" >&2')
    w('  echo "  run was interrupted to resume without redoing earlier, possibly" >&2')
    w('  echo "  expensive, stages; pass END_STAGE to stop early (e.g. stage0" >&2')
    w('  echo "  stage0 only imports the original workspace)." >&2')
    w('  echo "  stage1 / stage2 mean stage1a / stage2a as STAGE and" >&2')
    w('  echo "  stage1c / stage2c as END_STAGE." >&2')
    w('  exit 0')
    w('fi')
    w("")
    w("stage_index() {")
    w('  local i')
    w('  for i in "${!STAGES[@]}"; do')
    w('    [ "${STAGES[$i]}" = "$1" ] && { echo "$i"; return; }')
    w('  done')
    w('  echo -1')
    w("}")
    w('START_I=$(stage_index "$START_STAGE")')
    w('END_I=$(stage_index "$END_STAGE")')
    w('if [ "$START_I" -lt 0 ]; then')
    w('  echo "ERROR: unknown stage \'$START_STAGE\' (expected one of ${STAGES[*]})" >&2')
    w('  exit 1')
    w('fi')
    w('if [ "$END_I" -lt 0 ]; then')
    w('  echo "ERROR: unknown end stage \'$END_STAGE\' (expected one of ${STAGES[*]})" >&2')
    w('  exit 1')
    w('fi')
    w('if [ "$END_I" -lt "$START_I" ]; then')
    w('  echo "ERROR: end stage $END_STAGE comes before start stage $START_STAGE" >&2')
    w('  exit 1')
    w('fi')
    w("")
    w('for s in "${STAGES[@]:START_I:END_I-START_I+1}"; do "$s"; done')
    w('if [ "$END_I" -lt $(( ${#STAGES[@]} - 1 )) ]; then')
    w('  banner "stopped after $END_STAGE"')
    w('  echo "continue with: bash $0 ${STAGES[END_I+1]}"')
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
          f"bash {q(script_path)} <stage> [<end stage>]  ({', '.join(STAGES)})",
          file=sys.stderr)

    if a.print_only:
        print(script_text)
        return

    result = subprocess.run(["bash", str(script_path), a.start_stage, a.end_stage])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
