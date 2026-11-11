#!/usr/bin/env python3
"""
Submit a HeFESTo import script as a single-node SLURM job from a login node.

--importer picks which import script the job runs:
  plain     import_hefesto_subdirs.py          (default)
  derivs    import_hefesto_subdirs_derivs.py   (+ dn/dP, dn/dT shadows)
  affinity  import_hefesto_subdirs_affinity.py (+ qout affinity shadow)

Any switch not consumed here is forwarded verbatim to that script (--root,
--dataname, --phase-change-dataname, ...). They are validated against the
chosen script's parser before submission, so a typo fails here rather than
after the job has waited in the queue.

The job runs from the directory you submit from, so relative paths (including
the defaults --root . and --dataname DefaultHeFESTostorage.csv) resolve exactly
as they would if you ran the import directly.

With --wait the submitter blocks until the job finishes, echoes its log, and
exits with the job's exit code -- this is how gen_hefesto_resample_pipeline.py
runs its import stages in cluster mode.

Example:
    python scripts/submit_import_hefesto_subdirs.py --hours 12 --mem 32G \\
        --root /scratch/me/runs --dataname runs.csv --phase-change-dataname pc.csv
"""

from __future__ import annotations

import argparse
import importlib
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
IMPORTERS = {
    "plain": "import_hefesto_subdirs",
    "derivs": "import_hefesto_subdirs_derivs",
    "affinity": "import_hefesto_subdirs_affinity",
}

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def build_import_parser(importer: str) -> argparse.ArgumentParser:
    return importlib.import_module(IMPORTERS[importer]).build_parser()


def _quote_args(argv: list[str]) -> str:
    """Shell-quote argv for the bash script (shlex.join needs Python 3.8)."""
    return " ".join(shlex.quote(a) for a in argv)


def create_slurm_script(
    submit_dir: Path,
    import_script: Path,
    import_args: list[str],
    hours: int,
    mem: str,
    cpus: int,
    job_name: str,
    venv: str,
    partition: str | None,
) -> Path:
    """Write the SLURM script into submit_dir/logs and return its path."""
    logs_dir = submit_dir / "logs"
    # SLURM opens --output/--error before the script runs, so logs/ must exist now.
    logs_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_script = logs_dir / f"{job_name}_{stamp}.slurm"

    partition_line = f"#SBATCH --partition={partition}\n" if partition else ""
    command = _quote_args(["python", "-u", str(import_script), *import_args])

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={hours:02d}:00:00
{partition_line}#SBATCH --output=logs/{slurm_script.stem}_%j.out
#SBATCH --error=logs/{slurm_script.stem}_%j.err

# Preparations
source {venv}/bin/activate
export OPENBLAS_NUM_THREADS={cpus}

# Run from the submission directory so relative paths match the login CLI
cd "$SLURM_SUBMIT_DIR"

{command}
"""

    slurm_script.write_text(script_content)
    slurm_script.chmod(0o755)  # Make executable

    return slurm_script


def submit_job(args: argparse.Namespace, import_args: list[str]):
    submit_dir = Path.cwd()

    slurm_script = create_slurm_script(
        submit_dir,
        SCRIPTS_DIR / f"{IMPORTERS[args.importer]}.py",
        import_args,
        hours=args.hours,
        mem=args.mem,
        cpus=args.cpus,
        job_name=args.job_name,
        venv=args.venv,
        partition=args.partition,
    )
    print(f"Created: {slurm_script}")
    print(f"Importer: {IMPORTERS[args.importer]}.py")
    print(f"Forwarded import args: {_quote_args(import_args) or '(defaults)'}")

    # Logs share the script's timestamped stem, so --wait can find them afterwards
    log_stem = slurm_script.parent / slurm_script.stem

    if args.dry_run:
        wait_flag = "--wait " if args.wait else ""
        print(f"[DRY RUN] Would submit: sbatch {wait_flag}{slurm_script}")
        return

    if args.wait:
        print(f"Waiting for job to finish (log: {log_stem}_<jobid>.out)", flush=True)
        # sbatch --wait prints the job ID, blocks, then exits with the job's exit code
        rc = subprocess.run(["sbatch", "--wait", str(slurm_script)], cwd=submit_dir).returncode

        for out_log in sorted(log_stem.parent.glob(f"{log_stem.name}_*.out")):
            print(f"----- {out_log} -----")
            print(out_log.read_text(), end="", flush=True)
        for err_log in sorted(log_stem.parent.glob(f"{log_stem.name}_*.err")):
            err_text = err_log.read_text()
            if err_text.strip():
                print(f"----- {err_log} -----", file=sys.stderr)
                print(err_text, end="", file=sys.stderr, flush=True)

        if rc != 0:
            print(f"✗ Job failed (exit code {rc})", file=sys.stderr)
            sys.exit(rc)
        print("✓ Job finished")
        return

    try:
        result = subprocess.run(
            ["sbatch", str(slurm_script)],
            cwd=submit_dir,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit: {e.stderr}")
        sys.exit(1)

    # Extract job ID from output like "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    print(f"✓ Submitted job {job_id}")
    print(f"Log: {log_stem}_{job_id}.out")
    print("Monitor with: squeue -u $USER")
    print(f"Cancel with: scancel {job_id}")


if __name__ == "__main__":
    # Pick the importer first so --help can show that script's options
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--importer", choices=sorted(IMPORTERS), default="plain")
    importer = pre.parse_known_args()[0].importer

    parser = argparse.ArgumentParser(
        description=(
            "Submit a HeFESTo import script as a single-node SLURM job. "
            "Unrecognised switches are forwarded to the import script."
        ),
        epilog=(f"Import script options ({IMPORTERS[importer]}.py):\n"
                + build_import_parser(importer).format_help()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Prefix matching could swallow a forwarded switch (e.g. --d -> --dry-run)
        allow_abbrev=False,
    )
    parser.add_argument(
        "--importer",
        choices=sorted(IMPORTERS),
        default="plain",
        help="Import script the job runs (default: plain = import_hefesto_subdirs.py)"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=6,
        help="Time limit for the job in hours (default: 6)"
    )
    parser.add_argument(
        "--mem",
        type=str,
        default="16G",
        help="Memory for the job (default: 16G)"
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="CPUs per task (default: 1)"
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="import_hefesto",
        help="SLURM job name, also used for log file names (default: import_hefesto)"
    )
    parser.add_argument(
        "--venv",
        type=str,
        default="$HOME/HeFESTo/Benv",
        help="Virtual environment activated in the job (default: $HOME/HeFESTo/Benv)"
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="SLURM partition (default: cluster default)"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block until the job finishes, print its log, and exit with its exit code"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the SLURM script without submitting it"
    )

    args, import_args = parser.parse_known_args()

    # Fail on the login node, not after queueing, if the forwarded args are bad
    build_import_parser(args.importer).parse_args(import_args)

    submit_job(args, import_args)
