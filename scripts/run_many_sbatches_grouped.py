#!/usr/bin/env python3
"""
Submit SLURM array jobs for all simulation directories under BASE_DIR.

Each array job spans 100 workers; each worker runs 10 simulation directories
sequentially. Worker N handles Simulation directories base-9 through base,
where base = N * 10.

Monitors queue size and submits new jobs as capacity becomes available.
"""

import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def count_simulations(sim_dir: Path) -> int:
    """Count how many Simulation<N> directories exist in sim_dir."""
    sim_folders = list(sim_dir.glob("Simulation*"))
    valid_sims = [
        s for s in sim_folders
        if s.name.replace("Simulation", "").isdigit()
    ]
    return len(valid_sims)


def get_queued_jobs_count() -> int:
    """Get the current number of queued/running jobs for this user."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ["USER"], "-h", "-r"],
            capture_output=True,
            text=True,
            check=True
        )
        job_lines = [line for line in result.stdout.strip().split('\n') if line]
        return len(job_lines)
    except subprocess.CalledProcessError:
        print("Warning: Could not query job queue, assuming 0 jobs")
        return 0


def create_slurm_script(sim_dir: Path, n_workers: int, time_limit: int) -> Path:
    """Create a SLURM script where each array task runs 10 simulations sequentially."""
    slurm_script = sim_dir / "run_hefesto_array.slurm"

    script_content = f"""#!/bin/bash
#SBATCH --job-name=hefesto_{sim_dir.name}
#SBATCH --array=1-{n_workers}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:{time_limit:02d}:00
#SBATCH --output=logs/hefesto_%A_%a.out
#SBATCH --error=logs/hefesto_%A_%a.err

# Preparations
mkdir -p logs
source $HOME/HeFESTo/Benv/bin/activate
export LD_LIBRARY_PATH=$HOME/HeFESTo/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/HeFESTo/lib64:$LIBRARY_PATH

# Worker N handles Simulation directories (N*10)-9 through N*10
BASE_NUM=$(( SLURM_ARRAY_TASK_ID * 10 ))
for OFFSET in $(seq 0 9); do
    DIR_NUM=$(( BASE_NUM - OFFSET ))
    ACTIVE_DIR=Simulation${{DIR_NUM}}
    if [ -d "$ACTIVE_DIR" ]; then
        (cd "$ACTIVE_DIR" && $HOME/HeFESTo/HeFESToRepository/main)
    fi
done
"""

    slurm_script.write_text(script_content)
    slurm_script.chmod(0o755)

    return slurm_script


def load_checkpoint(base_dir: Path) -> set[str]:
    """Return set of already-submitted directory names from submitted_jobs.log."""
    log_path = base_dir / "submitted_jobs.log"
    if not log_path.exists():
        return set()
    submitted = set()
    with log_path.open() as f:
        for line in f:
            if "  job=" in line:
                submitted.add(line.split("  job=")[0].strip())
    return submitted


def submit_job(sim_dir: Path, slurm_script: Path) -> str | None:
    """Submit a single SLURM job. Returns job_id on success, None on failure."""
    try:
        result = subprocess.run(
            ["sbatch", str(slurm_script)],
            cwd=sim_dir,
            capture_output=True,
            text=True,
            check=True
        )
        job_id = result.stdout.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to submit: {e.stderr}")
        return None


def submit_jobs_with_queue_monitoring(
    base_dir: Path,
    max_queued: int = 10,
    check_interval: int = 60,
    dry_run: bool = False,
    time_limit: int = 100,
):
    """
    Submit grouped jobs while respecting queue limits.

    Args:
        base_dir: Base directory containing SIM_DIR folders
        max_queued: Maximum number of jobs to have queued at once
        check_interval: How often to check queue status (seconds)
        dry_run: If True, print what would be submitted without actually submitting
        time_limit: Time limit for each worker in minutes (covers 10 simulations)
    """
    base_dir = base_dir.resolve()

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        sys.exit(1)

    # Find all directories that contain Simulation* subdirectories
    sim_dirs = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir():
            n_sims = count_simulations(item)
            if n_sims > 0:
                n_workers = math.ceil(n_sims / 10)
                sim_dirs.append((item, n_sims, n_workers))

    if not sim_dirs:
        print(f"No directories with Simulation* folders found in {base_dir}")
        sys.exit(1)

    # Checkpoint: skip already-submitted directories
    already_submitted = load_checkpoint(base_dir)
    if already_submitted:
        skipped = [(d, ns, nw) for d, ns, nw in sim_dirs if d.name in already_submitted]
        sim_dirs = [(d, ns, nw) for d, ns, nw in sim_dirs if d.name not in already_submitted]
        print(f"Checkpoint found ({base_dir / 'submitted_jobs.log'}):")
        print(f"  Skipping {len(skipped)} already-submitted director{'y' if len(skipped)==1 else 'ies'}: "
              + ", ".join(d.name for d, _, _ in skipped))
        if not sim_dirs:
            print("All directories already submitted. Nothing to do.")
            return

    total_sims = sum(ns for _, ns, _ in sim_dirs)
    total_workers = sum(nw for _, _, nw in sim_dirs)

    print("=" * 70)
    print(f"Found {len(sim_dirs)} directories with {total_sims} total simulations:")
    for sim_dir, n_sims, n_workers in sim_dirs:
        print(f"  {sim_dir.name}: {n_sims} simulations → {n_workers} workers")
    print(f"\nQueue limit: {max_queued} jobs")
    print(f"Check interval: {check_interval} seconds")
    print(f"Total workers to submit: {total_workers}")
    print("=" * 70)
    print()

    # Prepare all SLURM scripts first
    pending_jobs = []
    for sim_dir, n_sims, n_workers in sim_dirs:
        slurm_script = create_slurm_script(sim_dir, n_workers, time_limit=time_limit)
        pending_jobs.append((sim_dir, slurm_script, n_workers))
        print(f"Prepared: {slurm_script} ({n_workers} workers)")

    print(f"\n{len(pending_jobs)} job arrays ready to submit")
    print()

    if dry_run:
        print("[DRY RUN MODE - Not actually submitting]")
        for sim_dir, slurm_script, n_workers in pending_jobs:
            print(f"Would submit: {sim_dir.name} ({n_workers} workers, 10 sims each)")
        return

    log_path = base_dir / "submitted_jobs.log"

    with log_path.open("a") as log_f:
        log_f.write(f"# Run started {datetime.now().isoformat()}\n")

    submitted_jobs = []

    while pending_jobs:
        current_queued = get_queued_jobs_count()
        print(f"[{time.strftime('%H:%M:%S')}] Current jobs in queue: {current_queued}/{max_queued}")

        jobs_to_remove = []

        for i, (sim_dir, slurm_script, n_workers) in enumerate(pending_jobs):
            if current_queued + n_workers <= max_queued:
                print(f"  Submitting {sim_dir.name} ({n_workers} workers)...", end=" ")
                job_id = submit_job(sim_dir, slurm_script)

                if job_id:
                    submitted_jobs.append((sim_dir.name, job_id, n_workers))
                    jobs_to_remove.append(i)
                    current_queued += n_workers
                    print(f"✓ Job {job_id}")
                    with log_path.open("a") as log_f:
                        log_f.write(f"{sim_dir.name}  job={job_id}  tasks={n_workers}\n")
                else:
                    print("✗ Failed")
            else:
                print(f"  Cannot submit {sim_dir.name} ({n_workers} workers) - would exceed limit")
                break

        for i in reversed(jobs_to_remove):
            pending_jobs.pop(i)

        if pending_jobs:
            print(f"\n  {len(pending_jobs)} job arrays remaining to submit")
            print(f"  Waiting {check_interval} seconds before checking queue again...")
            print()
            time.sleep(check_interval)
        else:
            print("\n  All jobs submitted!")
            break

    print()
    print("=" * 70)
    print(f"Successfully submitted {len(submitted_jobs)} job arrays:")
    total_submitted_workers = 0
    for dir_name, job_id, n_workers in submitted_jobs:
        print(f"  {dir_name}: Job {job_id} ({n_workers} workers × 10 sims)")
        total_submitted_workers += n_workers
    print()
    print(f"Total workers submitted: {total_submitted_workers}")
    print(f"Total simulations covered: {total_submitted_workers * 10}")
    print()
    print("Monitor with: squeue -u $USER")
    print(f"Cancel all with: scancel -u $USER")
    print("=" * 70)
    print(f"Log: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Submit grouped SLURM array jobs (100 workers, 10 sims each) with queue monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be submitted
  python run_many_sbatches_grouped.py --dry-run

  # Submit with default settings
  python run_many_sbatches_grouped.py

  # More conservative queue limit, check every 2 minutes
  python run_many_sbatches_grouped.py --max-queued 50 --check-interval 120
        """
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory containing SIM_DIR folders (default: current directory)"
    )
    parser.add_argument(
        "--max-queued",
        type=int,
        default=9500,
        help="Maximum number of jobs to have queued at once (default: 9500)"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Seconds between queue status checks (default: 60)"
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=100,
        help="Time limit per worker in minutes — should cover 10 simulations (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without actually submitting"
    )

    args = parser.parse_args()

    submit_jobs_with_queue_monitoring(
        args.base_dir,
        max_queued=args.max_queued,
        check_interval=args.check_interval,
        dry_run=args.dry_run,
        time_limit=args.time_limit,
    )
