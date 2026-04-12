#!/usr/bin/env python3
"""
Submit SLURM array jobs for all simulation directories under BASE_DIR.

Monitors queue size and submits new jobs as capacity becomes available.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def count_simulations(sim_dir: Path) -> int:
    """Count how many Simulation<N> directories exist in sim_dir."""
    sim_folders = list(sim_dir.glob("Simulation*"))
    # Filter to only numeric suffixes
    valid_sims = [
        s for s in sim_folders 
        if s.name.replace("Simulation", "").isdigit()
    ]
    return len(valid_sims)


def get_queued_jobs_count() -> int:
    """Get the current number of queued/running jobs for this user."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ["USER"], "-h"],
            capture_output=True,
            text=True,
            check=True
        )
        # Count non-empty lines
        job_lines = [line for line in result.stdout.strip().split('\n') if line]
        return len(job_lines)
    except subprocess.CalledProcessError:
        print("Warning: Could not query job queue, assuming 0 jobs")
        return 0

def create_slurm_script(sim_dir: Path, n_simulations: int, time_limit: int) -> Path:
    """Create a SLURM script for this specific SIM_DIR."""
    slurm_script = sim_dir / "run_hefesto_array.slurm"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=hefesto_{sim_dir.name}
#SBATCH --array=1-{n_simulations}
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

# Navigate to this simulation's directory
ACTIVE_DIR=Simulation${{SLURM_ARRAY_TASK_ID}}
cd $ACTIVE_DIR

$HOME/HeFESTo/HeFESToRepository/main
"""
    
    slurm_script.write_text(script_content)
    slurm_script.chmod(0o755)  # Make executable
    
    return slurm_script


def submit_job(sim_dir: Path, slurm_script: Path) -> str | None:
    """
    Submit a single SLURM job.
    
    Returns job_id on success, None on failure.
    """
    try:
        result = subprocess.run(
            ["sbatch", str(slurm_script)],
            cwd=sim_dir,
            capture_output=True,
            text=True,
            check=True
        )
        # Extract job ID from output like "Submitted batch job 12345"
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
    time_limit: int = 10
):
    """
    Submit jobs while respecting queue limits.
    
    Args:
        base_dir: Base directory containing SIM_DIR folders
        max_queued: Maximum number of jobs to have queued at once
        check_interval: How often to check queue status (seconds)
        dry_run: If True, print what would be submitted without actually submitting
        time_limit: Time limit for each job in minutes
    """
    base_dir = base_dir.resolve()
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        sys.exit(1)
    
    # Find all directories that contain Simulation* subdirectories
    sim_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir():
            n_sims = count_simulations(item)
            if n_sims > 0:
                sim_dirs.append((item, n_sims))
    
    if not sim_dirs:
        print(f"No directories with Simulation* folders found in {base_dir}")
        sys.exit(1)
    
    # Calculate total simulations
    total_sims = sum(n for _, n in sim_dirs)
    
    print("=" * 70)
    print(f"Found {len(sim_dirs)} directories with {total_sims} total simulations:")
    for sim_dir, n_sims in sim_dirs:
        print(f"  {sim_dir.name}: {n_sims} simulations")
    print(f"\nQueue limit: {max_queued} jobs")
    print(f"Check interval: {check_interval} seconds")
    print("=" * 70)
    print()
    
    # Prepare all SLURM scripts first
    pending_jobs = []
    for sim_dir, n_sims in sim_dirs:
        slurm_script = create_slurm_script(sim_dir, n_sims, time_limit=time_limit)
        pending_jobs.append((sim_dir, slurm_script, n_sims))
        print(f"Prepared: {slurm_script}")
    
    print(f"\n{len(pending_jobs)} job arrays ready to submit")
    print()
    
    if dry_run:
        print("[DRY RUN MODE - Not actually submitting]")
        for sim_dir, slurm_script, n_sims in pending_jobs:
            print(f"Would submit: {sim_dir.name} ({n_sims} tasks)")
        return
    
    # Submit jobs iteratively
    submitted_jobs = []
    
    while pending_jobs:
        # Check current queue status
        current_queued = get_queued_jobs_count()
        print(f"[{time.strftime('%H:%M:%S')}] Current jobs in queue: {current_queued}/{max_queued}")
        
        # Submit as many jobs as we can
        submitted_this_round = 0
        jobs_to_remove = []
        
        for i, (sim_dir, slurm_script, n_sims) in enumerate(pending_jobs):
            # Check if we can submit this array
            if current_queued + n_sims <= max_queued:
                print(f"  Submitting {sim_dir.name} ({n_sims} tasks)...", end=" ")
                job_id = submit_job(sim_dir, slurm_script)
                
                if job_id:
                    submitted_jobs.append((sim_dir.name, job_id, n_sims))
                    jobs_to_remove.append(i)
                    current_queued += n_sims
                    submitted_this_round += 1
                    print(f"✓ Job {job_id}")
                else:
                    print("✗ Failed")
                    # Don't increment current_queued on failure
            else:
                # Would exceed limit, stop trying to submit more this round
                print(f"  Cannot submit {sim_dir.name} ({n_sims} tasks) - would exceed limit")
                break
        
        # Remove submitted jobs from pending list (in reverse to preserve indices)
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
    
    # Summary
    print()
    print("=" * 70)
    print(f"Successfully submitted {len(submitted_jobs)} job arrays:")
    total_submitted_tasks = 0
    for dir_name, job_id, n_tasks in submitted_jobs:
        print(f"  {dir_name}: Job {job_id} ({n_tasks} tasks)")
        total_submitted_tasks += n_tasks
    print()
    print(f"Total tasks submitted: {total_submitted_tasks}")
    print()
    print("Monitor with: squeue -u $USER")
    print(f"Cancel all with: scancel -u $USER")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Submit SLURM array jobs with queue monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be submitted
  python submit_all_simulations.py --dry-run
  
  # Submit with default settings (max 9500 jobs)
  python submit_all_simulations.py
  
  # More conservative limit, check every 2 minutes
  python submit_all_simulations.py --max-queued 5000 --check-interval 120
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
        default=10,
        help="Time limit for each job in minutes (default: 10)"
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
        time_limit=args.time_limit
    )