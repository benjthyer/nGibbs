#!/usr/bin/env python3
"""
Submit SLURM array jobs for all simulation directories under BASE_DIR.

Each SIM_DIR containing SimulationN folders gets its own array job.
"""

import os
import subprocess
import sys
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


def create_slurm_script(sim_dir: Path, n_simulations: int) -> Path:
    """Create a SLURM script for this specific SIM_DIR."""
    slurm_script = sim_dir / "run_hefesto_array.slurm"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=hefesto_{sim_dir.name}
#SBATCH --array=1-{n_simulations}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
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


def submit_jobs(base_dir: Path, dry_run: bool = False):
    """
    Find all SIM_DIR directories and submit array jobs.
    
    Args:
        base_dir: Base directory containing SIM_DIR folders
        dry_run: If True, print what would be submitted without actually submitting
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
    
    print(f"Found {len(sim_dirs)} directories with simulations:")
    for sim_dir, n_sims in sim_dirs:
        print(f"  {sim_dir.name}: {n_sims} simulations")
    
    print()
    
    # Submit jobs for each directory
    job_ids = []
    for sim_dir, n_sims in sim_dirs:
        print(f"Processing {sim_dir.name} ({n_sims} simulations)...")
        
        # Create SLURM script
        slurm_script = create_slurm_script(sim_dir, n_sims)
        print(f"  Created: {slurm_script}")
        
        # Submit job
        if dry_run:
            print(f"  [DRY RUN] Would submit: sbatch {slurm_script}")
        else:
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
                job_ids.append((sim_dir.name, job_id))
                print(f"  ✓ Submitted job {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to submit: {e.stderr}")
        
        print()
    
    if not dry_run and job_ids:
        print("=" * 60)
        print("All jobs submitted successfully:")
        for dir_name, job_id in job_ids:
            print(f"  {dir_name}: {job_id}")
        print()
        print("Monitor with: squeue -u $USER")
        print(f"Cancel all with: scancel -u $USER")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Submit SLURM array jobs for HeFESTo simulation directories"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory containing SIM_DIR folders (default: current directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without actually submitting"
    )
    
    args = parser.parse_args()
    
    submit_jobs(args.base_dir, dry_run=args.dry_run)