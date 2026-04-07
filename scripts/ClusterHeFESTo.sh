#!/bin/bash
#SBATCH --job-name=hefesto_arrayDemo
#SBATCH --array=1-5              # One task per simulation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                    # Adjust based on your needs
#SBATCH --time=00:10:00             # Adjust based on runtime per task
#SBATCH --output=logs/hefesto_%A_%a.out
#SBATCH --error=logs/hefesto_%A_%a.err

# Make sure log directory exists
mkdir -p logs

# Activate your Python environment
source $HOME/HeFESTo/Benv/bin/activate

# Load required modules
#module load cmake/3.27.9
#module load openblas

# Set library paths for HeFESTo
#export LD_LIBRARY_PATH=$HOME/HeFESTo/lib64:$LD_LIBRARY_PATH
#export LIBRARY_PATH=$HOME/HeFESTo/lib64:$LIBRARY_PATH

# Navigate to this simulation's directory
SIM_DIR=$HOME/HeFESTo/DemoSimulations/Simulation${SLURM_ARRAY_TASK_ID}
cd $SIM_DIR

# Run HeFESTo for this simulation
echo "Running Simulation${SLURM_ARRAY_TASK_ID} in $(pwd)"
$HOME/HeFESTo/HeFESToRepository/main
#python -c "from HeFESTo import main; main()"

# Or if HeFESTo.main takes arguments:
# python -m HeFESTo --input ad.in --output results.dat

echo "Simulation${SLURM_ARRAY_TASK_ID} complete"