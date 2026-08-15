#!/bin/bash
#SBATCH --job-name=hefesto_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hefesto_prep.out
#SBATCH --error=logs/hefesto_prep.err

# Preparations
mkdir -p logs
source $HOME/HeFESTo/Benv/bin/activate
export LD_LIBRARY_PATH=$HOME/HeFESTo/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/HeFESTo/lib64:$LIBRARY_PATH
export OPENBLAS_NUM_THREADS=1

python HeFESTo/nGibbs/scripts/prepare_hefesto_tree_fine.py --directory /resnick/groups/asimowgroup/thyer/HeFESTo/Training081026_EarthAdiabats_finesample/ --control-dir HeFESTo/nGibbs/src/builder/HeFESTo/batch/shallowHeFESTo --phase-path /resnick/groups/asimowgroup/thyer/HeFESTo/Training081026_EarthAdiabatsPhaseBoundsFine.csv --limit 110000 --deep --deep-dt 0.01  
python HeFESTo/nGibbs/scripts/prepare_hefesto_tree_fine.py --directory /resnick/groups/asimowgroup/thyer/HeFESTo/Training081026_EarthAdiabats_finesample/ --control-dir HeFESTo/nGibbs/src/builder/HeFESTo/batch/shallowHeFESTo --phase-path /resnick/groups/asimowgroup/thyer/HeFESTo/Training081026_EarthAdiabatsPhaseBoundsFine2.csv --limit 40000 --deep --deep-dt 0.01

python run_many_sbatches_grouped.py --base-dir /resnick/groups/asimowgroup/thyer/HeFESTo/Training081026_EarthAdiabats_finesample/ --time-limit 20