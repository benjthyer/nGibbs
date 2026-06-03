#!/bin/bash
# prepare_simulations.sh demo - Run this from login node

python $HOME/HeFESTo/nMELTS/scripts/prepare_hefesto_tree_fulladiabat.py \
    --directory $HOME/HeFESTo/DemoSimulations \
    --georoc-dir $HOME/HeFESTo/nMELTS/data/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv \
    --control-path $HOME/HeFESTo/nMELTS/src/builder/alphamelts/batch/shallowHeFESTo \
    --n 10

echo "Simulation tree prepared. Now submit the array job with: sbatch run_hefesto_array.slurm"