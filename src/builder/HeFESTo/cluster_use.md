---Cluster workflow--

Load environment

source HeFESTo/Benv/bin/activate

prepare HeFESTo sims NOT IN $HOME BUT IN /resnick/groups/asimowgroup/thyer !! 
I have 50 GB  storage, this folder has 2TB. 

python $HOME/HeFESTo/nGibbs/scripts/prepare_hefesto_tree_fulladiabat.py     
    --directory /resnick/groups/asimowgroup/thyer/HeFESTo/Training061126_temp/
    --georoc-dir $HOME/HeFESTo/nGibbs/data/GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv     
    --control-path $HOME/HeFESTo/nGibbs/src/builder/alphamelts/batch/shallowHeFESTo     
    --n 3

# Can use run_sbatches.py for less than 10K jobs. But run_many_sbatches.py is general. 
# Both search recursively within the specified directory and run EVERY HEFESTO SIM, **even those that have already been run**

python HeFESTo/nGibbs/scripts/run_many_sbatches.py 
    --base-dir /resnick/groups/asimowgroup/thyer/HeFESTo/Training061126_temp/
    --time-limit 10 # minutes


# Note that to run HeFESTo on the login node (for testing) you will need to add nlopt to the path. Better to test with above functions and small n

(wait until jobs finish)

Read and trim results, build phase change csv for grid resampling. 

python $HOME/HeFESTo/nGibbs/scripts/import_hefesto_subdirs 
    --root  /resnick/groups/asimowgroup/thyer/HeFESTo/Training061126_temp/
    --dataname $HOME/HeFESTo/Mars_profiles.csv
    --phase-change-dataname /resnick/groups/asimowgroup/thyer/HeFESTo/Mars_profiles_phasechanges.csv

Build HeFESTo calls for resampling

python $HOME/HeFESTo/nGibbs/scripts/prepare_hefesto_tree_from_phase_changes.py     
    --directory /resnick/groups/asimowgroup/thyer/HeFESTo/Training061126_phasechanges/
    --phase-path /resnick/groups/asimowgroup/thyer/HeFESTo/Mars_profiles_phasechanges.csv
    --control-dir $HOME/HeFESTo/nGibbs/src/builder/alphamelts/batch/shallowHeFESTo     
    --limit 200000

    parser.add_argument(
        '--directory',
        type=Path,
        required=True,
        help='Base output directory where Batch#### folders will be created.',
    )
    parser.add_argument(
        '--phase-path',
        type=Path,
        required=True,
        help='Path to the phase-boundary CSV with paired rows.',
    )
    parser.add_argument(
        '--control-dir',
        type=Path,
        required=True,
        help=(
            'Path to control template directory (or a template file path). '
            'If a directory is given, shallowHeFESTo/deepHeFESTo is preferred.'

python HeFESTo/nGibbs/scripts/run_many_sbatches.py 
    --base-dir /resnick/groups/asimowgroup/thyer/HeFESTo/Training061126_temp/
    --time-limit 10 # minutes
