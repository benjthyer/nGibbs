#!/usr/bin/env bash
# Sequential fine-tune queue: each line of the queue file is
#   NAME|MODE(NPS/NPT)|WARM_START|extra --set args...
# Trains with recipes/training/HeFESTo{,Mars}CtsFineTune.yaml, saving temp_models/<NAME>.tar,
# then grid-tests it against the shipped model (and any REF models) on both test sets.
# Lines are consumed one at a time so the queue can be edited while running.
set -u
cd /home/bthyer/nGibbs
PY=/home/bthyer/miniconda3/envs/torch-env/bin/python
Q=${1:-ft_runs/queue.txt}
mkdir -p ft_runs
while true; do
  line=$(grep -v '^\s*#' "$Q" | grep -v '^\s*$' | head -1)
  [ -z "$line" ] && { echo "[queue] empty, exiting $(date)"; break; }
  # pop the line
  python3 - "$Q" "$line" <<'PY'
import sys
q, line = sys.argv[1], sys.argv[2]
lines = open(q).read().splitlines()
lines.remove(line)
open(q, 'w').write('\n'.join(lines) + '\n')
PY
  IFS='|' read -r NAME MODE WARM EXTRA <<< "$line"
  echo "$line" >> ft_runs/started.txt
  # MODE: NPS/NPT (Earth affinity ReSamp data) or Mars_NPS/Mars_NPT (Mars data)
  case "$MODE" in
    Mars_*) P=${MODE#Mars_}; TAR=HeFESTo_Mars_${P}; CFG=recipes/training/HeFESToMarsCtsFineTune.yaml
            SHIP=src/ngibbs/engine/TrainedModels/HeFESToMars/HeFESToMars_${P}.tar
            OLDTEST=src/ngibbs/engine/TrainedModels/HeFESToMars/HeFESToMars_${P}_Test_subset15000.tar.gz
            NEWTEST=data/MLready/HeFESTo/HeFESTo_Mars_${P}_Test.tar.gz ;;
    *)      TAR=HeFESTo_EarthAffinityReSamp_${MODE}; CFG=recipes/training/HeFESToCtsFineTune.yaml
            SHIP=src/ngibbs/engine/TrainedModels/HeFESToEarth/HeFESToEarth_${MODE}.tar
            OLDTEST=src/ngibbs/engine/TrainedModels/HeFESToEarth/HeFESToEarth_${MODE}_Test_subset15000.tar.gz
            NEWTEST=data/MLready/HeFESTo/HeFESTo_EarthAffinity_${MODE}_Test.tar.gz ;;
  esac
  LOG=ft_runs/${NAME}.log
  echo "[queue] $(date) START $NAME ($TAR from $WARM) $EXTRA"
  if [ "$WARM" != "SKIP" ]; then
    eval $PY src/builder/training/main.py train --config $CFG \
      --tarname $TAR --warm_start $WARM --set train6.DictFilePath=$NAME $EXTRA > $LOG 2>&1
    echo "[queue] $(date) train exit $? for $NAME"
  fi
  M=src/builder/training/temp_models/${NAME}.tar
  if [ ! -f "$M" ]; then echo "[queue] no checkpoint for $NAME (no improvement?)"; continue; fi
  REFS=$(cat ft_runs/refs_${MODE}.txt 2>/dev/null)
  $PY scripts/grid_deployment_test.py --models $M $REFS $SHIP --datasets $NEWTEST $OLDTEST \
      --max-samples 15000 --device cuda --no-plots --out grid_test_out/ft_${NAME} > ft_runs/${NAME}_grid.log 2>&1
  echo "[queue] $(date) grid exit $? for $NAME"
  python3 scripts/compare_grid_f1.py grid_test_out/ft_${NAME} > ft_runs/${NAME}_f1.txt 2>&1
  echo "[queue] $(date) DONE $NAME"
done
