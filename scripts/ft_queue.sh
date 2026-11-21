#!/usr/bin/env bash
# Sequential fine-tune queue: each line of the queue file is
#   NAME|MODE(NPS/NPT)|WARM_START|extra --set args...
# Trains with recipes/training/HeFESTo{,Mars}CtsFineTune.yaml, saving temp_models/<NAME>.tar,
# then grid-tests it against the shipped model (and any REF models) on both test sets.
# Lines are consumed one at a time so the queue can be edited while running.
set -u
cd /home/bthyer/nGibbs
PY=/home/bthyer/miniconda3/envs/torch-env/bin/python
L=logs/ft_runs
Q=${1:-$L/queue.txt}
mkdir -p $L
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
  echo "$line" >> $L/started.txt
  # MODE: NPS/NPT (Earth affinity ReSamp data) or Mars_NPS/Mars_NPT (Mars data)
  case "$MODE" in
    Mars_*) P=${MODE#Mars_}; TAR=HeFESTo_Mars_${P}; CFG=recipes/training/HeFESToMarsCtsFineTune.yaml
            SHIP=src/ngibbs/engine/TrainedModels/HeFESToMars/HeFESToMars_${P}.tar
            OLDTEST=src/ngibbs/engine/TrainedModels/HeFESToMars/HeFESToMars_${P}_Test_subset15000.tar.gz
            NEWTEST=data/MLready/HeFESTo/HeFESTo_Mars_${P}_Test.tar.gz ;;
    MELTS_*) N=${MODE#MELTS_}; S=${N:0:3}   # e.g. MELTS_102Closed_Cr_NPT
            TAR=$(echo $N | sed -E 's/(Closed|Open)_/\1FxBatch_/'); CFG=recipes/training/MELTSft_${N}.yaml
            SHIP=src/ngibbs/engine/TrainedModels/${S}/${N}.tar
            OLDTEST=src/ngibbs/engine/TrainedModels/${S}/${N}_Test_subset15000.tar.gz
            NEWTEST=data/MLready/${S}/${TAR}_Test.tar.gz ;;
    *)      TAR=HeFESTo_EarthAffinityReSamp_${MODE}; CFG=recipes/training/HeFESToCtsFineTune.yaml
            SHIP=src/ngibbs/engine/TrainedModels/HeFESToEarth/HeFESToEarth_${MODE}.tar
            OLDTEST=src/ngibbs/engine/TrainedModels/HeFESToEarth/HeFESToEarth_${MODE}_Test_subset15000.tar.gz
            NEWTEST=data/MLready/HeFESTo/HeFESTo_EarthAffinity_${MODE}_Test.tar.gz ;;
  esac
  # pause (don't fail) while / or /tmp is short on space; /tmp is a small RAM tmpfs
  while [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -lt 50 ] || \
        [ "$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9)" -lt 2 ]; do
    echo "[queue] $(date) low disk space (/ <50G or /tmp <2G), waiting before $NAME"; sleep 600
  done
  LOG=$L/${NAME}.log
  echo "[queue] $(date) START $NAME ($TAR from $WARM) $EXTRA"
  if [ "$WARM" != "SKIP" ]; then
    eval $PY src/builder/training/main.py train --config $CFG \
      --tarname $TAR --warm_start $WARM --set train6.DictFilePath=$NAME $EXTRA > $LOG 2>&1
    echo "[queue] $(date) train exit $? for $NAME"
  fi
  M=src/builder/training/temp_models/${NAME}.tar
  if [ ! -f "$M" ]; then echo "[queue] no checkpoint for $NAME (no improvement?)"; continue; fi
  REFS=$(cat $L/refs_${MODE}.txt 2>/dev/null)
  $PY scripts/grid_deployment_test.py --models $M $REFS $SHIP --datasets $NEWTEST $OLDTEST \
      --max-samples 15000 --device cuda --no-plots --out grid_test_out/ft_${NAME} > $L/${NAME}_grid.log 2>&1
  echo "[queue] $(date) grid exit $? for $NAME"
  python3 scripts/compare_grid_f1.py grid_test_out/ft_${NAME} > $L/${NAME}_f1.txt 2>&1
  echo "[queue] $(date) DONE $NAME"
done
